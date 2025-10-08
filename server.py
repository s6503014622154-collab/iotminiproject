# server.py  (insightface + Telegram + LINE Messaging Push + Email TEXT ONLY + Google Sheet) รักจุนซี่  
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np, cv2, os, requests, uuid

# ========== EMAIL (SMTP, TEXT ONLY) ==========
import smtplib, ssl
from email.message import EmailMessage

# NEW: ใช้สำหรับทำ timestamp และทำ raw_json
from datetime import datetime, timezone
import json

# --- EMAIL: กรอกค่าของคุณตรงนี้ ---
SMTP_HOST = "smtp.gmail.com"        # Gmail
SMTP_PORT = 465                     # ใช้ SSL ตรง ๆ
SMTP_USER = "p50802.2013@gmail.com" # อีเมลผู้ส่ง (login)
SMTP_PASS = "zddvsrhefdkijvfe"      # Gmail App Password (16 ตัว)
EMAIL_FROM = SMTP_USER              # ผู้ส่ง
EMAIL_TO   = "s6503014622154@email.kmutnb.ac.th"  # ปลายทาง (คั่นด้วย , ได้หลายคน)

def email_enabled() -> bool:
    return all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_FROM, EMAIL_TO])

def send_email_text(subject: str, body: str):
    """ส่งอีเมล 'ข้อความล้วน' (ไม่แนบรูป)"""
    if not email_enabled():
        return False, "Email not configured"
    recipients = [x.strip() for x in EMAIL_TO.split(",") if x.strip()]
    if not recipients:
        return False, "No recipients"

    msg = EmailMessage()
    msg["Subject"] = subject or "ESP32-CAM"
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)
    msg.set_content(body or "-")

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, str(e)
# ============================================

from insightface.app import FaceAnalysis

# ===== Telegram =====
BOT_TOKEN = "8241449093:AAFy4qxa4Ixtt-GTCw4aMsWvkK28sxt6cwY"
CHAT_ID   = "6151938406"

# ===== LINE Messaging API (Push) =====
LINE_CHANNEL_ACCESS_TOKEN = "tVH1IOew+14Sl7Sx/VdmslPNY3dTATeDKUHOOvl6HI2wy2Vb2L8nlyoBktAc0RU2YCa2GvrogTaYdT+iaf/dlyTJ1HkKKf6qtn1ZhqWKTiZ3DmCsp//m+FyDHX6A/meWcJY8aZlJV1tcjstt0ZgCYQdB04t89/1O/w1cDnyilFU="
LINE_USER_ID              = "U1ee052554c4960ad31556ab8f379ba2e"

# URL สาธารณะ (ใส่ใน env: PUBLIC_BASE_URL หรือแก้ตรงนี้ก็ได้)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
# PUBLIC_BASE_URL = "https://xxxx.ngrok-free.dev"

def line_enabled() -> bool:
    return bool(LINE_CHANNEL_ACCESS_TOKEN and LINE_USER_ID)

def line_push_text(message: str):
    if not line_enabled():
        return False, "LINE not configured"
    url = "https://api.line.me/v2/bot/message/push"
    headers = { "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                "Content-Type": "application/json" }
    payload = { "to": LINE_USER_ID,
                "messages": [ { "type": "text", "text": message or "-" } ] }
    r = requests.post(url, headers=headers, json=payload, timeout=15)
    return r.ok, r.text

def line_push_image_urls(original_url: str, preview_url: str):
    if not line_enabled():
        return False, "LINE not configured"
    url = "https://api.line.me/v2/bot/message/push"
    headers = { "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                "Content-Type": "application/json" }
    payload = { "to": LINE_USER_ID,
                "messages": [ {
                    "type": "image",
                    "originalContentUrl": original_url,
                    "previewImageUrl": preview_url
                } ] }
    r = requests.post(url, headers=headers, json=payload, timeout=15)
    return r.ok, r.text

# เก็บรูปลง /static -> ให้ LINE / Sheets โหลดผ่าน PUBLIC_BASE_URL/static/<fname>
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

def save_image_to_static(jpeg_bytes: bytes) -> str:
    fname = f"{uuid.uuid4().hex}.jpg"
    path  = os.path.join(STATIC_DIR, fname)
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return fname

# ===== NEW: Google Sheets (ผ่าน Apps Script Web App) =====
# ใส่ URL ที่ได้จาก Deploy Web App และ secret ให้ตรงกับ Apps Script
GSCRIPT_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbyjMXGNoGszh-mK9J7wCljCP-qArtssstwRcL3SOIfc0OyfffqJoHC1lTsqeHIgCxNcew/exec"
GSCRIPT_TOKEN      = "abc123"

def gsheet_enabled() -> bool:
    return bool(GSCRIPT_WEBAPP_URL and GSCRIPT_WEBAPP_URL.startswith("http"))

def gsheet_submit(record: dict):
    """ส่ง JSON ไป Apps Script เพื่อแปะลงชีต"""
    if not gsheet_enabled():
        return False, "gsheet disabled"
    payload = dict(record)
    if GSCRIPT_TOKEN:
        payload["token"] = GSCRIPT_TOKEN
    try:
        r = requests.post(GSCRIPT_WEBAPP_URL, json=payload, timeout=12)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

# ===== โปรไฟล์ & ตัวกรอง (โฟลเดอร์ DB แนะนำอังกฤษล้วน) =====
PROFILES = {
    "junsii": {"name": "จุนซี่", "age": 20, "status": "โสด", "height_cm": 175, "weight_kg": 62},
    "beam":   {"name": "เบงตาด", "age": 22, "status": "โสด", "height_cm": 171, "weight_kg": 90},
    "pet":    {"name": "เพชร",   "age": 23, "status": "โสด", "height_cm": 176, "weight_kg": 120},
}
DB_DIR    = "known_faces"     # known_faces/<person_id>/*.jpg
THRESHOLD = 0.35              # 0.30–0.45 (ต่ำ=ผ่อนปรน)
ALLOWED_SEND_IDS = {"junsii", "beam", "pet"}   # ว่าง set() = ส่งทุกคน
LOG_UNKNOWN      = False

app = FastAPI(title="ESP32-CAM Face Server (insightface)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- โมเดล ----------
app_face = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

def caption_for(pid: str) -> str:
    p = PROFILES.get(pid, {"name": pid})
    return (f"เจอ: {p['name']} "
            f"อายุ {p.get('age','?')} "
            f"สถานะ {p.get('status','?')} "
            f"สูง {p.get('height_cm','?')} ซม. "
            f"น้ำหนัก {p.get('weight_kg','?')} กก. "
            f"{p.get('extra','')}").strip()

def email_body_for(pid: str, sim: float) -> str:
    p = PROFILES.get(pid, {})
    return (
        f"id: {pid}\n"
        f"name: {p.get('name', pid)}\n"
        f"age: {p.get('age', '?')}\n"
        f"status: {p.get('status', '?')}\n"
        f"height_cm: {p.get('height_cm', '?')}\n"
        f"weight_kg: {p.get('weight_kg', '?')}\n"
        f"sim: {sim:.3f}"
    )

def telegram_send_photo(jpeg_bytes: bytes, caption: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    r = requests.post(
        url,
        data={"chat_id": CHAT_ID, "caption": caption},
        files={"photo": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        timeout=15,
    )
    return r.ok, r.text

# ---------- ดัชนีฐานข้อมูล ----------
index_vecs, index_labels = [], []

def build_index():
    global index_vecs, index_labels
    index_vecs, index_labels = [], []
    if not os.path.isdir(DB_DIR):
        index_vecs = np.zeros((0, 512), dtype=np.float32); index_labels = []
        return
    for person in os.listdir(DB_DIR):
        pdir = os.path.join(DB_DIR, person)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            path = os.path.join(pdir, fn)
            img = cv2.imread(path)
            if img is None:
                print("skip (read fail):", path); continue
            faces = app_face.get(img)
            if not faces:
                print("skip (no face):", path); continue
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            index_vecs.append(f.normed_embedding.astype(np.float32))
            index_labels.append(person)
    index_arr = np.array(index_vecs, dtype=np.float32) if len(index_vecs)>0 else np.zeros((0,512), dtype=np.float32)
    globals()["index_vecs"] = index_arr
    print(f"[index] loaded vectors: {index_arr.shape[0]}")

build_index()

# ---------- core logic ----------
def recognize_bytes(img_bytes: bytes):
    npimg = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return {"match": False, "reason": "decode_fail"}

    faces = app_face.get(img)
    if not faces:
        return {"match": False, "reason": "no_face"}

    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    q = f.normed_embedding

    if index_vecs.shape[0] == 0:
        return {"match": False, "reason": "empty_index"}

    sims = (index_vecs @ q)
    best_i = int(np.argmax(sims))
    best_sim = float(sims[best_i])

    # ---------- unknown ----------
    if best_sim < THRESHOLD:
        if LOG_UNKNOWN:
            caption = f"🚨 Unknown (sim={best_sim:.3f})"
            t_ok, t_resp = telegram_send_photo(img_bytes, caption)

            # LINE
            l_ok, l_resp = (False, "")
            image_url = ""
            if line_enabled():
                if PUBLIC_BASE_URL:
                    fname = save_image_to_static(img_bytes)
                    image_url = f"{PUBLIC_BASE_URL}/static/{fname}"
                    l_ok, l_resp = line_push_image_urls(image_url, image_url)
                    line_push_text(caption)
                else:
                    l_ok, l_resp = line_push_text(caption)

            # EMAIL (ข้อความล้วน)
            e_ok, e_resp = send_email_text(
                subject="[ESP32-CAM] Unknown person",
                body=f"Unknown person detected\nsim: {best_sim:.3f}"
            ) if email_enabled() else (False, "")

            # SHEETS (Apps Script)
            record = {
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "person_id": "",
                "name": "Unknown",
                "similarity": f"{best_sim:.3f}",
                "age": "", "status": "", "height_cm": "", "weight_kg": "",
                "image_url": image_url,
                "raw_json": json.dumps({"reason":"unknown","sim":best_sim}, ensure_ascii=False)
            }
            gs_ok, gs_resp = gsheet_submit(record) if gsheet_enabled() else (False, "")

            return {"match": False, "sim": best_sim,
                    "telegram": t_ok, "t_resp": t_resp,
                    "line": l_ok, "l_resp": l_resp,
                    "email": e_ok, "e_resp": e_resp,
                    "gsheet": gs_ok, "gs_resp": gs_resp}
        return {"match": False, "sim": best_sim}

    # ---------- recognized ----------
    person_id = index_labels[best_i]
    caption = caption_for(person_id)

    if ALLOWED_SEND_IDS and (person_id not in ALLOWED_SEND_IDS):
        return {"match": True, "id": person_id, "sim": best_sim,
                "telegram": False, "line": False, "email": False, "filtered": True}

    # Telegram (ส่งรูป)
    t_ok, t_resp = telegram_send_photo(img_bytes, caption)

    # LINE
    l_ok, l_resp = (False, "")
    image_url = ""
    if line_enabled():
        if PUBLIC_BASE_URL:
            fname = save_image_to_static(img_bytes)
            image_url = f"{PUBLIC_BASE_URL}/static/{fname}"
            l_ok, l_resp = line_push_image_urls(image_url, image_url)
            line_push_text(caption)
        else:
            l_ok, l_resp = line_push_text(caption + "\n(ไม่มี PUBLIC_BASE_URL เลยส่งเป็นข้อความ)")

    # EMAIL (ข้อความล้วน)
    e_ok, e_resp = (False, "")
    if email_enabled():
        body_text = email_body_for(person_id, best_sim)
        e_ok, e_resp = send_email_text(
            subject=f"[ESP32-CAM] {PROFILES.get(person_id, {}).get('name', person_id)}",
            body=body_text
        )

    # SHEETS (Apps Script)
    p = PROFILES.get(person_id, {})
    record = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "person_id": person_id,
        "name": p.get("name", person_id),
        "similarity": f"{best_sim:.3f}",
        "age": p.get("age",""),
        "status": p.get("status",""),
        "height_cm": p.get("height_cm",""),
        "weight_kg": p.get("weight_kg",""),
        "image_url": image_url,  # ว่างได้ถ้าไม่มี PUBLIC_BASE_URL
        "raw_json": json.dumps({"id":person_id,"sim":best_sim}, ensure_ascii=False)
    }
    gs_ok, gs_resp = gsheet_submit(record) if gsheet_enabled() else (False, "")

    return {"match": True, "id": person_id, "sim": best_sim,
            "telegram": t_ok, "t_resp": t_resp,
            "line": l_ok, "l_resp": l_resp,
            "email": e_ok, "e_resp": e_resp,
            "gsheet": gs_ok, "gs_resp": gs_resp}

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "indexed": int(index_vecs.shape[0]),
        "allowed_send_ids": list(ALLOWED_SEND_IDS),
        "threshold": THRESHOLD,
        "line_enabled": line_enabled(),
        "public_base_url": PUBLIC_BASE_URL or "",
        "email_enabled": email_enabled(),
        "smtp_host": SMTP_HOST,
        "gsheet_enabled": gsheet_enabled(),
    }

@app.post("/reload_index")
def reload_index():
    build_index()
    return {"ok": True, "indexed": int(index_vecs.shape[0])}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    img_bytes = await file.read()
    return JSONResponse(recognize_bytes(img_bytes))

@app.post("/recognize_url")
def recognize_url(url: str = Body(..., embed=True)):
    try:
        r = requests.get(url, timeout=10, headers={"Cache-Control": "no-cache"})
        if r.status_code != 200 or not r.content:
            return JSONResponse({"match": False, "reason": f"http {r.status_code}"})
        return JSONResponse(recognize_bytes(r.content))
    except Exception as e:
        return JSONResponse({"match": False, "error": str(e)})

@app.post("/snap_to_telegram")
def snap_to_telegram(url: str = Body(..., embed=True),
                     caption: str = Body("ESP32-CAM snapshot", embed=True)):
    try:
        r = requests.get(url, timeout=10, headers={"Cache-Control": "no-cache"})
        if r.status_code != 200 or not r.content:
            return {"ok": False, "reason": f"http {r.status_code}"}
        ok, resp = telegram_send_photo(r.content, caption)
        return {"ok": ok, "resp": resp, "bytes": len(r.content)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/snap_to_line")
def snap_to_line(url: str = Body(..., embed=True),
                 caption: str = Body("ESP32-CAM snapshot", embed=True)):
    if not line_enabled():
        return {"ok": False, "error": "LINE not configured"}
    try:
        r = requests.get(url, timeout=10, headers={"Cache-Control": "no-cache"})
        if r.status_code != 200 or not r.content:
            return {"ok": False, "reason": f"http {r.status_code}"}
        if PUBLIC_BASE_URL:
            fname = save_image_to_static(r.content)
            public_url = f"{PUBLIC_BASE_URL}/static/{fname}"
            ok, resp = line_push_image_urls(public_url, public_url)
            line_push_text(caption)
            return {"ok": ok, "resp": resp, "url": public_url}
        else:
            ok, resp = line_push_text(caption + "\n(ไม่มี PUBLIC_BASE_URL เลยส่งเป็นข้อความ)")
            return {"ok": ok, "resp": resp}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/snap_to_email")
def snap_to_email(url: str = Body(..., embed=True),
                  subject: str = Body("ESP32-CAM snapshot", embed=True),
                  body: str = Body("", embed=True)):
    if not email_enabled():
        return {"ok": False, "error": "Email not configured"}
    try:
        r = requests.get(url, timeout=10, headers={"Cache-Control": "no-cache"})
        if r.status_code != 200 or not r.content:
            return {"ok": False, "reason": f"http {r.status_code}"}
        ok, resp = send_email_text(subject=subject, body=body or "Snapshot captured.")
        return {"ok": ok, "resp": resp}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
