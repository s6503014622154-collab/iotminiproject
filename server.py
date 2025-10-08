# server.py  (insightface + Telegram + LINE Push + Email TEXT ONLY + Google Sheet via Service Account)
# Log to Google Sheet ONLY when a known person is recognized.
# Columns: timestamp, name, "อายุ X ปี", "สถานะ Y", "ส่วนสูง H ซม", "น้ำหนัก W กก"

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np, cv2, os, requests, uuid

# ========== EMAIL (SMTP, TEXT ONLY) ==========
import smtplib, ssl
from email.message import EmailMessage

# Utility
from datetime import datetime, timezone
import json

# ========== Google Sheets (Service Account, direct API) ==========
# pip install google-api-python-client google-auth google-auth-httplib2
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ======= CONFIG (แนะนำตั้งผ่าน ENV เวลาขึ้นโปรดักชัน) =======
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "p50802.2013@gmail.com")
SMTP_PASS = os.getenv("SMTP_PASS", "zddvsrhefdkijvfe")  # Gmail App Password
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO   = os.getenv("EMAIL_TO", "s6503014622154@email.kmutnb.ac.th")  # คั่น , ได้หลายคน

BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "8241449093:AAFy4qxa4Ixtt-GTCw4aMsWvkK28sxt6cwY")
CHAT_ID   = os.getenv("TG_CHAT_ID", "6151938406")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv(
    "LINE_CHANNEL_ACCESS_TOKEN",
    "tVH1IOew+14Sl7Sx/VdmslPNY3dTATeDKUHOOvl6HI2wy2Vb2L8nlyoBktAc0RU2YCa2GvrogTaYdT+iaf/dlyTJ1HkKKf6qtn1ZhqWKTiZ3DmCsp//m+FyDHX6A/meWcJY8aZlJV1tcjstt0ZgCYQdB04t89/1O/w1cDnyilFU=",
)
LINE_USER_ID = os.getenv("LINE_USER_ID", "U1ee052554c4960ad31556ab8f379ba2e")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()  # สำหรับ LINE รูป

# Google Sheets
SERVICE_ACCOUNT_FILE = os.getenv("GSA_KEY_FILE", "iotminiproject-474515-20048a3d1069.json")
SPREADSHEET_ID       = os.getenv("GSA_SHEET_ID", "1-GyT7N9Ip144ARdTvtvMSDcTB2t9-bYOojzwqmUO8IQ")
SHEET_NAME           = os.getenv("GSA_SHEET_NAME", "ชีต1")  # ชื่อแท็บ (ไม่ใช่ชื่อไฟล์)
SCOPES               = ["https://www.googleapis.com/auth/spreadsheets"]

# InsightFace / DB
from insightface.app import FaceAnalysis
PROFILES = {
    "junsii": {"name": "จุนซี่", "age": 20, "status": "โสด", "height_cm": 175, "weight_kg": 62},
    "beam":   {"name": "เบงตาด", "age": 22, "status": "โสด", "height_cm": 171, "weight_kg": 90},
    "pet":    {"name": "เพชร",   "age": 23, "status": "โสด", "height_cm": 176, "weight_kg": 120},
}
DB_DIR    = "known_faces"     # known_faces/<person_id>/*.jpg
THRESHOLD = 0.35              # 0.30–0.45 (ต่ำ=ผ่อนปรน)
ALLOWED_SEND_IDS = {"junsii", "beam", "pet"}   # ว่าง set() = ส่งทุกคน
LOG_UNKNOWN      = False       # ไม่บันทึก unknown

# ===== Static for public image (LINE preview) =====
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

def save_image_to_static(jpeg_bytes: bytes) -> str:
    fname = f"{uuid.uuid4().hex}.jpg"
    path  = os.path.join(STATIC_DIR, fname)
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return fname

# ========== Helpers ==========
def email_enabled() -> bool:
    return all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_FROM, EMAIL_TO])

def send_email_text(subject: str, body: str):
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

def telegram_send_photo(jpeg_bytes: bytes, caption: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    r = requests.post(
        url,
        data={"chat_id": CHAT_ID, "caption": caption},
        files={"photo": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        timeout=15,
    )
    return r.ok, r.text

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

# ========== Google Sheets writer (ONLY selected columns with units) ==========
def gsheet_enabled() -> bool:
    return os.path.exists(SERVICE_ACCOUNT_FILE) and bool(SPREADSHEET_ID) and bool(SHEET_NAME)

def _gsheet_service():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

def gsheet_submit_person(p: dict):
    """
    เขียน 1 แถว: timestamp, name,
                 "อายุ X ปี", "สถานะ Y", "ส่วนสูง H ซม", "น้ำหนัก W กก"
    """
    if not gsheet_enabled():
        return False, "sheets not configured"

    try:
        service = _gsheet_service()

        name   = p.get("name", "") or ""
        age    = p.get("age", "") or ""
        status = p.get("status", "") or ""
        height = p.get("height_cm", "") or ""
        weight = p.get("weight_kg", "") or ""

        age_text    = f"อายุ {age} ปี"      if str(age)    != "" else "อายุ -"
        status_text = f"สถานะ {status}"      if str(status) != "" else "สถานะ -"
        height_text = f"ส่วนสูง {height} ซม" if str(height) != "" else "ส่วนสูง -"
        weight_text = f"น้ำหนัก {weight} กก" if str(weight) != "" else "น้ำหนัก -"

        values = [[
            datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            name,
            age_text,
            status_text,
            height_text,
            weight_text,
        ]]

        body = {"values": values}
        result = service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{SHEET_NAME}!A:F",          # 6 คอลัมน์ตาม values ข้างบน
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body,
        ).execute()

        updated = result.get("updates", {}).get("updatedRows", 0)
        return True, f"appended_rows={updated}"
    except Exception as e:
        return False, str(e)

# ========== InsightFace ==========
app = FastAPI(title="ESP32-CAM Face Server (insightface)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app_face = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

def caption_for(pid: str) -> str:
    p = PROFILES.get(pid, {"name": pid})
    return (f"เจอ: {p['name']} "
            f"อายุ {p.get('age','?')} "
            f"สถานะ {p.get('status','?')} "
            f"สูง {p.get('height_cm','?')} ซม. "
            f"น้ำหนัก {p.get('weight_kg','?')} กก.").strip()

def email_body_for(pid: str, sim: float) -> str:
    p = PROFILES.get(pid, {})
    return (
        f"name: {p.get('name', pid)}\n"
        f"age: {p.get('age', '?')}\n"
        f"status: {p.get('status', '?')}\n"
        f"height_cm: {p.get('height_cm', '?')}\n"
        f"weight_kg: {p.get('weight_kg', '?')}\n"
        f"sim: {sim:.3f}"
    )

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
        return {"match": False, "sim": best_sim}

    # ---------- recognized ----------
    person_id = index_labels[best_i]
    caption = caption_for(person_id)

    # Filter: ส่งเฉพาะรายชื่อที่อนุญาต
    if ALLOWED_SEND_IDS and (person_id not in ALLOWED_SEND_IDS):
        return {"match": True, "id": person_id, "sim": best_sim, "filtered": True}

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
        e_ok, e_resp = send_email_text(
            subject=f"[ESP32-CAM] {PROFILES.get(person_id, {}).get('name', person_id)}",
            body=email_body_for(person_id, best_sim)
        )

    # SHEETS: เขียนเฉพาะข้อมูลคน (แบบมีหน่วยตามที่ป๋าขอ)
    p = PROFILES.get(person_id, {})
    gs_ok, gs_resp = (False, "")
    if gsheet_enabled():
        gs_ok, gs_resp = gsheet_submit_person({
            "name": p.get("name", person_id),
            "age": p.get("age",""),
            "status": p.get("status",""),
            "height_cm": p.get("height_cm",""),
            "weight_kg": p.get("weight_kg",""),
        })

    return {"match": True, "id": person_id, "sim": best_sim,
            "telegram": t_ok, "t_resp": t_resp,
            "line": l_ok, "l_resp": l_resp,
            "email": e_ok, "e_resp": e_resp,
            "gsheet": gs_ok, "gs_resp": gs_resp}

# ---------- Endpoints ----------
app = app  # already created

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
        "sheet_id": SPREADSHEET_ID[:6]+"..." if SPREADSHEET_ID else "",
        "sheet_name": SHEET_NAME,
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
