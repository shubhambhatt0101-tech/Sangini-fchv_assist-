"""
app.py  —  FCHV_assist · IMNCI Protocol Edition
======================================================
Integrates the complete FCHV Rule Engine (9 protocols) with:
  - FCHV Authentication (login, register, per-FCHV data isolation)
  - Groq Whisper STT  (online)
  - Gemini 2.5 Flash  (primary LLM — best Nepali clinical quality)
  - Groq LLaMA 3.3-70b (fallback LLM — free, fast)
  - Google Cloud TTS  (native ne-NP voice, sad/gentle tone)
  - ElevenLabs TTS    (fallback — multilingual)
  - gTTS              (offline fallback)
  - SVM triage        (always-on fallback)
  - Full IMNCI protocol decision tree (9 conditions)
  - Nepal hospital referral engine (25 facilities)
"""

import base64, hashlib, io, json, os, pickle, re, subprocess, tempfile
import time, unicodedata, uuid, wave
from collections import defaultdict
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY",      "AIzaSyCaF3LtXz6mhcyeGzCiil_QKyCOcjsVbh8")
GEMINI_MODEL        = "gemini-2.5-flash-preview-05-20"

GROQ_API_KEY        = os.environ.get("GROQ_API_KEY",        "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")
GROQ_MODEL          = "llama-3.3-70b-versatile"
GROQ_STT_MODEL      = "whisper-large-v3"

ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY",  "sk_78c3f4d63973da5a985b423b3ffee20e83aec0c58f1b09ab")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "XB0fDUnXU5powFXDhCwa")  # Charlotte — soft/gentle
ELEVENLABS_MODEL    = "eleven_multilingual_v2"

GOOGLE_TTS_API_KEY  = os.environ.get("GOOGLE_TTS_API_KEY",  "")   # optional

OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY",      "")
OPENAI_TTS_VOICE    = "nova"
OPENAI_TTS_MODEL    = "tts-1"

PIPER_NEPALI_MODEL  = "ne_NP-google-medium"
PIPER_ENGLISH_MODEL = "en_US-lessac-medium"
WHISPER_MODEL       = "medium"

MIN_ROOT_LEN        = 2
TRAINING_DIR        = "training_data"
LOG_FILE            = os.path.join(TRAINING_DIR, "training_log.json")
VOICES_DIR          = os.path.join("static", "voices")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "hvn-secret-2026-change-this")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Path(TRAINING_DIR).mkdir(parents=True, exist_ok=True)
Path(VOICES_DIR).mkdir(parents=True, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


# ─────────────────────────────────────────────────────────────
# FCHV DATA PATHS  (per-FCHV isolated files)
# ─────────────────────────────────────────────────────────────
FCHV_DIR   = os.path.join(TRAINING_DIR, "fchv_accounts")
Path(FCHV_DIR).mkdir(parents=True, exist_ok=True)

FCHV_INDEX = os.path.join(FCHV_DIR, "fchv_index.json")   # list of all FCHVs

if not os.path.exists(FCHV_INDEX):
    # Seed three demo accounts
    demo = [
        {"fchv_id": "FCHV-001", "name": "सुनिता थापा",   "phone": "9841000001", "district": "Dhankuta", "vdc": "Hile VDC",       "ward": "3", "pwd_hash": "", "registered_at": ""},
        {"fchv_id": "FCHV-002", "name": "रमिला श्रेष्ठ", "phone": "9841000002", "district": "Dhankuta", "vdc": "Pakhribas VDC",  "ward": "1", "pwd_hash": "", "registered_at": ""},
        {"fchv_id": "FCHV-003", "name": "गीता पौडेल",    "phone": "9841000003", "district": "Dhankuta", "vdc": "Dhankuta Bazar", "ward": "5", "pwd_hash": "", "registered_at": ""},
    ]
    passwords = {"FCHV-001": "sunita123", "FCHV-002": "ramila123", "FCHV-003": "geeta123"}
    for d in demo:
        d["pwd_hash"] = hashlib.sha256(passwords[d["fchv_id"]].encode()).hexdigest()
        d["registered_at"] = datetime.utcnow().isoformat() + "Z"
        fid = d["fchv_id"]
        for fname in ["patients.json", "sessions.json", "log.json"]:
            p = os.path.join(FCHV_DIR, fid, fname)
            Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)
            if not os.path.exists(p):
                with open(p, "w", encoding="utf-8") as f:
                    json.dump([], f)
    with open(FCHV_INDEX, "w", encoding="utf-8") as f:
        json.dump(demo, f, ensure_ascii=False, indent=2)


def load_fchv_index():
    try:    return json.load(open(FCHV_INDEX, encoding="utf-8"))
    except: return []

def save_fchv_index(lst):
    with open(FCHV_INDEX, "w", encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)

def fchv_data_path(fchv_id, filename):
    """Returns path to a per-FCHV data file and ensures directory exists."""
    p = os.path.join(FCHV_DIR, fchv_id, filename)
    Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump([], f)
    return p

def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# AUTH DECORATOR
# ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "fchv_id" not in session:
            wants_json = (
                request.method != "GET"
                or request.is_json
                or request.headers.get("X-Requested-With") == "XMLHttpRequest"
                or "application/json" in request.headers.get("Accept", "")
            )
            if wants_json:
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────
# IMNCI PROTOCOL ENGINE
# ─────────────────────────────────────────────────────────────

IMNCI_PROTOCOLS = {
    "danger_signs": {
        "label": "खतराका संकेतहरू",
        "label_en": "Danger Signs",
        "age_group": "दुई महिनादेखि पाँच वर्ष",
        "steps": [
            {"id": "s1", "q": "के नानीबाबुले दूध चुस्न वा केही पनि खान छोडेका छन्?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "feeding_stopped",
                     "advice": "यो खतराको संकेत हो। खान छोड्नु धेरै गम्भीर कुरा हो। ढिला नगरी तुरुन्तै अस्पताल लैजानुहोस्।"}},
            {"id": "s2", "q": "के नानीबाबुको शरीर काँपेको वा छारे रोग जस्तो झट्का आएको छ?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "convulsions",
                     "advice": "जीउ काँप्नु भनेको दिमाग वा कडा संक्रमणको लक्षण हो। तुरुन्तै आकस्मिक उपचारको लागि अस्पताल लैजानुहोस्।"}},
            {"id": "s3", "q": "के नानीबाबु एकदमै सुस्त, लल्याक-लुलुक परेका वा बेहोस जस्तै छन्?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "lethargy",
                     "advice": "यसरी सुस्त हुनु ज्यानको जोखिम हुनु हो। एक मिनेट पनि ढिला नगरी अस्पताल पुर्‍याउनुहोस्।"}},
            {"id": "s4", "q": "के नानीबाबुले खाएको सबै कुरा वान्ता गर्छन्?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "vomiting_all",
                     "advice": "केही पनि नपच्ने गरी वान्ता हुनु गम्भीर अवस्था हो। तुरुन्तै डाक्टरलाई देखाइहाल्नुहोस्।"}},
        ],
        "default": {"action": "continue", "risk": "low", "label": "no_danger_sign", "advice": "अहिले कुनै खतराको लक्षण देखिएन। जाँच जारी राख्नुहोस्।"}
    },
    "neonatal_sepsis": {
        "label": "नवजात शिशुको जाँच",
        "label_en": "Neonatal Sepsis",
        "age_group": "जन्मदेखि अन्ठाउन्न दिनसम्म",
        "steps": [
            {"id": "n1", "q": "के सानो नानीले दूध राम्ररी चुस्न सकेको छैन वा खानै मानेको छैन?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "feeding_difficulty",
                     "advice": "सानो नानीले दूध खान नसक्नु एकदमै गम्भीर कुरा हो। तुरुन्तै अस्पताल लैजानुहोस्।"}},
            {"id": "n2", "q": "के नानीको शरीर काँपेको छ?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "neonatal_convulsions",
                     "advice": "यो टाउको वा रगतको कडा संक्रमण हुन सक्छ। तुरुन्तै अस्पताल पुर्‍याउनुहोस्।"}},
            {"id": "n3", "q": "के नानीले एक मिनेटमा साठी पटक भन्दा धेरै छिटो-छिटो सास फेरेको छ?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "fast_breathing_neo",
                     "advice": "सास फेर्न गाह्रो हुनु गम्भीर समस्या हो। तुरुन्तै स्वास्थ्य संस्था लैजानुहोस्।"}},
            {"id": "n4", "q": "के नानी एकदमै सुस्त वा बेहोस जस्तो छ?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "neonatal_lethargy",
                     "advice": "नानीको अवस्था नाजुक छ। अस्पताल लैजान ढिला नगर्नुहोस्।"}},
            {"id": "n5", "q": "के नानीको नाभि रातो भएको, सुन्निएको वा पीप जस्तो आएको छ?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "umbilical_infection",
                     "advice": "नाभिमा संक्रमण भएको छ। यो फैलन सक्छ, तुरुन्तै जाँच गराउनुहोस्।"}},
            {"id": "n6", "q": "के नानीको जिउ वा आँखा पहेंलो देखिएको छ?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "neonatal_jaundice",
                     "advice": "यो जन्डिस हो। साना नानीहरूमा यो खतरनाक हुन सक्छ, स्वास्थ्यकर्मीलाई देखाइहाल्नुहोस्।"}},
        ],
        "default": {"action": "routine_care", "risk": "low", "label": "healthy_neonate",
                    "advice": "अहिले खतरा छैन। नानीलाई न्यानो राख्नुहोस् र नियमित दूध चुसाउनुहोस्। तीन दिनपछि फेरि जचाउनुहोस्।"}
    },
    "diarrhea": {
        "label": "झाडापखाला",
        "label_en": "Diarrhea",
        "age_group": "दुई महिनादेखि पाँच वर्ष",
        "steps": [
            {"id": "d1", "q": "के दिसामा रगत देखिएको छ?",
             "yes": {"action": "refer", "risk": "high", "label": "dysentery",
                     "advice": "यो डिसेनंटरि परेको हुन सक्छ। तुरुन्तै स्वास्थ्य संस्थामा जानुहोस्।"}},
            {"id": "d2", "q": "के नानीबाबु धेरै तिर्खाएका छन् वा आँखा भित्र भासिएका छन्?",
             "no": {"action": "home_care", "risk": "low", "label": "no_dehydration",
                    "advice": "शरीरमा पानीको कमी देखिएको छैन। जीवनजल, झोलिलो खानेकुरा र जिन्क चक्की दिनुहोस्।"}},
            {"id": "d3", "q": "पेटको छाला चिमोटेर छोड्दा बिस्तारै फर्किन्छ?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "severe_dehydration",
                     "advice": "शरीरमा पानीको धेरै कमी भइसक्यो। अस्पताल लैजाँदै गर्दा बाटोमा जीवनजल खुवाइरहनुहोस्।"},
             "no": {"action": "treat_and_refer", "risk": "medium", "label": "some_dehydration",
                    "advice": "थोरै पानीको कमी देखिएको छ। जीवनजल दिनुहोस् र स्वास्थ्य चौकीमा देखाउनुहोस्।"}},
        ],
        "default": {"action": "home_care", "risk": "low", "label": "simple_diarrhea",
                    "advice": "घरमै जीवनजल र प्रशस्त झोलिलो खाना दिनुहोस्। सुधार नभए जचाउनुहोस्।"}
    },
    "fever": {
        "label": "ज्वरो",
        "label_en": "Fever",
        "age_group": "दुई महिनादेखि पाँच वर्ष",
        "steps": [
            {"id": "f1", "q": "के शरीरमा राता डाबर वा दादुरा जस्ता बिमिरा आएका छन्?",
             "yes": {"action": "refer", "risk": "medium", "label": "measles",
                     "advice": "यो दादुरा हुन सक्छ। भिटामिन ए दिनुहोस् र स्वास्थ्य संस्थामा लैजानुहोस्।"}},
            {"id": "f2", "q": "के नानीबाबुलाई निकै जाडो हुने र पसिना आउने गरी ज्वरो आउँछ?",
             "yes": {"action": "test_and_refer", "risk": "medium", "label": "malaria_suspected",
                     "advice": "मलेरियाको शंका छ। रगत परीक्षण गर्न स्वास्थ्य संस्था लैजानुहोस्।"}},
            {"id": "f3", "q": "के नानीबाबुको घाँटी वा कान दुखेको छ?",
             "yes": {"action": "refer", "risk": "medium", "label": "throat_ear_infection",
                     "advice": "घाँटी वा कानको संक्रमण हुन सक्छ। स्वास्थ्यकर्मीलाई देखाएर उपचार सुरु गर्नुहोस्।"}},
        ],
        "default": {"action": "home_care", "risk": "low", "label": "simple_fever",
                    "advice": "साधारण ज्वरो हो। आराम गराउनुहोस् र सिटामोल दिनुहोस्। दुई दिनसम्म निको नभए जचाउनुहोस्।"}
    },
    "cough": {
        "label": "खोकी वा सास फेर्न गाह्रो",
        "label_en": "Cough / ARI",
        "age_group": "दुई महिनादेखि पाँच वर्ष",
        "steps": [
            {"id": "c1", "q": "सास फेर्दा नानीबाबुको छाती भित्र पसेको देखिन्छ?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "severe_pneumonia",
                     "advice": "यो कडा निमोनिया हो। छाती तान्नु एकदमै खतराको संकेत हो। तुरुन्तै अस्पताल लैजानुहोस्।"}},
            {"id": "c2", "q": "के नानीबाबुले छिटो-छिटो सास फेरेका छन्?",
             "yes": {"action": "refer", "risk": "high", "label": "pneumonia",
                     "advice": "निमोनिया भएको हुन सक्छ। तुरुन्तै स्वास्थ्य संस्था लैजानुहोस्। एन्टिबायोटिक ओखती आवश्यक पर्न सक्छ।"}},
            {"id": "c3", "q": "के नानीबाबुले सास फेर्दा घर्र-घर्र आवाज आउँछ?",
             "yes": {"action": "refer", "risk": "medium", "label": "wheezing",
                     "advice": "श्वासप्रश्वासमा अवरोध भएको हुन सक्छ। स्वास्थ्य संस्थामा जाँच गराउनुहोस्।"}},
        ],
        "default": {"action": "home_care", "risk": "low", "label": "common_cold",
                    "advice": "साधारण रुघाखोकी मात्र हो। तातो झोलिलो कुरा खुवाउनुहोस् र नानीबाबुलाई न्यानो राख्नुहोस्।"}
    },
    "ear_problem": {
        "label": "कानको समस्या",
        "label_en": "Ear Problem",
        "age_group": "सबै उमेर",
        "steps": [
            {"id": "e1", "q": "के नानीबाबुको कान दुखेको छ?",
             "no": {"action": "no_issue", "risk": "low", "label": "no_ear_issue", "advice": "कानमा केही समस्या देखिएन।"}},
            {"id": "e2", "q": "के कानबाट पीप वा पानी बगिरहेको छ?",
             "yes": {"action": "refer", "risk": "medium", "label": "ear_infection",
                     "advice": "कान पाकेको वा संक्रमण भएको हुन सक्छ। स्वास्थ्य संस्थामा गएर देखाउनुहोस्।"}},
            {"id": "e3", "q": "के कानको पछाडिको हड्डी सुन्निएको वा दुखेको छ?",
             "yes": {"action": "refer_urgently", "risk": "high", "label": "mastoiditis_suspected",
                     "advice": "यो हड्डी सम्म पुग्ने कडा संक्रमण हुन सक्छ। तुरुन्तै अस्पताल लैजानुहोस्।"}},
        ],
        "default": {"action": "home_care", "risk": "low", "label": "ear_pain_only",
                    "advice": "अहिलेलाई कान दुखाइ मात्र देखिन्छ। कान सफा राख्नुहोस् र ध्यान दिनुहोस्।"}
    },
    "malnutrition": {
        "label": "पोषणको अवस्था",
        "label_en": "Malnutrition",
        "age_group": "छ महिनादेखि पाँच वर्ष",
        "steps": [
            {"id": "m1", "q": "के नानीबाबुका दुवै खुट्टा सुन्निएका छन्?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "kwashiorkor",
                     "advice": "यो गम्भीर कुपोषण हो। तुरुन्तै पोषण सुधार केन्द्र वा अस्पताल पठाउनुहोस्।"}},
            {"id": "m2", "q": "के पाखुराको नाप रातो रङमा छ?",
             "yes": {"action": "refer_urgently", "risk": "critical", "label": "SAM",
                     "advice": "गम्भीर कुपोषण भएको छ। बच्चा निकै कमजोर छ। तुरुन्तै अस्पताल लैजानुहोस् र पोषिलो जाउलो खुवाउनुहोस्।"}},
            {"id": "m3", "q": "के पाखुराको नाप पहेंलो रङमा छ?",
             "yes": {"action": "treat_and_refer", "risk": "medium", "label": "MAM",
                     "advice": "मध्यम खालको कुपोषण देखिएको छ। पोषिलो खानेकुरा खुवाउनुहोस् र स्वास्थ्य संस्थामा सम्पर्क गर्नुहोस्।"}},
        ],
        "default": {"action": "routine_care", "risk": "low", "label": "normal_nutrition",
                    "advice": "पोषणको अवस्था राम्रो छ। नियमित पोषिलो खाना खुवाउन जारी राख्नुहोस्।"}
    },
    "eye_problem": {
        "label": "आँखाको समस्या",
        "label_en": "Eye Problem",
        "age_group": "सबै उमेर",
        "steps": [
            {"id": "ey1", "q": "के आँखा रातो छ वा आँखाबाट पीप र कचेरा आउँछ?",
             "yes": {"action": "refer", "risk": "medium", "label": "conjunctivitis",
                     "advice": "यो आँखा पाकेको हुन सक्छ। आँखाको मलम लगाउनुहोस् र स्वास्थ्यकर्मीलाई जचाउनुहोस्।"}},
            {"id": "ey2", "q": "के नानीबाबुले धमिलो देख्छन् वा राति कम देख्छन्?",
             "yes": {"action": "refer", "risk": "medium", "label": "vision_problem",
                     "advice": "दृष्टिमा समस्या हुन सक्छ। भिटामिन ए खुवाउनुहोस् र आँखा अस्पतालमा चेकजाँच गराउनुहोस्।"}},
        ],
        "default": {"action": "no_issue", "risk": "low", "label": "no_eye_issue", "advice": "आँखामा कुनै समस्या देखिएन।"}
    },
    "skin_problem": {
        "label": "छालाको समस्या",
        "label_en": "Skin Problem",
        "age_group": "सबै उमेर",
        "steps": [
            {"id": "sk1", "q": "के छालामा घाउ-खटिरा आएर पाकेको छ?",
             "yes": {"action": "refer", "risk": "medium", "label": "skin_infection",
                     "advice": "छालाको संक्रमण भएको छ। डाक्टरलाई देखाएर ओखती वा मलम लिनुहोस्।"}},
            {"id": "sk2", "q": "के शरीरमा निकै चिलाउने साना बिमिरा वा लुतो छ?",
             "yes": {"action": "treat_and_refer", "risk": "low", "label": "scabies",
                     "advice": "यो लुतो हुन सक्छ। औषधि लगाउनुहोस् र परिवारका सबैले सरसफाइमा ध्यान दिनुहोस्।"}},
        ],
        "default": {"action": "home_care", "risk": "low", "label": "minor_skin_issue",
                    "advice": "सामान्य छालाको समस्या हो। सफा राख्नुहोस् र केही दिन निगरानी गर्नुहोस्।"}
    },
}

CONDITION_KEYWORDS = {
    "दिसा": "diarrhea", "झाडापखाला": "diarrhea", "पखाला": "diarrhea",
    "ज्वरो": "fever", "बुखार": "fever", "तातो": "fever",
    "खोकी": "cough", "सास": "cough", "निमोनिया": "cough",
    "कान": "ear_problem", "कान दुख": "ear_problem",
    "कुपोषण": "malnutrition", "पाखुरा": "malnutrition", "MUAC": "malnutrition",
    "आँखा": "eye_problem", "दृष्टि": "eye_problem",
    "छाला": "skin_problem", "खटिरा": "skin_problem",
    "नवजात": "neonatal_sepsis", "नवजात शिशु": "neonatal_sepsis",
    "नाभि": "neonatal_sepsis", "जन्डिस": "neonatal_sepsis",
}

ACTION_META = {
    "home_care":       {"label": "🏠 घरमै हेरचाह",       "color": "low",      "en": "Home Care"},
    "monitor":         {"label": "👀 निगरानी गर्नुहोस्",  "color": "low",      "en": "Monitor"},
    "treat_and_refer": {"label": "💊 उपचार गरी पठाउनुहोस्", "color": "medium", "en": "Treat & Refer"},
    "refer":           {"label": "🏥 स्वास्थ्य संस्था",   "color": "medium",   "en": "Refer"},
    "refer_urgently":  {"label": "🚨 तुरुन्त अस्पताल!",   "color": "critical", "en": "Emergency Referral"},
    "test_and_refer":  {"label": "🔬 परीक्षण गर्नुहोस्",  "color": "medium",   "en": "Test & Refer"},
    "routine_care":    {"label": "✅ नियमित हेरचाह",      "color": "low",      "en": "Routine Care"},
    "no_issue":        {"label": "✅ समस्या छैन",          "color": "low",      "en": "No Issue"},
    "continue":        {"label": "➡️  अर्को जाँच",        "color": "low",      "en": "Continue Assessment"},
}

def detect_condition_from_text(text: str) -> str | None:
    for kw, condition in CONDITION_KEYWORDS.items():
        if kw in text:
            return condition
    return None

def run_protocol_from_answers(condition: str, answers: dict) -> dict:
    proto = IMNCI_PROTOCOLS.get(condition)
    if not proto:
        return {"error": f"Unknown condition: {condition}"}
    steps_taken = []
    for step in proto["steps"]:
        sid = step["id"]
        user_ans = answers.get(sid)
        steps_taken.append({"step": sid, "question": step["q"], "answer": user_ans})
        if user_ans is True and "yes" in step:
            result = step["yes"]
            meta = ACTION_META.get(result["action"], ACTION_META["refer"])
            return {"condition": condition, "condition_label": proto["label"],
                    "condition_label_en": proto["label_en"], "result": result,
                    "action_meta": meta, "steps_taken": steps_taken, "protocol": proto}
        if user_ans is False and "no" in step:
            result = step["no"]
            meta = ACTION_META.get(result["action"], ACTION_META["home_care"])
            return {"condition": condition, "condition_label": proto["label"],
                    "condition_label_en": proto["label_en"], "result": result,
                    "action_meta": meta, "steps_taken": steps_taken, "protocol": proto}
    result = proto["default"]
    meta = ACTION_META.get(result["action"], ACTION_META["home_care"])
    return {"condition": condition, "condition_label": proto["label"],
            "condition_label_en": proto["label_en"], "result": result,
            "action_meta": meta, "steps_taken": steps_taken, "protocol": proto}

def get_protocol_next_question(condition: str, answered: dict) -> dict | None:
    proto = IMNCI_PROTOCOLS.get(condition)
    if not proto:
        return None
    for step in proto["steps"]:
        sid = step["id"]
        if sid not in answered:
            return {"step_id": sid, "question": step["q"],
                    "condition": condition, "condition_label": proto["label"]}
    return None


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
IMNCI_SUMMARY = """
IMNCI Protocol Context (Nepal):
• Danger Signs (any age): stop feeding, convulsions, lethargy, vomiting everything → URGENT referral
• Neonatal (0–59 days): poor feeding, fast breathing (>60/min), jaundice, umbilical infection → URGENT
• Diarrhea: blood in stool=dysentery/refer; skin pinch slow=severe dehydration/urgent; thirsty+sunken eyes=some dehydration
• Fever: rash=measles/Vit A; chills+sweating=malaria RDT; >7 days=refer; ear/throat pain=infection
• Cough: chest indrawing=severe pneumonia/urgent; fast breathing=pneumonia/refer; >21 days=TB screen
• Malnutrition: edema=Kwashiorkor/urgent; MUAC<11.5cm=SAM/RUTF; MUAC 11.5–12.5=MAM/OTP
• Ear: discharge>14d=chronic/ENT; discharge<14d=acute; mastoid pain=urgent
Risk: critical/high → refer_urgently | medium → refer/monitor | low → home_care
"""

SYSTEM_PROMPT = f"""तपाईं एक अनुभवी नेपाली स्वास्थ्य सहायक हुनुहुन्छ।

{IMNCI_SUMMARY}

व्याकरण नियमहरू (अनिवार्य):
- सधैं शुद्ध नेपाली लेख्नुहोस् — रोमन वा हिन्दी मिसाउनु हुँदैन
- क्रियाको अन्त्य: "छ", "छन्", "हुन्छ", "गर्नुहोस्", "लैजानुहोस्"
- सम्बोधन: "नानीबाबु", "आमा", "बुवा" — "बच्चा" वा "child" होइन
- वाक्य छोटो र स्पष्ट राख्नुहोस्

उदाहरण राम्रो जवाफ:
प्रश्न: "बच्चालाई ज्वरो छ"
जवाफ: "नानीबाबुलाई ज्वरो आएको छ भने पहिले निधार र काखीमा भिजेको कपडा राख्नुहोस्। दुई दिनभित्र निको नभए नजिकको स्वास्थ्य चौकीमा लैजानुहोस्।"

उदाहरण राम्रो जवाफ:
प्रश्न: "झाडापखाला लागेको छ"  
जवाफ: "झाडापखाला लाग्दा शरीरमा पानीको कमी हुन्छ। नानीबाबुलाई जीवनजल र धेरै झोलिलो कुरा खुवाउनुहोस्। दिसामा रगत देखिए तुरुन्तै अस्पताल लैजानुहोस्।"

नियमहरू:
- जवाफ २-३ वाक्यमा सीमित राख्नुहोस्
- माया र सहानुभूतिका साथ बोल्नुहोस्
- औषधिको डोज कहिल्यै नभन्नुहोस्
- उच्च जोखिममा तुरुन्त अस्पताल पठाउनुहोस्"""

TRIAGE_RESPONSES = {
    "low":    "तपाईंको नानीको लक्षण कम जोखिमको देखिन्छ। आराम गराउनुहोस्, प्रशस्त पानी पिउन दिनुहोस्, र नियमित निगरानी जारी राख्नुहोस्।",
    "medium": "तपाईंको नानीको लक्षण मध्यम जोखिमको छ। कृपया नजिकको स्वास्थ्य केन्द्रमा लैजानुहोस्।",
    "high":   "उच्च जोखिम देखियो। अहिले नै अस्पताल लैजानुहोस् र तुरुन्त चिकित्सक सहायता लिनुहोस्।",
}


# ─────────────────────────────────────────────────────────────
# INTERNET / CONNECTIVITY
# ─────────────────────────────────────────────────────────────
def is_online() -> bool:
    import socket
    for host in ("generativelanguage.googleapis.com", "api.groq.com", "api.elevenlabs.io"):
        try:
            socket.setdefaulttimeout(2)
            socket.create_connection((host, 443))
            return True
        except Exception:
            continue
    return False

def detect_language(text: str) -> str:
    if not text: return "ne"
    dev = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total = len([c for c in text if c.strip()])
    return "ne" if (total == 0 or dev / total > 0.3) else "en"


# ─────────────────────────────────────────────────────────────
# AUDIO CONVERSION
# ─────────────────────────────────────────────────────────────
def convert_to_wav(audio_bytes: bytes, src_fmt: str = "webm") -> tuple:
    src_path = wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{src_fmt}", delete=False) as src:
            src.write(audio_bytes); src_path = src.name
        wav_path = src_path.replace(f".{src_fmt}", ".wav")
        cmd = ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path]
        extra = {"creationflags": 0x08000000} if os.name == "nt" else {}
        r = subprocess.run(cmd, capture_output=True, timeout=30, **extra)
        if r.returncode == 0 and os.path.exists(wav_path):
            with open(wav_path, "rb") as f: wav_bytes = f.read()
            return wav_bytes, "wav"
    except Exception as e:
        print(f"⚠️  ffmpeg: {e}")
    finally:
        for p in [src_path, wav_path]:
            try:
                if p and os.path.exists(p): os.unlink(p)
            except: pass
    return audio_bytes, src_fmt


# ─────────────────────────────────────────────────────────────
# LLM — GEMINI 2.5 FLASH (primary) + GROQ FALLBACK
# ─────────────────────────────────────────────────────────────
def _build_proto_context(protocol_result: dict | None) -> str:
    if not protocol_result or "result" not in protocol_result:
        return ""
    r = protocol_result["result"]
    return (
        f"\n\nIMNCI Protocol Result: {protocol_result.get('condition_label', '')} "
        f"→ {r.get('label', '')} [{r.get('risk', '').upper()}]"
        f"\nRecommended action: {r.get('advice', '')}"
    )

def _build_user_message(text: str, svm_pred: str, protocol_result: dict | None) -> str:
    proto_ctx = _build_proto_context(protocol_result)
    return (
        f"प्रश्न: {text}\n"
        f"SVM triage: '{svm_pred.upper()}' जोखिम।{proto_ctx}\n\n"
        f"IMNCI protocol र उपरोक्त जानकारीको आधारमा नेपालीमा स्पष्ट स्वास्थ्य सल्लाह दिनुहोस् — "
        f"2-3 वाक्य। आमाबाबुलाई माया र सहानुभूतिका साथ बोल्नुहोस्।"
    )


def gemini_answer(text: str, svm_pred: str, history: list, protocol_result: dict | None = None) -> str:
    """Primary LLM — Gemini 2.5 Flash via REST (no SDK needed)."""
    import requests

    user_msg = _build_user_message(text, svm_pred, protocol_result)

    contents = []
    for h in history[-6:]:
        role = "user" if h.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [{"text": h.get("content", "")}]})
    contents.append({"role": "user", "parts": [{"text": user_msg}]})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": 350,
            "topP": 0.9,
        },
    }

    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}",
        json=payload,
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from e


def groq_answer(text: str, svm_pred: str, history: list, protocol_result: dict | None = None) -> str:
    """Fallback LLM — Groq LLaMA 3.3-70b-versatile."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": _build_user_message(text, svm_pred, protocol_result)})
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.35,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()


def get_ai_answer(text: str, svm_pred: str, history: list, protocol_result: dict | None = None) -> tuple:
    """Try Gemini first → fallback to Groq. Returns (answer_text, provider_name)."""
    if GEMINI_API_KEY:
        try:
            answer = gemini_answer(text, svm_pred, history, protocol_result)
            return answer, "gemini"
        except Exception as e:
            print(f"⚠️  Gemini failed ({e}), falling back to Groq…")

    if GROQ_API_KEY:
        try:
            answer = groq_answer(text, svm_pred, history, protocol_result)
            return answer, "groq"
        except Exception as e:
            print(f"⚠️  Groq also failed: {e}")

    return TRIAGE_RESPONSES.get(svm_pred, TRIAGE_RESPONSES["medium"]), "offline"


# ─────────────────────────────────────────────────────────────
# STT  — Groq Whisper
# ─────────────────────────────────────────────────────────────
def groq_stt(audio_bytes: bytes, fmt: str = "wav") -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    mime_map = {
        "wav": "audio/wav", "mp3": "audio/mp3", "webm": "audio/webm",
        "ogg": "audio/ogg", "m4a": "audio/mp4", "flac": "audio/flac",
    }
    mime = mime_map.get(fmt, "audio/wav")
    t = client.audio.transcriptions.create(
        file=(f"audio.{fmt}", audio_bytes, mime),
        model=GROQ_STT_MODEL,
        language="ne",
        response_format="text",
        temperature=0.0,
    )
    return t.strip() if isinstance(t, str) else str(t).strip()


# ─────────────────────────────────────────────────────────────
# TTS  — Google Cloud ne-NP → ElevenLabs → gTTS
# ─────────────────────────────────────────────────────────────
def google_tts_nepali(text: str) -> bytes | None:
    """Google Cloud TTS — native ne-NP-Standard-A voice, empathetic tone."""
    if not GOOGLE_TTS_API_KEY:
        return None
    try:
        import requests
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": "ne-NP",
                "name": "ne-NP-Standard-A",
                "ssmlGender": "FEMALE",
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": 0.87,
                "pitch": -3.0,
                "volumeGainDb": 1.0,
            },
        }
        r = requests.post(
            f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}",
            json=payload,
            timeout=12,
        )
        if r.status_code == 200:
            audio_b64 = r.json().get("audioContent", "")
            if audio_b64:
                return base64.b64decode(audio_b64)
        else:
            print(f"⚠️  Google TTS HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"⚠️  Google TTS: {e}")
    return None


def elevenlabs_tts(text: str) -> bytes | None:
    """ElevenLabs — Charlotte voice, tuned for soft/empathetic tone."""
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY.startswith("YOUR_"):
        return None
    try:
        import requests
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": ELEVENLABS_MODEL,
                "voice_settings": {
                    "stability": 0.75,
                    "similarity_boost": 0.65,
                    "style": 0.45,
                    "use_speaker_boost": False,
                },
            },
            timeout=15,
        )
        if r.status_code == 200:
            return r.content
        else:
            print(f"⚠️  ElevenLabs HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"⚠️  ElevenLabs: {e}")
    return None


def gtts_fallback(text: str, lang: str = "ne") -> bytes | None:
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"⚠️  gTTS: {e}")
    return None


_piper_available = None
def _check_piper() -> bool:
    global _piper_available
    if _piper_available is not None: return _piper_available
    try: import piper; _piper_available = True
    except ImportError: _piper_available = False
    return _piper_available

def piper_tts(text: str, lang: str = "ne") -> bytes | None:
    if not _check_piper(): return None
    return None


def text_to_speech(text: str, online: bool = True) -> tuple:
    """
    TTS priority (Nepali):
      1. Google Cloud TTS ne-NP
      2. ElevenLabs Charlotte
      3. gTTS
    TTS priority (English):
      1. ElevenLabs
      2. gTTS
    """
    lang = detect_language(text)
    if online:
        if lang == "ne":
            a = google_tts_nepali(text)
            if a: return a, "google_ne_NP"
            a = elevenlabs_tts(text)
            if a: return a, "elevenlabs_ne"
            a = gtts_fallback(text, "ne")
            if a: return a, "gtts_ne"
        else:
            a = elevenlabs_tts(text)
            if a: return a, "elevenlabs_en"
            a = gtts_fallback(text, "en")
            if a: return a, "gtts_en"
    else:
        a = piper_tts(text, lang)
        if a: return a, "piper_offline"
        a = gtts_fallback(text, lang)
        if a: return a, "gtts_fallback"
    return None, "none"


# ─────────────────────────────────────────────────────────────
# OFFLINE VOICES
# ─────────────────────────────────────────────────────────────
def _check_voices():
    found = set()
    for cls in ("low", "medium", "high"):
        p = os.path.join(VOICES_DIR, f"{cls}.mp3")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            found.add(cls)
    return found

VOICES_AVAILABLE = _check_voices()


# ─────────────────────────────────────────────────────────────
# LOAD WHISPER (local fallback)
# ─────────────────────────────────────────────────────────────
print(f"Loading Whisper '{WHISPER_MODEL}'…")
try:
    import whisper as _whisper
    whisper_model = _whisper.load_model(WHISPER_MODEL)
    print("✅  Whisper ready.")
except ImportError:
    raise SystemExit("❌  pip install openai-whisper")

def whisper_transcribe(audio_bytes: bytes, fmt: str = "webm") -> str:
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name
    try:
        r = whisper_model.transcribe(
            tmp_path, language="ne", task="transcribe",
            fp16=False, verbose=False, beam_size=5, best_of=5,
            temperature=0.0, condition_on_previous_text=False,
        )
        return r.get("text", "").strip()
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────
# LOAD SVM + NLP
# ─────────────────────────────────────────────────────────────
def _load(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path): raise FileNotFoundError(f"Missing: {path}")
    return path

print("Loading SVM…")
svm_model    = pickle.load(open(_load("svm_model.pkl"), "rb"))
norm_map     = json.load(open(_load("normalization_map.json"), encoding="utf-8"))
stopword_set = set(json.load(open(_load("normalized_stopwords.json"), encoding="utf-8"))["stopwords"])
suffix_list  = json.load(open(_load("normalized_suffixes.json"), encoding="utf-8"))["suffixes"]
print("✅  SVM ready.\n")

_sorted_keys = sorted(norm_map.keys(), key=len, reverse=True)
_protected   = set("".join(norm_map.values()))

def normalize(text):
    text = unicodedata.normalize("NFC", text)
    for key in _sorted_keys:
        rep = norm_map[key]
        if rep == "" and key in _protected: continue
        if key in text: text = text.replace(key, rep)
    return " ".join(text.split())

def tokenize(text):
    tokens = []
    for t in text.split():
        tokens.extend(p for p in re.split(r"[–\-]", t) if p)
    return tokens

def preprocess(raw):
    norm = normalize(raw)
    toks = tokenize(norm)
    toks = [t for t in toks if t not in stopword_set]
    toks = [t[:-len(s)] if t.endswith(s) and len(t)-len(s) >= MIN_ROOT_LEN else t
            for t in toks for s in suffix_list[:1]] if suffix_list else toks
    return norm, toks, " ".join(toks)


# ─────────────────────────────────────────────────────────────
# LOG  (per-FCHV when authenticated, global fallback)
# ─────────────────────────────────────────────────────────────
def log_entry(transcript, answer, mode, prediction, stt_ms, total_ms,
              tts_provider="", protocol_condition=None) -> str:
    session_id = str(uuid.uuid4())[:8]
    entry = {
        "session_id":   session_id,
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "mode":         mode,
        "question":     transcript,
        "answer":       answer,
        "prediction":   prediction,
        "tts_provider": tts_provider,
        "protocol":     protocol_condition or "",
        "stt_ms":       stt_ms,
        "total_ms":     total_ms,
    }
    # Write to per-FCHV log if authenticated, otherwise global log
    fchv_id = session.get("fchv_id")
    if fchv_id:
        log_path = fchv_data_path(fchv_id, "log.json")
    else:
        log_path = LOG_FILE
    try:
        log = json.load(open(log_path, encoding="utf-8"))
    except: log = []
    log.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    return session_id


# ─────────────────────────────────────────────────────────────
# HOSPITAL DATABASE + REFERRAL ENGINE
# ─────────────────────────────────────────────────────────────
import random

HOSPITALS = [
    {"id": "bpkihs", "name": "B.P. Koirala Institute of Health Sciences", "short": "BPKIHS",
     "district": "Sunsari", "zone": "Koshi", "type": "tertiary", "level": 3, "distance_km": 45,
     "specialties": ["obstetrics", "neonatal ICU", "surgery", "cardiology"],
     "beds_total": 500, "emergency": True, "ambulance": True, "contact": "+977-025-525555", "lat": 26.467, "lon": 87.283},
    {"id": "bir", "name": "Bir Hospital", "short": "Bir Hospital",
     "district": "Kathmandu", "zone": "Bagmati", "type": "tertiary", "level": 3, "distance_km": 110,
     "specialties": ["trauma", "surgery", "obstetrics", "neurology"],
     "beds_total": 450, "emergency": True, "ambulance": True, "contact": "+977-01-4221119", "lat": 27.705, "lon": 85.314},
    {"id": "paropakar", "name": "Paropakar Maternity & Women's Hospital", "short": "Paropakar Maternity",
     "district": "Kathmandu", "zone": "Bagmati", "type": "tertiary", "level": 3, "distance_km": 115,
     "specialties": ["obstetrics", "gynecology", "neonatal ICU", "family planning"],
     "beds_total": 300, "emergency": True, "ambulance": True, "contact": "+977-01-4262068", "lat": 27.695, "lon": 85.318},
    {"id": "dhankuta", "name": "Dhankuta District Hospital", "short": "Dhankuta Hospital",
     "district": "Dhankuta", "zone": "Koshi", "type": "district", "level": 2, "distance_km": 18,
     "specialties": ["obstetrics", "surgery", "pediatrics", "emergency"],
     "beds_total": 100, "emergency": True, "ambulance": True, "contact": "+977-026-520088", "lat": 26.978, "lon": 87.339},
    {"id": "hetauda", "name": "Hetauda Hospital", "short": "Hetauda Hospital",
     "district": "Makwanpur", "zone": "Narayani", "type": "district", "level": 2, "distance_km": 28,
     "specialties": ["obstetrics", "surgery", "pediatrics", "emergency"],
     "beds_total": 120, "emergency": True, "ambulance": True, "contact": "+977-057-520077", "lat": 27.428, "lon": 85.031},
    {"id": "pokhara", "name": "Western Regional Hospital Pokhara", "short": "Pokhara Regional",
     "district": "Kaski", "zone": "Gandaki", "type": "tertiary", "level": 3, "distance_km": 62,
     "specialties": ["obstetrics", "neonatal ICU", "surgery", "trauma"],
     "beds_total": 350, "emergency": True, "ambulance": True, "contact": "+977-061-520066", "lat": 28.209, "lon": 83.985},
    {"id": "phc_hile", "name": "Hile Primary Health Centre", "short": "Hile PHC",
     "district": "Dhankuta", "zone": "Koshi", "type": "phc", "level": 1, "distance_km": 6,
     "specialties": ["antenatal care", "basic emergency", "family planning"],
     "beds_total": 15, "emergency": False, "ambulance": False, "contact": "+977-026-540012", "lat": 27.042, "lon": 87.307},
    {"id": "hp_pakhribas", "name": "Pakhribas Health Post", "short": "Pakhribas HP",
     "district": "Dhankuta", "zone": "Koshi", "type": "healthpost", "level": 0, "distance_km": 2,
     "specialties": ["first aid", "antenatal checkup", "referral"],
     "beds_total": 4, "emergency": False, "ambulance": False, "contact": "+977-026-541100", "lat": 27.043, "lon": 87.324},
]

CONDITION_MAP = {
    "रगत":   {"condition": "Possible hemorrhage",        "ne": "रक्तस्राव",       "severity_boost": True},
    "सुन्निएको": {"condition": "Possible preeclampsia",  "ne": "प्रिएक्लाम्पसिया","severity_boost": True},
    "ज्वरो": {"condition": "Fever",                      "ne": "ज्वरो",            "severity_boost": False},
    "बान्ता": {"condition": "Vomiting",                  "ne": "बान्ता",           "severity_boost": False},
    "पेट":   {"condition": "Abdominal pain",              "ne": "पेट दुखाइ",       "severity_boost": False},
    "श्वास": {"condition": "Breathing difficulty",        "ne": "सास फेर्न गाह्रो","severity_boost": True},
    "बच्चा": {"condition": "Child concern",               "ne": "शिशु स्वास्थ्य", "severity_boost": False},
    "प्रसव": {"condition": "Labour",                      "ne": "प्रसव",            "severity_boost": True},
    "गर्भ":  {"condition": "Pregnancy",                   "ne": "गर्भावस्था",      "severity_boost": False},
    "मुर्छा":{"condition": "Loss of consciousness",       "ne": "बेहोस",            "severity_boost": True},
}
ROAD_CONDITIONS = ["Clear", "Muddy after rain", "Under repair — allow extra time", "Good condition"]

def simulate_beds(hospital_id: str) -> dict:
    seed = sum(ord(c) for c in hospital_id)
    r = random.Random(seed + int(time.time() // 3600))
    total = next((h["beds_total"] for h in HOSPITALS if h["id"] == hospital_id), 50)
    occupied = r.randint(int(total * 0.5), int(total * 0.92))
    free = total - occupied
    status = "Full" if free == 0 else ("Limited" if free <= 3 else "Available")
    return {"total": total, "occupied": occupied, "free": free, "status": status}

def estimate_travel(dist: float) -> dict:
    if dist <= 3:    mins, mode = dist * 8, "Walking"
    elif dist <= 15: mins, mode = dist * 4, "Motorbike / jeep"
    elif dist <= 40: mins, mode = dist * 3.2, "Vehicle"
    else:            mins, mode = dist * 2.5, "Vehicle (highway)"
    return {
        "minutes": int(mins),
        "hours_mins": f"{int(mins)//60}h {int(mins)%60}m" if mins >= 60 else f"{int(mins)} min",
        "mode": mode,
    }

def build_referral(prediction: str, transcript: str, answer: str, protocol_result: dict | None = None) -> dict:
    effective_risk = prediction
    if protocol_result and "result" in protocol_result:
        proto_risk = protocol_result["result"].get("risk", "")
        risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        if risk_order.get(proto_risk, 0) > risk_order.get(prediction, 0):
            effective_risk = "high" if proto_risk == "critical" else proto_risk

    txt_lower = (transcript or "") + " " + (answer or "")
    detected = []; boosted = False
    for kw, info in CONDITION_MAP.items():
        if kw in txt_lower:
            detected.append(info)
            if info["severity_boost"]: boosted = True
    if effective_risk == "medium" and boosted: effective_risk = "high"

    level_map = {"high": 2, "medium": 1, "low": 0, "critical": 2}
    min_level = level_map.get(effective_risk, 1)
    candidates = [h for h in HOSPITALS if h["level"] >= min_level and
                  (effective_risk not in ["high", "critical"] or h["emergency"])]
    candidates.sort(key=lambda h: h["distance_km"])
    if not candidates: candidates = sorted(HOSPITALS, key=lambda h: h["distance_km"])

    primary = candidates[0]
    alternate = candidates[1] if len(candidates) > 1 else None
    pb = simulate_beds(primary["id"])
    referred = (alternate if pb["status"] == "Full" and alternate else primary)
    rb = simulate_beds(referred["id"])
    travel = estimate_travel(referred["distance_km"])

    if protocol_result and "result" in protocol_result:
        reason_ne = protocol_result["result"].get("advice", "")[:80]
    elif detected:
        reason_ne = " + ".join(d["ne"] for d in list({d["ne"]: d for d in detected}.values())[:3])
    else:
        reason_ne = {"high": "उच्च जोखिम", "medium": "मध्यम जोखिम", "low": "नियमित अनुगमन"}.get(effective_risk, "")

    urgency_map = {
        "high":    {"label": "IMMEDIATE", "ne": "तुरुन्त — ढिला नगर्नुहोस्!", "color": "high"},
        "critical":{"label": "IMMEDIATE", "ne": "तुरुन्त — ढिला नगर्नुहोस्!", "color": "high"},
        "medium":  {"label": "WITHIN 24H", "ne": "२४ घण्टाभित्र जानुहोस्",    "color": "medium"},
        "low":     {"label": "ROUTINE",    "ne": "सुविधाअनुसार भेट गर्नुहोस्","color": "low"},
    }
    urgency = urgency_map.get(effective_risk, urgency_map["medium"])

    actions = []
    if effective_risk in ["high", "critical"]:
        if referred["ambulance"]: actions.append({"icon": "🚑", "label": "Call Ambulance",    "sub": referred["contact"], "type": "call",   "urgent": True})
        actions.append({"icon": "📞", "label": "Call Hospital",      "sub": referred["contact"], "type": "call",   "urgent": True})
        actions.append({"icon": "👨‍👩‍👧", "label": "Notify Family",    "sub": "Inform immediately","type": "action", "urgent": True})
    elif effective_risk == "medium":
        actions.append({"icon": "📞", "label": "Schedule Appointment","sub": referred["contact"], "type": "call",   "urgent": False})
        actions.append({"icon": "📋", "label": "Monitor Symptoms",    "sub": "Document changes",  "type": "action", "urgent": False})
    else:
        actions.append({"icon": "📋", "label": "Routine Checkup",    "sub": "Next scheduled visit","type": "action","urgent": False})
        actions.append({"icon": "💧", "label": "Self-care",           "sub": "Rest + hydration",   "type": "action","urgent": False})

    return {
        "risk": effective_risk, "urgency": urgency, "reason_ne": reason_ne,
        "hospital": {
            "id": referred["id"], "name": referred["name"], "short": referred["short"],
            "district": referred["district"], "type": referred["type"],
            "distance": referred["distance_km"], "contact": referred["contact"],
            "emergency": referred["emergency"], "ambulance": referred["ambulance"],
            "specialties": referred["specialties"][:4], "beds": rb,
        },
        "travel": travel,
        "road_condition": random.Random(referred["id"]).choice(ROAD_CONDITIONS),
        "actions": actions,
        "redirect_reason": (f"{primary['short']} full → redirecting" if pb["status"] == "Full" and alternate else None),
        "alternate": ({"name": alternate["short"], "distance": alternate["distance_km"], "contact": alternate["contact"]}
                      if alternate and alternate["id"] != referred["id"] else None),
        "protocol_result": protocol_result,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/login")
def login_page():
    if "fchv_id" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/fchv/login", methods=["POST"])
def fchv_login():
    data = request.get_json(silent=True) or {}
    fchv_id  = (data.get("fchv_id") or "").strip().upper()
    password = data.get("password", "")
    if not fchv_id or not password:
        return jsonify({"error": "ID र पासवर्ड आवश्यक छ।"}), 400

    accounts = load_fchv_index()
    account = next((a for a in accounts if a["fchv_id"] == fchv_id), None)
    if not account or account["pwd_hash"] != hash_pwd(password):
        return jsonify({"error": "गलत FCHV ID वा पासवर्ड।"}), 401

    session["fchv_id"]   = account["fchv_id"]
    session["fchv_name"] = account["name"]
    session["district"]  = account.get("district", "")
    session["vdc"]       = account.get("vdc", "")
    return jsonify({"success": True, "fchv_id": account["fchv_id"], "name": account["name"]})


@app.route("/fchv/logout", methods=["POST", "GET"])
def fchv_logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.route("/fchv/register", methods=["GET"])
def fchv_register_page():
    return render_template("fchv_register.html")


@app.route("/fchv/register", methods=["POST"])
def fchv_register():
    data = request.get_json(silent=True) or {}
    name     = (data.get("name") or "").strip()
    phone    = (data.get("phone") or "").strip()
    district = (data.get("district") or "").strip()
    vdc      = (data.get("vdc") or "").strip()
    ward     = (data.get("ward") or "").strip()
    password = data.get("password", "")

    if not name or not phone or not district or not password:
        return jsonify({"error": "अनिवार्य फिल्डहरू भर्नुहोस्।"}), 400
    if len(password) < 8:
        return jsonify({"error": "पासवर्ड कम्तिमा ८ अक्षर हुनुपर्छ।"}), 400

    accounts = load_fchv_index()

    if any(a["phone"] == phone for a in accounts):
        return jsonify({"error": "यो फोन नम्बर पहिले नै दर्ता भइसकेको छ।"}), 409

    next_num = max(
        (int(a["fchv_id"].split("-")[-1]) for a in accounts if a["fchv_id"].startswith("FCHV-")),
        default=0
    ) + 1
    fchv_id = f"FCHV-{str(next_num).zfill(3)}"

    account = {
        "fchv_id": fchv_id, "name": name, "phone": phone,
        "district": district, "vdc": vdc, "ward": ward,
        "pwd_hash": hash_pwd(password),
        "registered_at": datetime.utcnow().isoformat() + "Z",
    }
    accounts.append(account)
    save_fchv_index(accounts)

    for fname in ["patients.json", "sessions.json", "log.json"]:
        fchv_data_path(fchv_id, fname)  # creates file if absent

    return jsonify({"success": True, "fchv_id": fchv_id, "name": name})


@app.route("/fchv/next_id")
def fchv_next_id():
    accounts = load_fchv_index()
    next_num = max(
        (int(a["fchv_id"].split("-")[-1]) for a in accounts if a["fchv_id"].startswith("FCHV-")),
        default=0
    ) + 1
    return jsonify({"next_number": next_num})


@app.route("/fchv/me")
@login_required
def fchv_me():
    return jsonify({
        "fchv_id":  session["fchv_id"],
        "name":     session["fchv_name"],
        "district": session.get("district", ""),
        "vdc":      session.get("vdc", ""),
    })


# ─────────────────────────────────────────────────────────────
# MAIN ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/index1")
def index1():
    return render_template("index1.html")

@app.route("/chat", methods=["POST"])
def chat():
    t0 = time.time()
    online = is_online()
    ai_ready = online and (bool(GEMINI_API_KEY) or bool(GROQ_API_KEY))

    try: history = json.loads(request.form.get("history", "[]"))
    except: history = []

    input_type = request.form.get("input_type", "audio")
    transcript = answer = mode = provider = tts_provider = ""
    prediction = "medium"
    stt_ms = 0
    protocol_result = None

    # ── TEXT INPUT ──────────────────────────────────────────
    if input_type == "text":
        transcript = request.form.get("text_input", "").strip()
        if not transcript:
            return jsonify({"error": "No text provided"}), 400
        _, _, mi = preprocess(transcript)
        prediction = svm_model.predict([mi])[0]
        detected_condition = detect_condition_from_text(transcript)
        if detected_condition:
            protocol_result = {
                "detected": detected_condition,
                "protocol": IMNCI_PROTOCOLS[detected_condition],
                "next_question": get_protocol_next_question(detected_condition, {}),
            }
        if ai_ready:
            answer, provider = get_ai_answer(transcript, prediction, history, protocol_result)
            mode = f"ai_{provider}_text"
        else:
            answer = TRIAGE_RESPONSES.get(prediction, TRIAGE_RESPONSES["medium"])
            mode = "svm_offline"; provider = "offline"

    # ── AUDIO INPUT ─────────────────────────────────────────
    else:
        audio_file = request.files.get("audio")
        if not audio_file:
            return jsonify({"error": "No audio received"}), 400
        audio_bytes = audio_file.read()
        filename = audio_file.filename or "recording.webm"
        fmt = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
        if len(audio_bytes) < 500:
            return jsonify({"error": "Too short — speak for at least 2 seconds"}), 400

        groq_stt_ok = online and bool(GROQ_API_KEY)
        if groq_stt_ok:
            try:
                wav_bytes, wav_fmt = convert_to_wav(audio_bytes, fmt)
                transcript = groq_stt(wav_bytes, wav_fmt)
                stt_ms = int((time.time() - t0) * 1000)
            except Exception as e:
                print(f"⚠️  Groq STT failed: {e}, falling back to local Whisper")
                transcript = whisper_transcribe(audio_bytes, fmt)
                stt_ms = int((time.time() - t0) * 1000)
        else:
            transcript = whisper_transcribe(audio_bytes, fmt)
            stt_ms = int((time.time() - t0) * 1000)

        if not transcript:
            return jsonify({"error": "No speech detected."}), 422

        _, _, mi = preprocess(transcript)
        prediction = svm_model.predict([mi])[0]
        detected_condition = detect_condition_from_text(transcript)
        if detected_condition:
            protocol_result = {
                "detected": detected_condition,
                "protocol": IMNCI_PROTOCOLS[detected_condition],
                "next_question": get_protocol_next_question(detected_condition, {}),
            }

        if ai_ready:
            answer, provider = get_ai_answer(transcript, prediction, history, protocol_result)
            mode = f"ai_{provider}"
        else:
            answer = TRIAGE_RESPONSES.get(prediction, TRIAGE_RESPONSES["medium"])
            mode = "svm_offline"; provider = "offline"

    if not answer:
        return jsonify({"error": "Could not generate response"}), 500

    total_ms = int((time.time() - t0) * 1000)

    # ── TTS ─────────────────────────────────────────────────
    voice_url = None; tts_provider = "none"
    mp3, tts_provider = text_to_speech(answer, online=online)
    if mp3:
        tts_path = os.path.join(VOICES_DIR, f"dyn_{str(uuid.uuid4())[:8]}.mp3")
        with open(tts_path, "wb") as f: f.write(mp3)
        voice_url = f"/static/voices/{os.path.basename(tts_path)}"
    elif prediction in VOICES_AVAILABLE:
        voice_url = f"/static/voices/{prediction}.mp3"; tts_provider = "prerendered"

    proto_condition = protocol_result.get("detected") if protocol_result else None
    session_id = log_entry(transcript, answer, mode, prediction, stt_ms, total_ms, tts_provider, proto_condition)
    referral = build_referral(prediction, transcript, answer, protocol_result)

    return jsonify({
        "transcript": transcript, "answer": answer, "mode": mode,
        "provider": provider, "prediction": prediction, "session_id": session_id,
        "stt_ms": stt_ms, "total_ms": total_ms, "voice_url": voice_url,
        "tts_provider": tts_provider, "referral": referral,
        "protocol_result": protocol_result, "input_type": input_type,
    })


@app.route("/protocol", methods=["POST"])
def protocol_endpoint():
    data = request.get_json(silent=True) or {}
    condition = data.get("condition")
    answers = data.get("answers", {})
    if not condition or condition not in IMNCI_PROTOCOLS:
        return jsonify({"error": "Unknown condition", "available": list(IMNCI_PROTOCOLS.keys())}), 400
    result = run_protocol_from_answers(condition, answers)
    next_q = get_protocol_next_question(condition, answers)
    return jsonify({
        "result": result, "next_question": next_q,
        "complete": next_q is None,
        "all_protocols": {k: {"label": v["label"], "label_en": v["label_en"], "age_group": v["age_group"]}
                          for k, v in IMNCI_PROTOCOLS.items()},
    })


@app.route("/protocols")
def protocols_list():
    return jsonify({k: {"label": v["label"], "label_en": v["label_en"],
                        "age_group": v["age_group"], "step_count": len(v["steps"])}
                    for k, v in IMNCI_PROTOCOLS.items()})


@app.route("/hospitals")
def hospitals():
    return jsonify([{**h, "beds": simulate_beds(h["id"]), "travel": estimate_travel(h["distance_km"])}
                    for h in HOSPITALS])


@app.route("/status")
def status():
    online = is_online()
    gem_ok = bool(GEMINI_API_KEY)
    grq_ok = bool(GROQ_API_KEY)
    el_ok  = bool(ELEVENLABS_API_KEY) and not ELEVENLABS_API_KEY.startswith("YOUR_")
    ggl_ok = bool(GOOGLE_TTS_API_KEY)
    return jsonify({
        "online": online,
        "ai_available": online and (gem_ok or grq_ok),
        "primary_llm": f"Gemini {GEMINI_MODEL}" if gem_ok else "Groq (fallback)",
        "gemini_configured": gem_ok,
        "gemini_model": GEMINI_MODEL,
        "groq_configured": grq_ok,
        "groq_model": GROQ_MODEL,
        "groq_stt_model": GROQ_STT_MODEL,
        "elevenlabs_configured": el_ok,
        "elevenlabs_voice": ELEVENLABS_VOICE_ID,
        "google_tts_configured": ggl_ok,
        "piper_available": _check_piper(),
        "whisper_model": WHISPER_MODEL,
        "imnci_protocols": len(IMNCI_PROTOCOLS),
        "tts_priority_nepali": (
            "Google ne-NP → ElevenLabs → gTTS" if ggl_ok
            else "ElevenLabs → gTTS" if el_ok
            else "gTTS"
        ),
        "tts_current_nepali": (
            "google_ne_NP" if ggl_ok
            else "elevenlabs" if el_ok
            else "gtts"
        ),
    })


@app.route("/data")
@login_required
def data_stats():
    fchv_id = session["fchv_id"]
    log_path = fchv_data_path(fchv_id, "log.json")
    try: log = json.load(open(log_path, encoding="utf-8"))
    except: log = []
    counts = {"low": 0, "medium": 0, "high": 0}
    by_mode = defaultdict(int); by_tts = defaultdict(int)
    ms_list = []; daily = defaultdict(int); by_protocol = defaultdict(int)
    for e in log:
        p = e.get("prediction", "")
        if p in counts: counts[p] += 1
        by_mode[e.get("mode", "unknown")] += 1
        by_tts[e.get("tts_provider", "unknown")] += 1
        if e.get("total_ms"): ms_list.append(e["total_ms"])
        day = (e.get("timestamp") or "")[:10]
        if day: daily[day] += 1
        if e.get("protocol"): by_protocol[e["protocol"]] += 1
    return jsonify({
        "total": len(log), "counts": counts, "by_mode": dict(by_mode),
        "by_tts": dict(by_tts),
        "ai_online": sum(v for k, v in by_mode.items() if k.startswith("ai_")),
        "svm_offline": by_mode.get("svm_offline", 0),
        "avg_ms": int(sum(ms_list) / len(ms_list)) if ms_list else 0,
        "daily": dict(sorted(daily.items())[-7:]),
        "recent": log[-20:][::-1],
        "by_protocol": dict(by_protocol),
    })


def _cleanup():
    now = time.time()
    for f in Path(VOICES_DIR).glob("dyn_*.mp3"):
        if now - f.stat().st_mtime > 3600: f.unlink(missing_ok=True)

_cleanup()


@app.route("/tts", methods=["POST"])
def tts_endpoint():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400
    online = is_online()
    mp3, _ = text_to_speech(text, online=online)
    if mp3:
        return send_file(
            io.BytesIO(mp3),
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="tts.mp3",
        )
    return jsonify({"error": "TTS unavailable"}), 503


# ─────────────────────────────────────────────────────────────
# PATIENT REGISTRATION & SESSION MANAGEMENT  (per-FCHV scoped)
# ─────────────────────────────────────────────────────────────

@app.route("/register")
@login_required
def register_page():
    return render_template("register_patient.html")


@app.route("/register_patient", methods=["POST"])
@login_required
def register_patient():
    data = request.get_json()
    fchv_id = session["fchv_id"]
    for field in ["id", "name", "phone"]:
        if not data.get(field):
            return jsonify({"error": f"Missing: {field}"}), 400

    path = fchv_data_path(fchv_id, "patients.json")
    try:    patients = json.load(open(path, encoding="utf-8"))
    except: patients = []

    if any(p["id"] == data["id"] for p in patients):
        return jsonify({"error": "Patient ID already exists"}), 409
    if any(p["phone"] == data["phone"] for p in patients):
        return jsonify({"error": "Phone already registered"}), 409

    patient = {
        "id":           data["id"],
        "name":         data["name"],
        "phone":        data["phone"],
        "sex":          data.get("sex", ""),
        "location":     data.get("location", ""),
        "age":          data.get("age"),
        "ageGroup":     data.get("ageGroup"),
        "registeredBy": fchv_id,
        "registeredAt": data.get("registeredAt", datetime.utcnow().isoformat() + "Z"),
    }
    patients.append(patient)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=2)
    return jsonify({"success": True, "patient": patient})


@app.route("/search_patients")
@login_required
def search_patients():
    fchv_id = session["fchv_id"]
    query   = request.args.get("q", "").lower().strip()
    path    = fchv_data_path(fchv_id, "patients.json")
    try:    patients = json.load(open(path, encoding="utf-8"))
    except: patients = []
    if not query:
        return jsonify({"patients": patients[::-1][:50]})
    results = [p for p in patients if (
        query in p.get("name", "").lower() or
        query in p.get("phone", "") or
        query in p.get("id", "").lower()
    )]
    return jsonify({"patients": results[:50]})


@app.route("/get_patient/<patient_id>")
@login_required
def get_patient(patient_id):
    fchv_id = session["fchv_id"]
    path    = fchv_data_path(fchv_id, "patients.json")
    try:    patients = json.load(open(path, encoding="utf-8"))
    except: return jsonify({"error": "No patients found"}), 404
    for p in patients:
        if p["id"] == patient_id:
            return jsonify({"patient": p})
    return jsonify({"error": "Patient not found"}), 404


@app.route("/save_session", methods=["POST"])
@login_required
def save_session_route():
    fchv_id = session["fchv_id"]
    data    = request.get_json()
    for field in ["patientId", "transcript", "answer", "prediction"]:
        if field not in data:
            return jsonify({"error": f"Missing: {field}"}), 400

    path = fchv_data_path(fchv_id, "sessions.json")
    try:    sessions_data = json.load(open(path, encoding="utf-8"))
    except: sessions_data = []

    entry = {
        "sessionId":       str(uuid.uuid4())[:8],
        "fchvId":          fchv_id,
        "patientId":       data["patientId"],
        "patientName":     data.get("patientName", ""),
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "transcript":      data["transcript"],
        "answer":          data["answer"],
        "prediction":      data["prediction"],
        "protocol":        data.get("protocol", ""),
        "protocolAnswers": data.get("protocolAnswers", {}),
        "mode":            data.get("mode", ""),
        "provider":        data.get("provider", ""),
    }
    sessions_data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sessions_data, f, ensure_ascii=False, indent=2)
    return jsonify({"success": True, "sessionId": entry["sessionId"]})


@app.route("/get_patient_history/<patient_id>")
@login_required
def get_patient_history(patient_id):
    fchv_id = session["fchv_id"]
    path    = fchv_data_path(fchv_id, "sessions.json")
    try:    sessions_data = json.load(open(path, encoding="utf-8"))
    except: return jsonify({"sessions": []})
    hist = [s for s in sessions_data if s.get("patientId") == patient_id]
    hist.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify({"sessions": hist})


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀  FCHV_assist IMNCI → http://localhost:5000")
    print(f"    Primary LLM  : Gemini {GEMINI_MODEL}")
    print(f"    Fallback LLM : Groq {GROQ_MODEL}")
    print(f"    TTS Nepali   : Google ne-NP → ElevenLabs Charlotte → gTTS")
    print(f"    IMNCI Protocols: {len(IMNCI_PROTOCOLS)} conditions loaded")
    app.run(debug=False, host="0.0.0.0", port=5000)
