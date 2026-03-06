import json
import boto3
import os
import re

# ── CONFIG ─────────────────────────────────────────────────────────────────────
S3_BUCKET         = os.environ.get("NCCN_BUCKET", "gradient-descent-guidelines")
S3_KEY_CLINICAL   = os.environ.get("NCCN_KEY",    "nccn_chunks.json")
S3_KEY_PATIENT    = os.environ.get("NCCN_PATIENT_KEY", "nccn_patient_care_chunks.json")
MODEL_ID          = "arn:aws:bedrock:ap-south-1:788674045219:inference-profile/apac.amazon.nova-pro-v1:0"

bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
s3      = boto3.client("s3")

_clinical_cache     = None
_patient_care_cache = None

def load_clinical_chunks():
    global _clinical_cache
    if _clinical_cache is not None:
        return _clinical_cache
    try:
        print("Loading clinical NCCN chunks...")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY_CLINICAL)
        _clinical_cache = json.loads(obj["Body"].read().decode("utf-8"))
        print(f"Loaded {len(_clinical_cache)} clinical chunks")
    except Exception as e:
        print(f"Clinical chunk load error: {e}")
        _clinical_cache = []
    return _clinical_cache

def load_patient_care_chunks():
    global _patient_care_cache
    if _patient_care_cache is not None:
        return _patient_care_cache
    try:
        print("Loading patient-care NCCN chunks...")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY_PATIENT)
        _patient_care_cache = json.loads(obj["Body"].read().decode("utf-8"))
        print(f"Loaded {len(_patient_care_cache)} patient-care chunks")
    except Exception as e:
        print(f"Patient care chunks not found (optional): {e}")
        _patient_care_cache = []
    return _patient_care_cache

CLINICAL_BONUS_TERMS = {
    "her2","er","pr","brca","ki67","pdl1","stage","neoadjuvant","adjuvant",
    "metastatic","surgery","radiation","chemotherapy","immunotherapy","hormone",
    "trastuzumab","pertuzumab","docetaxel","paclitaxel","carboplatin","capecitabine",
    "letrozole","anastrozole","tamoxifen","pembrolizumab","olaparib","eribulin",
    "nccn","category","preferred","regimen","monitoring","dose","cycle",
    "response","progression","recurrence","biomarker"
}

def search_chunks(chunks, query, cancer_type="", top_k=8):
    if not chunks:
        return []
    query_lower  = query.lower()
    query_terms  = set(re.findall(r'\b\w{3,}\b', query_lower))
    stop_words   = {"the","a","an","is","are","was","for","of","in","to","and",
                    "or","with","what","does","say","about","patient","please",
                    "tell","me","give","how","when","this","that","they","their"}
    query_terms -= stop_words
    cancer_lower = cancer_type.lower().split()[0] if cancer_type else ""
    scored = []
    for chunk in chunks:
        text_lower     = chunk.get("text","").lower()
        chunk_keywords = set(chunk.get("keywords", []))
        score          = 0
        for term in query_terms:
            if term in text_lower:
                score += 2
            if any(term in kw for kw in chunk_keywords):
                score += 1
        for kw in CLINICAL_BONUS_TERMS:
            if kw in text_lower:
                score += 1
        if cancer_lower and cancer_lower in text_lower:
            score += 4
        if any(w in text_lower for w in ["category 1","category 2a","preferred","recommended regimen"]):
            score += 3
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

def search_combined(query, cancer_type="", clinical_k=6, patient_k=4):
    clinical_results     = search_chunks(load_clinical_chunks(),    query, cancer_type, clinical_k)
    patient_care_results = search_chunks(load_patient_care_chunks(), query, cancer_type, patient_k)
    return patient_care_results + clinical_results

def format_context(chunks):
    if not chunks:
        return "No specific guideline sections retrieved."
    lines = []
    for i, c in enumerate(chunks):
        ctype = c.get("chunk_type", "clinical")
        label = "Patient Care Guidelines" if ctype == "patient_care" else "Clinical Guidelines"
        lines.append(
            f"[{label} | {c.get('cancer_type','?')} | {c.get('source','?')} | Page {c.get('page','?')}]\n{c.get('text','')}"
        )
    return "\n\n---\n\n".join(lines)

def invoke_model(system_text, messages, max_tokens=2000):
    try:
        body = {
            "system": [{"text": system_text}],
            "messages": messages,
            "inferenceConfig": {
                "max_new_tokens": max_tokens,
                "temperature": 0.2,
                "topP": 0.9
            }
        }
        response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
        result = json.loads(response["body"].read())
        return result["output"]["message"]["content"][0]["text"]
    except Exception as e:
        print(f"Model error: {e}")
        return "AI temporarily unavailable. Please retry."

def build_patient_summary(patient):
    if not patient:
        return "No patient data."
    bm  = patient.get("biomarkers", {})
    tx  = patient.get("treatment_history", [])
    rad = patient.get("radiation", {})
    tx_str = "; ".join([
        f"Line {t.get('line','?')}: {t.get('regimen','?')} → {t.get('response','?')}"
        for t in tx
    ]) if tx else "None"
    if rad.get("given") == "Yes":
        rad_str = (f"Radiation: Yes | Site: {rad.get('site','?')} | "
                   f"Dose: {rad.get('dose','?')} Gy | "
                   f"Fractions: {rad.get('fractions','?')} | "
                   f"Intent: {rad.get('intent','?')}")
    else:
        rad_str = "Radiation: None"
    return f"""Patient: {patient.get('name','Unknown')}, {patient.get('age','?')}y, {patient.get('gender','?')}
Cancer: {patient.get('cancer_type','?')} | Stage: {patient.get('stage','?')} | ECOG: {patient.get('ecog','?')}
Conditions: {', '.join(patient.get('conditions', []))}
Biomarkers: ER={bm.get('ER','?')}, PR={bm.get('PR','?')}, HER2={bm.get('HER2','?')}, Ki-67={bm.get('Ki67','?')}, BRCA={bm.get('BRCA','?')}, PD-L1={bm.get('PD_L1','?')}
{rad_str}
Prior Treatment: {tx_str}""".strip()

# ── HANDLER 1 — /clinical/query ───────────────────────────────────────────────
def handle_clinical_query(body):
    question    = body.get("question", "")
    patient     = body.get("patient", {})
    cancer_type = patient.get("cancer_type", "")
    history     = body.get("chat_history", [])
    chunks  = search_chunks(load_clinical_chunks(), question, cancer_type, top_k=10)
    context = format_context(chunks)
    sources = [{"source": c.get("source",""), "page": c.get("page",0),
                "cancer_type": c.get("cancer_type","")} for c in chunks]
    patient_summary = build_patient_summary(patient)
    system_text = f"""You are a senior oncology clinical decision support assistant for Gradient Descent, a clinical AI platform for Indian oncologists.

NCCN GUIDELINE SECTIONS RETRIEVED:
{context}

PATIENT PROFILE:
{patient_summary}

CRITICAL INSTRUCTIONS:
- Be specific: exact drug names, doses (mg/m² or fixed), schedules, cycle counts
- List ALL applicable treatment options for this subtype — do not limit to one
- Always cite NCCN evidence category
- Include India-specific cost context for every option
- Account for radiation history if present in patient profile

REQUIRED FORMAT — output exactly this structure:

### CLINICAL SUMMARY
• [Subtype classification with biomarker rationale]
• [Stage and clinical meaning]
• [Performance status implication]
• [Key prognostic factors]
• [Treatment intent: curative vs palliative]

### NCCN RECOMMENDED TREATMENT OPTIONS

OPTION: <Full Regimen Name>
- Indication: <specific patient subset>
- NCCN Category: <Category 1 / 2A / 2B>
- Drugs & Doses: <drug name dose/m² schedule — be specific>
- Cycles: <number and context e.g. 6 cycles neoadjuvant>
- Pros: <evidence strength, response rates>
- Cons: <key toxicities, contraindications>
- India Cost (per cycle): <Govt ₹X | Jan Aushadhi ₹X | Private ₹X>

[List ALL relevant options — typically 3-6]

### NEXT-LINE / PROGRESSION OPTIONS
• [Specific regimens if current treatment fails]

### MONITORING PLAN
• [Specific labs with frequency]
• [Imaging schedule]
• [Cardiac or other organ monitoring if applicable]

### INDIA-SPECIFIC NOTES
• [Generic availability, PMJAY eligibility, practical challenges]

[NCCN guideline reference only — all clinical decisions rest with the treating physician]"""
    messages = []
    for h in history[-6:]:
        if h.get("role") in ("user","assistant"):
            messages.append({"role": h["role"], "content": [{"text": h.get("content","")}]})
    messages.append({"role": "user", "content": [{"text": question}]})
    answer = invoke_model(system_text, messages, max_tokens=3000)
    return {"answer": answer, "sources": sources, "chunks_used": len(chunks), "approval_required": True}

# ── HANDLER 2 — /patient/simplify ────────────────────────────────────────────
def handle_patient_simplify(body):
    doctor_plan  = body.get("doctor_plan", "")
    patient      = body.get("patient", {})
    language     = body.get("language", "en")
    cancer_type  = patient.get("cancer_type", "Cancer")
    patient_name = patient.get("name", "")
    doctor_name  = body.get("doctor_name", "Your Doctor")
    chunks   = search_combined("patient care diet exercise side effects support recovery",
                                cancer_type, clinical_k=3, patient_k=5)
    care_ctx = format_context(chunks)
    lang_note = (
        "Respond entirely in simple Hindi. For medical terms, write them in English with a Hindi explanation in brackets."
        if language == "hi"
        else "Respond in simple, warm English. Explain every medical term immediately in plain words."
    )
    system_text = f"""You are a compassionate patient educator for Gradient Descent.
Rewrite the doctor's clinical plan so a patient with no medical background can fully understand it and feel reassured.

{lang_note}

NCCN PATIENT CARE & CLINICAL GUIDELINES CONTEXT:
{care_ctx}

ABSOLUTE RULES:
1. Explain every medical term in simple words immediately after using it
2. Be warm, personal, and reassuring throughout — this person is scared
3. Never contradict the doctor's plan — only simplify the language
4. Keep all medical facts 100% accurate
5. Use realistic India-specific costs

OUTPUT EXACTLY these sections with these exact headers — no other format:

SECTION_HISTORY:
[3-4 warm sentences. Explain {patient_name}'s diagnosis in simple words — what was found, what the stage means in everyday language, how the body is responding.]

SECTION_TREATMENT:
[For EACH medicine/treatment in the doctor's plan, one block per medicine:
💊 [Medicine Name]
→ What it does: [simple analogy — e.g. "acts like a guided missile that finds HER2 cancer cells"]
→ How you'll receive it: [tablet at home / drip at hospital / injection]
→ How often: [plain English — e.g. "once every 3 weeks"]
→ Common side effects: [simple words, reassuring tone]
→ Important tip: [one practical instruction for this medicine]

Include surgery or radiation if the doctor mentioned it.]

SECTION_COSTS:
[Practical India costs — be specific:
🏥 Government hospital / AIIMS / Tata Memorial: ₹X–₹Y per cycle
💊 Jan Aushadhi generic medicines: ₹X–₹Y (same medicine, much lower cost)
🏬 Private hospital branded: ₹X–₹Y per cycle
📋 Ayushman Bharat PMJAY: [specific note — does this cancer type qualify?]
💡 Money-saving tip: [one specific practical suggestion]
⚠️ All costs are approximate. They vary by hospital, city, and insurance. Always ask the social worker at your hospital.]

SECTION_SELFCARE:
🥗 What to eat: [3-4 specific foods good for this cancer + 2 specific things to avoid during treatment]
🚶 Gentle movement: [safe, specific exercise guidance suitable during this treatment]
🧼 Taking care of yourself: [hygiene and practical body care during chemo/radiation]
💙 For your mind: [one warm, practical mental wellness suggestion — mention family support]
🚨 Call your doctor immediately if you notice: [3 specific red-flag symptoms for this exact treatment]

WARMTH:
[Two personal warm sentences addressed to {patient_name} by name. Express genuine hope, remind them they are not alone, mention their strength. Sign as: "With care and hope, Dr. {doctor_name} & Team Gradient Descent 💙"]"""
    messages = [{"role": "user", "content": [{"text": f"Please simplify this for {patient_name}:\n\nDOCTOR'S PLAN:\n{doctor_plan}"}]}]
    answer = invoke_model(system_text, messages, max_tokens=3000)
    return {"simplified_plan": answer}

# ── HANDLER 3 — /report/summarize ────────────────────────────────────────────
def handle_report_summary(body):
    report_text = body.get("report_text", "")
    patient     = body.get("patient", {})
    cancer_type = patient.get("cancer_type", "")
    chunks  = search_chunks(load_clinical_chunks(),
                            f"staging treatment surgery response {' '.join(patient.get('conditions', []))}",
                            cancer_type, top_k=8)
    context = format_context(chunks)
    sources = [{"source": c.get("source",""), "page": c.get("page",0),
                "cancer_type": c.get("cancer_type","")} for c in chunks]
    system_text = f"""You are a clinical AI assistant for Gradient Descent.
Summarise the patient report against NCCN guidelines.

NCCN CONTEXT:
{context}

Patient: {patient.get('name')}, {patient.get('age')}y {patient.get('gender')}

Format exactly as:

### REPORT FINDINGS
• [Key findings in clinical language]

### NCCN INTERPRETATION
• [What findings mean per guidelines]

### CLINICAL FLAGS
• 🔴 HIGH: [Critical findings needing immediate action]
• 🟡 MEDIUM: [Important non-urgent findings]
• 🟢 NOTE: [Worth tracking]

### SUGGESTED NEXT STEPS
• [Specific NCCN-recommended actions]

[NCCN reference only — clinical decisions rest with the treating physician]"""
    messages = [{"role": "user", "content": [{"text": f"Report:\n\n{report_text}"}]}]
    summary = invoke_model(system_text, messages, max_tokens=1500)
    return {"summary": summary, "sources": sources}

# ── HANDLER 4 — /patient/query ───────────────────────────────────────────────
def handle_patient_query(body):
    question    = body.get("question", "")
    language    = body.get("language", "en")
    cancer_type = body.get("cancer_type", "")
    history     = body.get("chat_history", [])
    chunks  = search_combined(question, cancer_type, clinical_k=4, patient_k=5)
    context = format_context(chunks)
    if language == "hi":
        system_text = f"""आप Gradient Descent के मरीज सहायक हैं।

NCCN दिशानिर्देश (मरीज देखभाल और उपचार दोनों से):
{context}

नियम:
- हमेशा सरल, गर्मजोशी भरी हिंदी में जवाब दें
- मेडिकल शब्दों को तुरंत सरल हिंदी में समझाएं
- अधिकतम 4-5 वाक्य, जब तक विस्तार जरूरी न हो
- कोई दवाई न बताएं — हमेशा कहें "अपने डॉक्टर से पूछें"
- भावनात्मक सवालों पर पहले सहानुभूति दिखाएं, फिर व्यावहारिक जवाब दें
- खान-पान, व्यायाम, देखभाल के सवालों पर NCCN patient care guidelines का उपयोग करें"""
    else:
        system_text = f"""You are the patient health assistant for Gradient Descent.

NCCN Guidelines context (patient care + clinical — for accuracy, translate to simple language):
{context}

Rules:
- Always respond in simple, warm, non-medical language
- Explain any medical term you must use immediately after in plain words
- Keep to 4-5 sentences unless the question genuinely needs more
- Never prescribe or recommend specific medicines — always say "ask your doctor"
- For diet, exercise, lifestyle questions: draw from patient care guidelines context above
- For emotional or fear-based questions: empathy first, practical second
- For treatment questions: give simple accurate information then encourage doctor conversation"""
    messages = []
    for h in history[-4:]:
        if h.get("role") in ("user","assistant"):
            messages.append({"role": h["role"], "content": [{"text": h.get("content","")}]})
    messages.append({"role": "user", "content": [{"text": question}]})
    answer = invoke_model(system_text, messages, max_tokens=700)
    return {"answer": answer}

# ── MAIN ROUTER ───────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    # CORS preflight
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin":  "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS"
            },
            "body": ""
        }
    headers = {
        "Content-Type":                 "application/json",
        "Access-Control-Allow-Origin":  "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS"
    }
    try:
        body = json.loads(event.get("body", "{}"))
        path = event.get("rawPath", event.get("path", "/clinical/query"))

        # ✅ FIXED: all 4 routes correctly mapped — no duplicates
        if   "/clinical/query"   in path: result = handle_clinical_query(body)
        elif "/patient/simplify" in path: result = handle_patient_simplify(body)
        elif "/report/summarize" in path: result = handle_report_summary(body)
        elif "/patient/query"    in path: result = handle_patient_query(body)
        else: result = {"error": f"Unknown path: {path}"}

        return {"statusCode": 200, "headers": headers, "body": json.dumps(result)}
    except Exception as e:
        print(f"Lambda error: {str(e)}")
        return {"statusCode": 500, "headers": headers,
                "body": json.dumps({"error": str(e)})}