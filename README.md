# Gradient Descent — NCCN-Grounded Oncology AI for India

> **AWS AI for Bharat Hackathon 2025**

An AI platform that gives Indian oncologists structured, guideline-grounded clinical decisions in seconds — and gives their patients a treatment plan they can actually understand, in English or Hindi.

---

## 🔗 Links

| | |
|---|---|
| 🌐 **Live Demo** | https://prototypegradientdescent.netlify.app/ |
| 🎬 **Demo Video** | https://youtu.be/6oLmamkM4Gg |

---

## The Problem

India records over 14 lakh new cancer cases every year. Doctors spend significant time manually cross-referencing complex NCCN guidelines for each patient — while patients receive treatment plans written in technical English they cannot understand, with no Hindi support and no way to ask follow-up questions.

---

## The Solution

Gradient Descent is a two-sided AI platform:

**For Doctors** — structured NCCN-grounded clinical decisions personalised to each patient's biomarkers, stage, radiation history, and prior treatment lines. Drug names, doses (mg/m²), schedules, NCCN evidence categories, and India-specific costs — all in one response.

**For Patients** — the approved plan rewritten in warm, simple language in English or Hindi, with every medicine explained, realistic India costs (govt / Jan Aushadhi / private / PMJAY), self-care guidance, and a 24/7 chatbot grounded in NCCN patient-care guidelines.

---

## Architecture

```
[Netlify Frontend]
        │
        │ HTTPS POST
        ▼
[Amazon API Gateway]
        │
        ├── /clinical/query
        ├── /patient/simplify
        ├── /patient/query
        └── /report/summarize
        │
        ▼
[AWS Lambda — Python 3.12]
        │
        ├── Searches NCCN chunks (custom soft-score keyword search)
        ├── Builds patient-contextualised prompt
        │
        ├──────────────────────────┐
        ▼                          ▼
[Amazon S3]              [Amazon Bedrock]
nccn_chunks.json         Nova Pro — APAC
4,163 clinical chunks    inference profile
+ patient-care chunks    (ap-south-1)
```

---

## AWS Services Used

| Service | Usage |
|---|---|
| **Amazon Bedrock (Nova Pro)** | All AI inference — clinical plans, patient simplification, Hindi translation, Q&A |
| **AWS Lambda** | Serverless Python backend — 4 endpoints, zero idle cost |
| **Amazon API Gateway** | HTTP API routing with CORS |
| **Amazon S3** | Stores 4,163 NCCN clinical chunks + patient-care chunks as JSON |

---

## Features

- **NCCN Guideline RAG** — 4,163 chunks from Breast (v1.2026), Cervical (v2.2026), and Oral Cancer (v1.2026) guidelines
- **Structured Clinical Plans** — drug names, doses, NCCN evidence categories, India-specific costs per response
- **Doctor Approval Gate** — AI cannot deliver anything to patients without doctor review and approval
- **Patient Plan Simplifier** — separate AI endpoint rewrites clinical plans in plain language
- **Hindi / English Toggle** — full AI re-translation by Nova Pro, not a rule-based swap
- **India Cost Context** — government, Jan Aushadhi generic, private branded rates + PMJAY eligibility
- **Patient Q&A Chatbot** — grounded in NCCN patient-care guidelines, locked until doctor releases plan
- **Structured Patient Registry** — biomarkers, radiation (dose, fractions, intent), treatment lines, ECOG status

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Single-file HTML / CSS / JS — no framework |
| Hosting | Netlify (free tier) |
| Backend | AWS Lambda — Python 3.12 |
| API | Amazon API Gateway — HTTP API |
| AI Model | Amazon Bedrock — Nova Pro (APAC, ap-south-1) |
| Data Store | Amazon S3 |
| RAG Pipeline | PyMuPDF (PDF chunking) + custom soft-score keyword search |
| Guidelines | NCCN 2026 — Breast, Cervical, Oral Cancer |
| Languages | English + Hindi |

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /clinical/query` | Doctor clinical decision support — NCCN-grounded treatment plan |
| `POST /patient/simplify` | Rewrites approved doctor plan in patient-friendly language |
| `POST /patient/query` | Patient chatbot — answers questions from NCCN patient-care guidelines |
| `POST /report/summarize` | Summarises uploaded biopsy/imaging reports against NCCN guidelines |

---

## Performance

| Metric | Result |
|---|---|
| Clinical query response | 8–15 seconds |
| Patient plan simplification | 10–18 seconds |
| Chunk retrieval (4,163 chunks) | < 100ms (Lambda in-memory cache) |
| Lambda warm start | ~80ms |
| Cost per clinical query | ~₹1.50–₹2.50 |
| Monthly cost (500 patients/day) | ~₹2,000–₹3,500 |

---

## Repository Structure

```
gradient-descent/
├── index.html           # Complete frontend — single file, no build step
├── lambda_function.py   # AWS Lambda backend — all 4 endpoints
└── README.md
```

> **Note:** NCCN chunk JSON files are not included in this repository as they contain proprietary NCCN guideline content. The S3 bucket is configured via Lambda environment variables.

---

## Environment Variables (Lambda)

| Variable | Description |
|---|---|
| `NCCN_BUCKET` | S3 bucket name containing guideline chunks |
| `NCCN_KEY` | S3 key for clinical chunks JSON |
| `NCCN_PATIENT_KEY` | S3 key for patient-care chunks JSON |

---

## Future Development

- Report upload — biopsy/imaging PDF summarisation (endpoint already built)
- Patient login and cross-session plan persistence
- WhatsApp plan delivery via Twilio
- Tamil, Telugu, Bengali language support
- Lung, Colorectal, Head & Neck NCCN guidelines
- Vector search via Amazon OpenSearch
- ABDM (Ayushman Bharat Digital Mission) integration

---

## Disclaimer

This platform is built for hackathon demonstration purposes. All clinical decisions remain with the treating physician. NCCN guideline references are used for educational and research purposes only.

---

*Built with ❤️ by Team Gradient Descent — AWS AI for Bharat Hackathon 2025*
