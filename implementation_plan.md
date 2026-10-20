# MEDIQ AI — Intelligent OPD Triage and Queue Management System
## Architecture & Engineering Implementation Plan

### 1. Executive Summary & Clinical Governance
**MEDIQ AI** is an intelligent, safety-first Outpatient Department (OPD) Triage and Queue Management System designed to eliminate chaotic patient crowding, streamline clinical workflows, and catch life-threatening emergencies immediately upon arrival at high-volume healthcare facilities.

> [!IMPORTANT]
> **Clinical Safety Notice**: MEDIQ AI is a **clinical decision-support prototype**, not an autonomous diagnostic system. All AI recommendations are advisory and must be reviewed by qualified clinicians. Deterministic safety rules always override AI predictions.

---

### 2. High-Level System Architecture

```mermaid
graph TD
    subgraph Clients["Frontend Clients (React + Vite + Tailwind)"]
        PatientUI["Patient Portal<br/>- OPD Intake & Vitals<br/>- Live Queue Ticket<br/>- Escalation Banners"]
        ClinicalUI["Doctor / Nurse Dashboard<br/>- Live Dept Queue<br/>- AI Triage Card & Overrides<br/>- Call / Consult / Refer"]
        AdminUI["Admin & Analytics Console<br/>- Real-time KPIs & Recharts<br/>- Staff/Dept Ops<br/>- Audit Trail & Seeder"]
    end

    subgraph Backend["FastAPI Application Services"]
        AuthMiddleware["JWT RBAC Middleware<br/>(Patient, Doctor, Nurse, Admin)"]
        WSManager["WebSocket / SSE Manager<br/>Channel-based broadcasting"]
        
        subgraph TriageEngine["Modular Triage Service"]
            DeterministicRules["Deterministic Safety Rules<br/>(Critical Vitals & Red Flag Keywords)"]
            ContradictionChecker["Data Quality & Contradiction Checker<br/>(Systolic < Diastolic, Out-of-bounds, Missing)"]
            MLModel["Scikit-Learn Classifier<br/>(TF-IDF + Scaled Vitals + Calibrated Probabilities)"]
        end

        QueueService["Queue & Consultation Service<br/>- Urgency & Wait Calculation<br/>- Status Transitions"]
        AuditService["Audit Trail & Clinical Governance"]
    end

    subgraph Storage["Persistence & ML Assets"]
        Database[("SQLAlchemy Models<br/>PostgreSQL / SQLite")]
        SyntheticData["Synthetic OPD Generator<br/>(30,000 Record Pipeline)"]
        ModelArtifacts["Trained ML Pipelines<br/>(mediq_triage_model.joblib)"]
    end

    PatientUI <-->|REST API + WebSocket| AuthMiddleware
    ClinicalUI <-->|REST API + WebSocket| AuthMiddleware
    AdminUI <-->|REST API + WebSocket| AuthMiddleware

    AuthMiddleware --> QueueService
    QueueService --> TriageEngine
    DeterministicRules -->|Critical Escalation| QueueService
    ContradictionChecker -->|Abstention Alert| QueueService
    MLModel -->|Advisory Priority 1-5| QueueService
    QueueService --> Database
    QueueService --> WSManager
    WSManager -->|Live Status Updates| Clients
    AuditService --> Database
```

---

### 3. Database Entity Relationship Model

```mermaid
erDiagram
    USERS ||--o{ PATIENT_PROFILES : has
    USERS ||--o{ STAFF_PROFILES : has
    USERS ||--o{ AUDIT_LOGS : performs
    DEPARTMENTS ||--o{ STAFF_PROFILES : assigns
    DEPARTMENTS ||--o{ VISITS : directs
    DEPARTMENTS ||--o{ QUEUE_ENTRIES : contains

    PATIENT_PROFILES ||--o{ VISITS : initiates
    VISITS ||--o{ INTAKE_RECORDS : includes
    VISITS ||--o{ VITAL_SIGNS : measures
    VISITS ||--o{ TRIAGE_ASSESSMENTS : evaluates
    VISITS ||--o{ SAFETY_RULE_EVENTS : triggers
    VISITS ||--o{ QUEUE_ENTRIES : creates
    VISITS ||--o{ CONSULTATIONS : records

    TRIAGE_ASSESSMENTS ||--o{ TRIAGE_OVERRIDES : reviewed_by
    QUEUE_ENTRIES ||--o{ QUEUE_STATUS_EVENTS : tracks

    USERS {
        uuid id PK
        string email UK
        string hashed_password
        string role
        string full_name
        string phone
        boolean is_active
        datetime created_at
    }

    DEPARTMENTS {
        uuid id PK
        string name
        string code UK
        int current_wait_time_minutes
        boolean is_active
        int capacity
    }

    VISITS {
        uuid id PK
        uuid patient_id FK
        uuid department_id FK
        text chief_complaint
        string onset_duration
        text symptoms_text
        string visit_status
        datetime created_at
    }

    VITAL_SIGNS {
        uuid id PK
        uuid visit_id FK
        float heart_rate
        float respiratory_rate
        float systolic_bp
        float diastolic_bp
        float oxygen_saturation
        float temperature_c
        int pain_scale
        boolean is_valid
        text validation_errors
        datetime recorded_at
    }

    TRIAGE_ASSESSMENTS {
        uuid id PK
        uuid visit_id FK
        string model_version
        int suggested_priority
        uuid recommended_department_id FK
        text explanation
        float confidence_score
        boolean uncertainty_flag
        text missing_data_flags
        boolean abstention_flag
        text abstention_reason
        datetime created_at
    }

    SAFETY_RULE_EVENTS {
        uuid id PK
        uuid visit_id FK
        string rule_code
        string rule_name
        string severity
        text trigger_details
        datetime triggered_at
    }

    QUEUE_ENTRIES {
        uuid id PK
        uuid visit_id FK
        uuid department_id FK
        string queue_number
        int priority_level
        int confirmed_priority
        string status
        int estimated_wait_minutes
        int position_in_queue
        datetime called_at
        datetime consultation_started_at
        datetime completed_at
    }
```

---

### 4. Priority Categorization Scheme (Clinical Urgency Levels 1–5)

| Urgency Level | Category | Clinical Description | Maximum Safe Target Wait | Color Badge |
|---|---|---|---|---|
| **Priority 1** | **Resuscitation** | Immediate life threat; airway, breathing, severe shock, or cardiac arrest | **Immediate (0 min)** | Red (`#DC2626`) |
| **Priority 2** | **Emergent** | High risk, severe chest pain, respiratory distress, acute altered mental state | **< 15 minutes** | Orange (`#EA580C`) |
| **Priority 3** | **Urgent** | Moderate distress, 2+ system complaints, stable vitals requiring rapid workup | **< 30 minutes** | Yellow (`#CA8A04`) |
| **Priority 4** | **Less Urgent** | Low risk, stable chronic conditions, mild symptoms, single uncomplicated issue | **< 60 minutes** | Green (`#16A34A`) |
| **Priority 5** | **Non-Urgent** | Routine medication renewal, minor wound review, suture removal, preventative | **< 120 minutes** | Blue (`#2563EB`) |

---

### 5. Deterministic Safety-Rule Layer & Abstention Logic

The safety rules execute **before** ML inference and have statutory authority over model predictions:

1. **Vital Sign Extreme Thresholds**:
   - `CRIT_O2`: $SpO_2 < 85\%$ $\rightarrow$ Priority 1 (Resuscitation)
   - `CRIT_SBP_LOW`: $Systolic\ BP < 70\ mmHg$ $\rightarrow$ Priority 1 (Septic/Cardiogenic Shock)
   - `CRIT_SBP_HIGH`: $Systolic\ BP > 220\ mmHg$ or $Diastolic\ BP > 120\ mmHg$ $\rightarrow$ Priority 1 (Hypertensive Crisis)
   - `CRIT_HR`: $HR < 40\ bpm$ or $HR > 160\ bpm$ $\rightarrow$ Priority 1
   - `CRIT_RR`: $RR < 8\ bpm$ or $RR > 35\ bpm$ $\rightarrow$ Priority 1
   - `CRIT_TEMP`: $Temp > 41.5^\circ C$ or $< 34^\circ C$ $\rightarrow$ Priority 1

2. **Severe Red Flag Lexicon**:
   - Keywords: `"crushing chest pain"`, `"radiating to arm"`, `"anaphylaxis"`, `"stridor"`, `"unresponsive"`, `"sudden facial droop"`, `"thunderclap headache"`, `"massive hemoptysis"`.
   - Triggers automatic Priority 1 or 2 escalation.

3. **Data Quality & Abstention Engine**:
   - **Contradiction**: $Systolic\ BP \le Diastolic\ BP$ or $SpO_2 > 100\%$ or $HR \le 0$.
   - **Missing Critical Data**: Severe chest pain or acute breathlessness with NO vitals provided.
   - **Abstention Decision**: Returns `abstention_flag = True`, status `"Cannot assess safely — immediate nurse/clinical review required"`, prevents misleading model score, escalates to triage nurse desk immediately.

4. **Asymmetric ML Evaluation Cost**:
   - In synthetic training and evaluation, an under-triage error for a true emergency (e.g. predicting P3/P4 for a P1/P2 patient) is penalized with a **$20\times$ loss weight** relative to an over-triage error.

---

### 6. Step-by-Step Implementation Roadmap

- [x] **Phase 1: Environment & Architecture Plan** (Workspace analysis, git initialization, setup plan)
- [x] **Phase 2: Database Models, Alembic Migrations, & Core Config** (14 normalized SQLAlchemy models, Alembic migrations applied, JWT RBAC security)
- [x] **Phase 3: Modular Triage Service & Synthetic ML Pipeline** (Deterministic safety rules, contradiction detector, 10,000+ synthetic cohort, scikit-learn training, asymmetric 20x cost evaluation)
- [x] **Phase 4: Backend REST APIs & WebSocket Real-time Manager** (Auth, visits, queue management, clinician overrides, audit trails, analytics, channel-isolated WebSockets)
- [x] **Phase 5: Frontend Development (React + Tailwind + Lucide + Recharts)** (Patient Portal, Doctor/Nurse Clinical Dashboard, Admin Console, Public Kiosk)
- [x] **Phase 6: Comprehensive Automated Testing** (Pytest suite testing auth, RBAC, safety rules, triage inference, queue transitions, overrides, audit logs with 100% pass rate)
- [x] **Phase 7: End-to-End Live Demonstration & Verification** (13/13 acceptance criteria verified live via verify_live_e2e.py)
- [x] **Phase 8: Documentation & Repository Finalization** (Prinstine Git commit, master README.md, .env.example, architecture Mermaid diagrams)
