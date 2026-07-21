# Halcyon Credit — Demo Applicant Profiles

Three ready-to-use applicant profiles for presenting the Halcyon Credit underwriting copilot.
Enter these values in the **Loan Application** form of the UI (`ui/index.html`).

> **Note:** The Application ID and Verification Status are **not** entered — the system
> generates the ID and determines verification status automatically from the employment profile.

---

## 1. ✅ Priya Sharma — Expected Decision: **APPROVE**

A strong, low-risk applicant: stable salaried employment, healthy income, low debt load,
clean credit history. Passes every policy check and lands in the Low risk band.

| Field | Value |
|---|---|
| **Full Name** | Priya Sharma |
| **Loan Amount** | 12000 |
| **Loan Purpose** | Home Improvement |
| **Loan Term** | 36 months |
| **Home Ownership** | Mortgage |
| **Annual Income** | 72000 |
| **Employment Type** | Salaried |
| **Months at Current Employer** | 84 |
| **Existing Monthly Debt Payments** | 870 |
| **Credit Score** | 720 |
| **Revolving Utilisation** | 22 |
| **Delinquencies (last 2 years)** | 0 |
| **Open Credit Accounts** | 8 |
| **Credit History Age** | 96 |
| **Public Records** | 0 |
| **Credit Inquiries (last 6 months)** | 1 |

**Why it approves:** DTI is only 14.5% (well under the 40% hard stop), loan-to-income is 0.17,
credit score 720 with zero delinquencies and 8 years of history. No hard stops, no advisory
flags — a textbook approval.

---

## 2. ❌ Rakesh Verma — Expected Decision: **DECLINE**

An over-leveraged applicant seeking to consolidate debt. His debt-to-income ratio breaches
the **POL-001 hard stop (DTI > 40%)**, which mandates an automatic decline regardless of
anything else.

| Field | Value |
|---|---|
| **Full Name** | Rakesh Verma |
| **Loan Amount** | 20000 |
| **Loan Purpose** | Debt Consolidation |
| **Loan Term** | 60 months |
| **Home Ownership** | Rent |
| **Annual Income** | 40000 |
| **Employment Type** | Self-Employed |
| **Months at Current Employer** | 30 |
| **Existing Monthly Debt Payments** | 1500 |
| **Credit Score** | 610 |
| **Revolving Utilisation** | 78 |
| **Delinquencies (last 2 years)** | 1 |
| **Open Credit Accounts** | 6 |
| **Credit History Age** | 70 |
| **Public Records** | 0 |
| **Credit Inquiries (last 6 months)** | 4 |

**Why it declines:** DTI = (1,500 × 12) / 40,000 = **45%**, triggering hard stop **POL-001**.
Also flagged: debt consolidation with DTI > 35% (**POL-004**), 78% card utilisation,
and 4 recent credit inquiries. The hard stop is enforced deterministically — the LLM
cannot override it.

---

## 3. ⚠️ Ananya Patel — Expected Decision: **REFER**

A young, thin-file applicant: responsible so far, but with only 14 months of credit
history and 2 open accounts, there isn't enough data to auto-decide. Policy **POL-005**
mandates that thin-file applicants are referred to a human underwriter — never auto-declined.

| Field | Value |
|---|---|
| **Full Name** | Ananya Patel |
| **Loan Amount** | 10000 |
| **Loan Purpose** | Car |
| **Loan Term** | 36 months |
| **Home Ownership** | Rent |
| **Annual Income** | 48000 |
| **Employment Type** | Salaried |
| **Months at Current Employer** | 20 |
| **Existing Monthly Debt Payments** | 400 |
| **Credit Score** | 690 |
| **Revolving Utilisation** | 35 |
| **Delinquencies (last 2 years)** | 0 |
| **Open Credit Accounts** | 2 |
| **Credit History Age** | 14 |
| **Public Records** | 0 |
| **Credit Inquiries (last 6 months)** | 2 |

**Why it refers:** Credit history of 14 months (< 24) and only 2 open accounts (< 3) mark
this as a **thin file** (**POL-005** advisory flag). Her DTI (10%) and score (690) are fine,
but the system routes thin files to a human underwriter rather than deciding automatically —
demonstrating responsible AI guardrails.

---

## Presentation flow suggestion

1. **Priya (APPROVE)** — show the happy path: agents verify, risk model scores Low,
   judge passes the faithfulness gate, clean approval with cited reasons.
2. **Rakesh (DECLINE)** — show policy enforcement: the deterministic POL-001 hard stop
   in the Risk Model Output section, and decline reasons citing the DTI breach.
3. **Ananya (REFER)** — show the human-in-the-loop guardrail: thin-file flag,
   REFER verdict, and explain that the AI knows when *not* to decide.

For each run, highlight the **Full Audit Trace** and **Evaluation (Faithfulness Gate)**
sections — every claim in the reasons is verified against source data before the record
is written.