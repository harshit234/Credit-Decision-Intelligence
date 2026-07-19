# Halcyon Credit — Project Handover & Context Summary

This document serves as a complete context-transfer file summarizing the progress, team structure, and future roadmap for the **Halcyon Credit Agentic Underwriting Copilot** (Capstone Project 02 - Futurense × IIT Gandhinagar).

---

## 1. Team Structure & Git Attribution

To ensure proper tracking and GitHub commit attribution, the team's work distribution is mapped as follows:

| Member | GitHub | Email | Primary Domain |
|---|---|---|---|
| **Harshit** | `harshit234` | harshitgautam05@gmail.com | ML Model Training (LightGBM/XGBoost) + API Integration |
| **Himkar** | `Himkar001` | himkarvashistha13003@gmail.com | ChromaDB Policy KB, Prompt Engineering & Docs |
| **Aditya** | `sedoCdA` | adiarora1301@gmail.com | Project Scaffolding, LangGraph State, & Data Pipelines |
| **Ayush** | `aloo7` | aloolifts@gmail.com | Model Evaluation, Feature Validation, & UI/Frontend |

---

## 2. What We Have Covered (Stages 1 & 2 Completed)

We have successfully completed the foundational data engineering, machine learning, and policy infrastructure phases. Everything is currently committed and merged into the `dev` branch.

### The Dataset Pivot & Data Engineering
- **The Pivot:** Switched from LendingClub to the **American Express Default Prediction** time-series dataset (15GB, 5.5M rows) to capture behavioral "default spirals" and break the 0.75 PR-AUC mathematical limit of static data.
- **Aggregation Pipeline:** Built a memory-safe pipeline (`build_amex_dataset.py`) to process 13-month histories for 298,828 sampled customers.
- **Feature Engineering:** Extracted temporal statistics and built 8 domain-specific composites (e.g., `delinquency_escalation`), resulting in a **1,136-feature schema**.

### Core ML Risk Model
- **Model:** Trained a LightGBM architecture (`train_risk_model.py`) optimized for the high-dimensional, temporal feature space.
- **Results:** Shattered the >0.90 PR-AUC target, achieving **PR-AUC: 0.9378**, **ROC-AUC: 0.9610**, and **Default Recall: 91.69%**.
- **Artifacts Saved:** The trained model (`lgbm_halcyon_v1.txt`) and schema (`feature_schema_v1.json`) are persisted.

### Policy Knowledge Base
- **Infrastructure:** Ingested Halcyon's credit policies into a local **ChromaDB** vector database via `build_policy_kb.py`.

---

## 3. What We Need to Do Next (Stage 3: LangGraph Agents)

With the ML model trained and the policy database populated, Stage 3 is purely focused on **Agentic Engineering**. We will build a multi-agent LangGraph system to orchestrate the underwriting process.

**Work Distribution:**
1. **Aditya (Graph State):** Defines the `UnderwritingState(TypedDict)` containing applicant data, risk score, policy flags, and final decision. Builds the Supervisor routing logic.
2. **Harshit (Risk Node):** Builds the `RiskScoringAgent` to load the LightGBM model, execute inference, and append the risk score and SHAP explanation to the Graph State.
3. **Himkar (Policy Node):** Builds the `PolicyAgent` using LangChain to query ChromaDB for relevant rules and append policy flags (e.g., "Refer: DTI > 40%") to the Graph State.
4. **Ayush (Evaluator):** Builds the pipeline to run the `amex_sample_50rows.csv` test set through the complete LangGraph network to validate end-to-end decisions.

*Git Branch:* `stage-3/langgraph-agents`

---

## 4. Further Stages (Stage 4 & 5 Roadmap)

To bring the Capstone to the finish line, the final stages will integrate the agentic backend with a user-facing application and finalize the academic deliverables.

### Stage 4: UI & API Deployment (Frontend & Backend Integration)
Once the LangGraph orchestration works in the terminal, we will expose it to users.
- **Harshit:** Wraps the LangGraph underwriting pipeline in a **FastAPI** backend to expose REST endpoints (e.g., `/api/v1/underwrite`).
- **Ayush:** Builds the interactive **Streamlit or Next.js Frontend** where a loan officer can input applicant data, see the LightGBM risk score, read the Policy Agent's reasoning, and approve/decline the loan.
- **Aditya & Himkar:** Ensure containerization (`Dockerfile` and `docker-compose.yml`) so the app can run on any machine easily.
*Git Branch:* `stage-4/ui-deployment`

### Stage 5: Final Delivery, Evaluation, & Documentation
The final polish for the Futurense × IIT Gandhinagar panel presentation.
- **Himkar & Aditya:** Finalize the GitHub repository `README.md`, update the Architecture Diagrams, and polish the `dataset_card.md`.
- **Harshit & Ayush:** Prepare the Capstone Presentation (PPTX), record the final demonstration video of the Underwriting Copilot in action, and ensure all evaluation metrics (PR-AUC, Agentic Accuracy) are heavily highlighted.
*Git Branch:* `stage-5/final-delivery` (Merged directly to `main` for release)
