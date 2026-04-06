# Technical Specification: SyncPulse AI
**Objective:** A Privacy-First, AI-Mediated Management Ecosystem that synchronizes technical output (GitHub) with team communication (Multi-platform Chatbots) to optimize group project orchestration in Education 4.0.

---

## 1. System Architecture & Components

### 1.1 Central Management Platform (Web)
*   **Instructor Dashboard:** 
    *   Entity Relationship: `Course -> Group -> Members`.
    *   Integration: Link specific GitHub Repositories to Groups.
    *   Analytics: View aggregated "Health Status" (Traffic Light system) across all groups.
*   **Student Dashboard:**
    *   Visualization: Interactive heatmap of contributions (Commits + Chat volume).
    *   Health Meter: Visual display of current group synchronization and progress.

### 1.2 Multi-Platform Chatbot Layer
*   **Protocol Support:** Integration with WhatsApp, Discord, WeChat, and QQ.
*   **Data Ingestion:** Real-time collection of chat messages and GitHub Webhook events (Push, PR, Issue).
*   **Ephemeral Listener:** A message-passing interface that triggers the Privacy Filter before any data persistence.

### 1.3 The Privacy & Anonymization Engine (Middleware)
*   **PII Stripper:** A Regex/NLP-based filter that replaces names, emails, and phone numbers with generic tokens (e.g., `[Student_A]`, `[Student_B]`).
*   **Non-Persistence Policy:** Original raw chat text must **never** be stored in the cloud/database. Only anonymized metadata and LLM-generated summaries are persisted.

---

## 2. Functional Requirements

### 2.1 Automated Summarization & Progress Tracking
*   **GitHub Integration:** Pull commit messages and PR descriptions to map technical progress.
*   **Contextual Synthesis:** Use an LLM to marry Chat activity with Code activity to generate a "Current State" summary.
*   **Verification Loop:** Weekly summaries are sent to the Group Chat for member validation before being visible to the Instructor.

### 2.2 Intelligent Intervention & Nudging
*   **Stall Detection:** Trigger internal "Yellow Alerts" if zero commits/messages are detected for `X` days.
*   **Workload Imbalance:** Analyze semantic contribution (e.g., Architecting vs. Documentation) to detect "Freeloading" or "Hero-coding."
*   **Proactive Nudges:** Send private, supportive AI suggestions to under-contributing members or teams lacking direction.

### 2.3 Escalation & Permission Logic
*   **Red Alert Protocol:** In cases of severe stagnation, the Instructor is notified of a "Warning Status."
*   **Transcript Lockdown:** Original discussion text is managed locally (client-side) by the group leader. It is only accessible via a "Permission Request" flow, ensuring transparency.
*   **Transparency Log:** A public log within the bot showing exactly when the bot is "Analyzing" or "Reporting" to ensure students know the operational status.

### 2.4 Collaboration Modes
*   **Pause Mode (Privacy Shield):** A consensus-driven state where the AI stops data collection for a specific duration (e.g., 2 hours) to allow unregulated brainstorming.
*   **Post-Mortem Engine:** Upon project completion, the system generates a **Peer Review Draft** based on the semester's contribution data, categorizing work by type (Logic, UI, Docs).

---

## 3. Data Flow and Logic

### 3.1 The Analysis Pipeline
1.  **Ingest:** Webhook (GitHub) + API (Chat).
2.  **Clean:** Strip PII -> Anonymize IDs.
3.  **Process:** Send Anonymized Payload to LLM (e.g., GPT-4o / Claude 3.5).
4.  **Summarize:** Prompt: *"Identify current roadblocks, key achievements, and workload distribution for [Student_A...N]."*
5.  **Output:** Store Summary -> Update Dashboard -> Delete Raw Fragment.

### 3.2 Health Status Calculation
*   **Metric A:** Frequency/Consistency of GitHub Commits.
*   **Metric B:** Semantic alignment between Chat discussion and Code output.
*   **Metric C:** Equity of interaction (e.g., Gini Coefficient of chat participation).

---

## 4. Technical Constraints for Vibe Coding
*   **Tech Stack Recommended:** Next.js (Frontend), Supabase/PostgreSQL (Backend), Tailwind CSS (UI), LangChain/OpenAI API (LLM).
*   **Modularity:** Ensure the "Anonymizer" is a standalone utility to allow for easy testing/auditing.
*   **Bot Frameworks:** Use `Probot` for GitHub and `Matrix` or `Twilio` for multi-channel messaging APIs.

---

**Instruction to AI Agent:** 
*Focus on building the Privacy Filter and the GitHub/Chat data-mapping logic first. The UI for the Instructor Dashboard should follow the "Traffic Light" UX pattern.*