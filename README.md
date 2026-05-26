# Self-RAG Incident Resolution Agent

This project implements a sophisticated, autonomous agent for automated incident resolution using a Self-Correcting Retrieval-Augmented Generation (Self-RAG) framework. The agent leverages a knowledge base of enterprise runbooks to diagnose incidents and generate actionable mitigation plans.

It is built with Python, LangGraph, and FAISS, and is designed to work with both local (Ollama) and production-grade (Gemini) Large Language Models.

![Agent Architecture](https://i.imgur.com/your-architecture-diagram.png)  <!-- Placeholder: You can create and upload a diagram of your graph -->

---

## 🚀 Key Features

- **Automated Incident Diagnosis**: Ingests a plain-text incident description (e.g., "High CPU utilization on EC2 instance").
- **Self-Correcting Retrieval**:
    - Retrieves relevant runbooks from a FAISS vector store.
    - **Grades** the relevance of retrieved documents and re-queries if they are not a good fit.
    - **Rewrites** queries automatically to improve search results, attempting to find relevant information up to 3 times before escalating.
- **Robust Plan Generation**:
    - Generates a human-readable explanation of the mitigation plan.
    - Extracts a sequence of executable shell commands directly from the runbook.
- **Built-in Quality Gates**:
    - **Hallucination Check**: Ensures that all generated commands are explicitly present in the source runbook, preventing the LLM from inventing steps.
    - **Resolution Check**: Validates that the generated plan is actionable and contains valid diagnostic or mitigation commands.
- **Autonomous Workflow**: Uses LangGraph to manage a stateful, multi-step execution graph, enabling complex logic like loops and conditional branching.
- **Human Escalation**: If the agent cannot find a relevant runbook or generate a valid plan after multiple attempts, it automatically escalates the incident with a detailed summary of its attempts.
- **Comprehensive Metrics**: Includes a dedicated test suite (`test_metrics.py`) to evaluate the agent's performance across a range of incidents, producing resume-ready statistics.

---

## 📊 Performance Metrics

The agent was evaluated against a test suite of 10 diverse and challenging incidents. The results demonstrate a high degree of autonomy and accuracy.

| Metric                     | Value                  | Description                                                 |
| -------------------------- | ---------------------- | ----------------------------------------------------------- |
| **Success Rate**           | **XX.X%**              | % of incidents resolved without needing human escalation.   |
| **Avg. Resolution Time**   | **XX.XXs**             | Average time from incident ingestion to plan generation.    |
| **Avg. LLM Calls**         | **X.X**                | Average number of LLM calls per incident (lower is better). |
| **Avg. Retrieval Score**   | **X.XX / 5.0**         | Average relevance score of the final retrieved document.    |

*(Note: Replace the 'XX.X' placeholders with the actual results from running `python test_metrics.py`)*

---

## 🛠️ Tech Stack

- **Core Framework**: Python, LangGraph
- **LLM Orchestration**: LangChain
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLMs Supported**:
    - Google Gemini (`gemini-1.5-flash`) for production
    - Ollama (`qwen3.5:4b`) for local development
- **CLI**: `rich` for enhanced terminal UI

---

## ⚙️ Setup and Usage

### Prerequisites

- Python 3.10+
- An active Gemini API key (for production mode)
- Ollama installed (for local development)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/self-rag-agent.git
    cd self-rag-agent
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure your environment:**
    - Create a `.env` file in the project root.
    - Add your Gemini API key:
      ```
      GEMINI_API_KEY="your_api_key_here"
      ```

### Running the Agent

**1. Interactive Mode (Single Incident)**

This is the primary way to use the agent.

```bash
python main.py "Your incident description here"

# Example:
python main.py "SSH connection to EC2 instance is timing out"
```

**2. Local Development (with Ollama)**

To use a local model and save on API costs:

-   **Start the Ollama server** in a separate terminal:
    ```bash
    ollama serve
    ```
-   **Set the environment** in your `.env` file:
    ```
    ENVIRONMENT=development
    ```
-   **Run the agent** as usual. It will automatically connect to Ollama.

**3. Running the Performance Test Suite**

To generate the performance metrics report:

```bash
python test_metrics.py
```

---

## 🏗️ Architecture

The agent operates as a state machine orchestrated by LangGraph.

1.  **Retrieve**: The initial incident description is used to query the FAISS vector store for relevant runbooks.
2.  **Grade**: The retrieved documents are graded for relevance by the LLM.
3.  **Decide (After Grading)**:
    - If a document is **relevant**, the workflow proceeds to **Generate**.
    - If **irrelevant**, the agent attempts to **Rewrite** the query. If it has already retried multiple times, it will **Escalate**.
4.  **Generate**: The LLM generates a mitigation plan and extracts executable commands from the relevant runbook.
5.  **Validate (Hallucination & Resolution)**: The generated plan goes through two fast, rule-based checks to ensure it is grounded in the source document and is actionable.
6.  **Decide (After Validation)**:
    - If the plan is valid, the process **Ends** and the plan is presented to the user.
    - If invalid, the agent will **Escalate**.

This self-correcting loop of `Retrieve → Grade → Rewrite` is the core of the Self-RAG pattern and makes the agent highly resilient to ambiguous queries or imperfect document matches.
