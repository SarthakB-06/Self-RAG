import streamlit as st
import time
import pandas as pd
import requests
import os

# Must be the first Streamlit command
st.set_page_config(
    page_title="Enterprise DevOps SRE Copilot",
    page_icon="ðŸ› ï¸",
    layout="wide"
)

# Configuration for FastAPI Backend
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/diagnose")
HEALTH_URL = os.getenv("HEALTH_URL", "http://localhost:8000/health")


def check_backend():
    with st.spinner("Waking up FastAPI backend (Render free-tier cold start may take up to 50 seconds)..."):
        for attempt in range(3):
            try:
                # Increased timeout to 60s to account for waking up dead API
                response = requests.get(HEALTH_URL, timeout=60)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                time.sleep(2)
        return False


def main():
    st.title("🛠️ Enterprise DevOps SRE Copilot")

    is_backend_up = check_backend()
    if is_backend_up:
        st.success("🔌 Connected to FastAPI Backend successfully!")
    else:
        st.error(
            "⚠️ Cannot connect to FastAPI Backend. Ensure `uvicorn api:app --reload` is running on port 8000.")
        st.stop()

    st.markdown("""
        This UI is completely decoupled from the LangGraph logic! It sends an HTTP POST request to our FastAPI backend 
        which runs the robust Self-RAG architecture to help diagnose and resolve infrastructure incidents.
    """)

    # Incident Input Area
    st.subheader("Report an Incident")
    incident_query = st.text_area(
        "Describe the infrastructure issue you are facing:",
        placeholder="e.g. EC2 instance cannot connect to RDS database from private subnet",
        height=100
    )

    if st.button("Diagnose via API & Generate Runbook", type="primary"):
        if not incident_query.strip():
            st.warning("Please enter an incident description.")
            return

        with st.spinner("Sending payload to FastAPI backend and analyzing incident..."):
            start_time = time.time()
            try:
                # Send the POST request to the API
                response = requests.post(
                    API_URL, json={"incident_description": incident_query})
                response.raise_for_status()

                # Parse the JSON response mapping perfectly to our IncidentResponse Pydantic model
                result = response.json()
                calc_time = time.time() - start_time
                st.success(
                    f"Analysis complete via API in {calc_time:.2f} seconds!")

            except Exception as e:
                st.error(f"Failed to connect to backend API: {e}")
                return

        # Layout for results
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📄 Mitigation Plan")
            if result.get("requires_escalation") and not result.get("is_resolved"):
                st.error(
                    "⚠️ Escalation Required: The agent could not confidently resolve this issue with the current runbooks.")
            else:
                st.success("✅ Confidence Threshold Met")

            st.write(result.get("plan_explanation",
                     "No explanation generated."))

            commands = result.get("executable_commands", [])
            if commands:
                st.subheader("💻 Executable Commands")
                st.code("\n".join(commands), language="bash")

        with col2:
            st.subheader("📊 Execution Metrics")

            # Metadata
            st.markdown("### Process Overview")
            st.info(
                f"**Path Taken:** {' → '.join(result.get('execution_path', []))}")
            st.metric("Total Duration (ms)",
                      f"{result.get('total_duration_ms', 0):.2f}")
            st.metric("LLM Calls", result.get("total_llm_calls", 0))

            # Performance Timing
            st.markdown("### Timing Breakdown (s)")
            timing_data = {
                "Step": ["Retrieval", "Relevance Grading", "Generation", "Hallucination Check", "Resolution Check"],
                "Time (s)": [
                    round(result.get('retrieval_time', 0.0), 2),
                    round(result.get('relevance_grading_time', 0.0), 2),
                    round(result.get('generation_time', 0.0), 2),
                    round(result.get('hallucination_check_time', 0.0), 2),
                    round(result.get('resolution_check_time', 0.0), 2),
                ]
            }
            st.dataframe(pd.DataFrame(timing_data), hide_index=True)

            st.metric("Retrieval Context Score",
                      f"{result.get('retrieval_score', 0.0):.4f}")

            if result.get("errors"):
                st.markdown("### Errors")
                for err in result["errors"]:
                    st.error(err)


if __name__ == "__main__":
    main()
