"""
Generation node for creating mitigation plans from runbooks.
"""

import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from src.state import AgentState
from src.prompts import MITIGATION_GENERATOR_PROMPT
from src.config import Config, logger
from src.llm_factory import create_llm

llm = create_llm()


def generate_mitigation_plan(state: AgentState) -> Dict[str, Any]:
    """
    Generate a mitigation plan from the selected runbook.

    Args:
        state: Current agent state with documents

    Returns:
        Updated state with plan_explanation and executable_commands
    """

    logger.info("Generating mitigation plan from selected document")

    doc_index = state["selected_document_index"]
    document = state["documents"][doc_index]

    prompt = MITIGATION_GENERATOR_PROMPT.format(
        incident=state["initial_incident"],
        document=document
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_content = response.content

        # More robust JSON parsing
        try:
            # Find the start and end of the JSON block
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_content[json_start:json_end]
                result = json.loads(json_str)
                plan_explanation = result.get("explanation", "")
                executable_commands = result.get("commands", [])

                logger.info(
                    f"Generated mitigation plan with explanation and {len(executable_commands)} commands")

                return {
                    "plan_explanation": plan_explanation,
                    "executable_commands": executable_commands,
                    "generation_retry_count": state["generation_retry_count"] + 1,
                    "total_llm_calls": state["total_llm_calls"] + 1,
                    "execution_path": state["execution_path"] + ["generate"]
                }
            else:
                # If no JSON object is found, trigger the error path
                raise json.JSONDecodeError(
                    "No JSON object found in response", response_content, 0)

        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse LLM response as JSON: {response_content}")
            return {
                "plan_explanation": "",
                "executable_commands": [],
                "errors": state["errors"] + ["Generation error: Invalid LLM response format"],
                "execution_path": state["execution_path"] + ["generate_parse_error"]
            }
    except Exception as e:
        logger.error(f"Error during mitigation plan generation: {e}")
        return {
            "plan_explanation": "",
            "executable_commands": [],
            "errors": state["errors"] + [f"Generation error: {str(e)}"],
            "execution_path": state["execution_path"] + ["generate_error"]
        }
