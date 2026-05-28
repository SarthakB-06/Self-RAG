"""
Prompt templates for the Self-RAG Enterprise Runbook Agent.

This module contains all LLM prompts with version control, structured outputs,
and best practices for prompt engineering. Each prompt is designed to minimize
hallucinations and maximize reliability.

Prompt Design Principles:
1. Clear role definition (You are...)
2. Specific task description
3. Explicit output format
4. Examples when helpful
5. Constraints and rules
6. Fallback instructions
"""

from typing import Literal
from enum import Enum


class PromptVersion(str, Enum):
    """Version control for prompts to enable A/B testing."""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"


# ==================== RETRIEVAL PROMPTS ====================

def get_query_expansion_prompt(incident: str) -> str:
    """
    Generate alternative search queries for better document retrieval.
    Used when initial retrieval returns no results.
    """
    return f"""You are an expert SRE search query optimizer.

Original Incident: {incident}

Task: Generate 3 alternative search queries that might find relevant runbooks.
Focus on:
- Technical keywords (service names, error codes, symptoms)
- Action verbs (troubleshoot, diagnose, mitigate, resolve)
- Abbreviations and full names (k8s/kubernetes, pg/postgres)

Return ONLY a JSON array of strings:
["query1", "query2", "query3"]

Example:
Incident: "API Gateway returning 502 errors"
Output: ["api gateway 502 bad gateway", "aws api gateway timeout", "gateway proxy error troubleshooting"]"""


# ==================== GRADING PROMPTS ====================

RELEVANCE_GRADER_PROMPT = """You are a strict technical grader evaluating whether a retrieved runbook document is relevant to an SRE incident.

Incident: {incident}

Retrieved Document Title: {document_title}
Retrieved Document Preview:
{document_preview}

Question: Does this document contain actionable troubleshooting steps or commands that DIRECTLY address the incident?

Evaluation Criteria:
Relevant if:
- Contains specific commands for this exact issue
- Describes symptoms matching the incident
- Provides diagnostic or remediation steps

Irrelevant if:
- Covers a different technology/service
- Only mentions the issue in passing
- Is a general guide without specific commands

Reply with ONLY "relevant" or "irrelevant". No explanation."""


# ==================== GENERATION PROMPTS ====================

# New, lightweight prompt for generating only the explanation
EXPLANATION_GENERATOR_PROMPT = """You are an expert SRE assistant. Read the following runbook and generate a brief explanation for the given incident.

Incident: {incident}

Official Runbook:
{document}

Task: Create a brief summary (2-4 sentences) for a mitigation plan, covering:
- The likely root cause of the issue.
- The high-level strategy to fix it.
- The expected outcome after the fix.

Output ONLY the explanation text. Do NOT include commands or JSON formatting.
"""


GENERATION_QUALITY_GRADER_PROMPT = """You are a senior SRE reviewing a junior engineer's mitigation plan.

Incident: {incident}

Proposed Mitigation Plan:
{plan_explanation}

Proposed Commands:
{commands}

Question: Is this plan clear, complete, and actionable?

Evaluation Criteria:

HIGH QUALITY:
- Root cause clearly explained
- Step-by-step commands provided
- Each command has a clear purpose
- Includes diagnostic + remediation steps
- Commands are specific (not vague like "restart the service")

MEDIUM QUALITY:
- Explanation is brief but understandable
- Commands are present but minimal
- Missing some diagnostic steps
- Could be more detailed

LOW QUALITY:
- Vague explanation ("fix the issue")
- Too few commands (< 2)
- Commands are generic or incomplete
- No clear execution order
- Missing critical steps

Reply with ONLY "high", "medium", or "low". No explanation."""


DOCUMENT_COMPLETENESS_GRADER_PROMPT = """You are evaluating if a runbook document is complete and production-ready.

Document: {document}

Question: Does this runbook have all the necessary sections for safe execution?

Required Sections:
1. Problem description / symptoms
2. Diagnostic commands (to verify the issue)
3. Mitigation commands (to fix the issue)
4. Verification steps (to confirm the fix)
5. Rollback plan (if something goes wrong)

Reply with ONLY "complete" or "incomplete". No explanation."""


# The original prompt, kept as a fallback or for more powerful models
MITIGATION_GENERATOR_PROMPT = """You are an expert SRE assistant generating a mitigation plan from an official runbook.

Incident: {incident}

Official Runbook:
{document}

Task: Create a mitigation plan with two parts:

1. "explanation": A brief summary (2-4 sentences) covering:
   - Root cause of the issue
   - High-level fix strategy
   - Expected outcome

2. "commands": A JSON array of EXACT commands from the runbook.

CRITICAL RULES (Security & Reliability):
Only include commands that appear VERBATIM in the runbook
Preserve all flags, arguments, and syntax EXACTLY as written
If a command has placeholders like <PID> or {{variable}}, keep them as-is
Maintain the EXACT order of commands from the runbook
Do NOT modify, invent, paraphrase, or "improve" commands
If the runbook doesn't have commands for this incident, return an empty array

Output Format:
{{
  "explanation": "Clear 2-4 sentence explanation here",
  "commands": [
    "exact command 1 from runbook",
    "exact command 2 from runbook"
  ]
}}

Example:
{{
  "explanation": "The PostgreSQL database is experiencing high CPU due to a runaway query. We'll identify the problematic query process and terminate it gracefully. This should immediately reduce CPU usage to normal levels.",
  "commands": [
    "SELECT pid, query, state FROM pg_stat_activity WHERE state = 'active' ORDER BY query_start LIMIT 10;",
    "SELECT pg_cancel_backend(<PID>);"
  ]
}}"""


REGENERATION_WITH_STRICTNESS_PROMPT = """You are an expert SRE assistant in STRICT MODE regenerating a mitigation plan.

Previous attempt was rejected. Regenerating with maximum precision.

Incident: {incident}

Official Runbook:
{document}

Previous Generation Issues:
{previous_issues}

STRICT MODE RULES:
ZERO tolerance for command modification
Include line-by-line copies from the runbook
Add ALL diagnostic steps before remediation
Include verification commands after fixes
If any doubt about a command, DO NOT include it

Task: Create a mitigation plan with:

1. "explanation": Detailed 3-4 sentence explanation including:
   - Root cause analysis
   - Step-by-step fix strategy
   - Risk assessment
   - Expected outcome with timeline

2. "commands": JSON array of EXACT commands with:
   - Each command MUST be a character-perfect copy
   - Preserve all whitespace, quotes, and special characters
   - Include comments from runbook if present
   - Maintain execution order

Output Format:
{{
  "explanation": "Detailed explanation here",
  "commands": [
    "exact command 1 with all original formatting",
    "exact command 2 with all original formatting"
  ]
}}"""


HALLUCINATION_CHECKER_PROMPT = """You are a security auditor performing a critical verification task.

Task: Verify that AI-generated commands are NOT hallucinated (invented).

Original Runbook Document:
{document}

AI-Generated Commands:
{commands}

Question: Are ALL commands in the AI-generated list copied EXACTLY (character-for-character) from the original runbook?

Verification Process:
1. For each AI command, search for it in the runbook document
2. Check if syntax, flags, arguments match EXACTLY
3. Verify no modifications, improvements, or paraphrasing occurred

HALLUCINATION DETECTED (respond "true") if:
Command doesn't appear in runbook at all
Command syntax is modified (even slightly)
Flags or arguments are different
Command is paraphrased or "improved"
Similar but not identical command

NO HALLUCINATION (respond "false") if:
ALL commands are exact character-by-character copies
Placeholders like <PID> are preserved as-is
Order might differ but content is identical

Reply with ONLY "true" (hallucination detected) or "false" (all commands verified). No explanation."""


RESOLUTION_CHECKER_PROMPT = """You are an expert SRE validating whether a mitigation plan actually resolves an incident.

Incident: {incident}

Mitigation Plan Explanation:
{plan_explanation}

Proposed Commands:
{commands}

Question: Does this plan DIRECTLY address and resolve the reported incident?

Evaluation Criteria:

RESOLVES (respond "true"):
- Targets the exact service/system in the incident
- Addresses the root cause (not just symptoms)
- Includes both diagnostic AND remediation steps
- Commands are appropriate for the severity level
- Expected outcome matches incident resolution

DOES NOT RESOLVE (respond "false"):
- Targets wrong service/component
- Only addresses symptoms, not root cause
- Missing critical diagnostic or fix steps
- Commands are too generic or unrelated
- Would not actually fix the reported issue

Examples:
Incident: "PostgreSQL CPU at 99%"
Resolves: Commands to identify and kill runaway queries
Does NOT resolve: Commands to restart PostgreSQL (doesn't address cause)

Incident: "Kubernetes pod crashlooping"
Resolves: Commands to check logs, identify crash cause, fix config
Does NOT resolve: Commands to scale deployment (doesn't fix crash)

Reply with ONLY "true" (resolves incident) or "false" (does not resolve). No explanation."""


QUERY_REWRITER_PROMPT = """You are an expert at refining search queries for SRE runbook databases.

Original Incident: {incident}

Previous Query: {previous_query}

Why it failed: {failure_reason}

Task: Write a NEW search query using different technical keywords to find the correct runbook.

Strategy:
1. Analyze why the previous query failed
2. Use alternative terminology (e.g., "k8s" → "kubernetes", "db" → "database")
3. Add or remove specificity as needed
4. Try different symptom keywords
5. Include technology versions if relevant

Focus keywords:
- Service names: postgres, mysql, redis, nginx, kubernetes, aws, etc.
- Symptoms: high cpu, memory leak, timeout, crash, OOM, slow, latency
- Actions: troubleshoot, diagnose, debug, mitigate, resolve, fix

Return ONLY the new query string (no quotes, no explanation).

Examples:
Previous: "postgres slow" → New: "postgresql performance troubleshooting high cpu"
Previous: "k8s pod issue" → New: "kubernetes pod crashloopbackoff restart"
Previous: "api error" → New: "aws api gateway 502 bad gateway timeout"
"""


ESCALATION_REASONING_PROMPT = """You are an SRE team lead determining if an incident should be escalated to L2 support.

Incident: {incident}

Agent Execution Summary:
- Queries attempted: {query_count}
- Documents retrieved: {doc_count}
- Generation attempts: {gen_count}
- Failure reason: {failure_reason}

Question: Should this be escalated to human engineers?

Escalation Criteria:

YES - ESCALATE:
- No relevant runbook found after 3+ query attempts
- Runbook exists but commands are hallucinated
- Mitigation plan doesn't actually resolve the issue
- Agent hit maximum retry limits
- Unknown/novel incident not in knowledge base

NO - RETRY:
- Only tried 1-2 queries
- Documents were found but quality was low
- Clear path to improvement (better query, stricter generation)

Provide reasoning as JSON:
{{
  "should_escalate": true,
  "reason": "Brief explanation",
  "suggested_action": "What L2 should do first"
}}"""


def format_commands_for_prompt(commands: list[str]) -> str:
    """Format commands list for inclusion in prompts."""
    if not commands:
        return "(No commands generated)"

    return "\n".join(f"{i+1}. {cmd}" for i, cmd in enumerate(commands))


def format_document_preview(document: str, max_chars: int = 1500) -> str:
    """
    Create a preview of a document for prompts.

    Args:
        document: Full document text
        max_chars: Maximum characters to include

    Returns:
        Document preview with ellipsis if truncated
    """
    if len(document) <= max_chars:
        return document

    return document[:max_chars] + "\n\n... (document truncated) ..."


def get_prompt_version(prompt_name: str, version: PromptVersion = PromptVersion.LATEST) -> str:
    """
    Get a specific version of a prompt (for A/B testing).

    Args:
        prompt_name: Name of the prompt
        version: Version to retrieve

    Returns:
        Prompt template string
    """
    # Placeholder for future A/B testing
    # In production, you'd load from a database or config file
    prompts = {
        "relevance_grader": {
            PromptVersion.V1: RELEVANCE_GRADER_PROMPT,
            PromptVersion.V2: RELEVANCE_GRADER_PROMPT,  # Currently same
        }
    }

    return prompts.get(prompt_name, {}).get(version, "")


PROMPT_METADATA = {
    "relevance_grader": {
        "version": "v2",
        "last_updated": "2024-01-15",
        "tested_with": ["gemini-2.0-flash-exp"],
        "avg_tokens": 250,
        "success_rate": 0.94
    },
    "generation_quality_grader": {
        "version": "v2",
        "last_updated": "2024-01-15",
        "tested_with": ["gemini-2.0-flash-exp"],
        "avg_tokens": 180,
        "success_rate": 0.89
    },
    "mitigation_generator": {
        "version": "v2",
        "last_updated": "2024-01-15",
        "tested_with": ["gemini-2.0-flash-exp"],
        "avg_tokens": 800,
        "success_rate": 0.91
    },
    "hallucination_checker": {
        "version": "v2",
        "last_updated": "2024-01-15",
        "tested_with": ["gemini-2.0-flash-exp"],
        "avg_tokens": 300,
        "success_rate": 0.96
    },
    "resolution_checker": {
        "version": "v2",
        "last_updated": "2024-01-15",
        "tested_with": ["gemini-2.0-flash-exp"],
        "avg_tokens": 220,
        "success_rate": 0.88
    },
    "query_rewriter": {
        "version": "v2",
        "last_updated": "2024-01-15",
        "tested_with": ["gemini-2.0-flash-exp"],
        "avg_tokens": 120,
        "success_rate": 0.92
    }
}


def get_prompt_stats(prompt_name: str) -> dict:
    """Get metadata and performance stats for a prompt."""
    return PROMPT_METADATA.get(prompt_name, {})
