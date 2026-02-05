"""
LLM Request/Response Logger
===========================

Captures detailed logs of LLM interactions in two formats:
1. RAW: Unformatted request/response (direct API calls)
2. FORMATTED: DSPy-formatted prompts and completions

Logs are written to:
- outputs/<run>/llm_logs/phase1_scenario_generation.log
- outputs/<run>/llm_logs/phase2_report_generation.log
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Module logger
logger = logging.getLogger(__name__)

# Constants
LLM_LOG_DIR = "llm_logs"


class LLMInteractionLogger:
    """
    Comprehensive LLM interaction logger for both raw and DSPy-formatted calls.
    
    Captures:
    - Timestamp
    - Phase (1=scenario, 2=report)
    - Signature name
    - Input fields (raw)
    - DSPy-formatted prompt (messages)
    - Raw API response
    - Parsed output fields
    - Token usage (if available)
    - Latency
    
    Usage:
        logger = LLMInteractionLogger(output_dir, phase=1)
        logger.log_request(signature_name, input_fields)
        logger.log_response(dspy_result, raw_response)
    """
    
    def __init__(self, output_dir: Optional[Path] = None, phase: int = 1, report_id: str = "UNKNOWN"):
        """
        Initialize the logger.
        
        Args:
            output_dir: Directory to save logs
            phase: Pipeline phase (1=Config/Sim, 2=Analysis/Report)
            report_id: Report identifier for log grouping
        """
        self.output_dir = output_dir or Path("outputs/logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase = phase
        self.report_id = report_id
        self.phase_name = "scenario_generation" if phase == 1 else "report_generation"
        
        # Use the central logger
        from src.core.logging_config import get_logger
        self.logger = get_logger(f"AeroGuardian.LLM.{self.phase_name}")
        
        # Request tracking
        self._request_start_time: Optional[datetime] = None
        self._current_request: Dict = {}
        
        self.logger.info(f"LLM Interaction Logger initialized for {self.report_id} (Phase {phase})")

    def log_request_start(
        self,
        signature_name: str,
        input_fields: Dict[str, Any],
    ) -> None:
        """
        Log the start of an LLM request.
        
        Args:
            signature_name: DSPy signature class name
            input_fields: Input field values passed to the signature
        """
        self._request_start_time = datetime.now()
        self._current_request = {
            "timestamp": self._request_start_time.isoformat(),
            "incident_id": self.report_id,
            "phase": self.phase,
            "signature_name": signature_name,
            "input_fields": input_fields,
        }
        
        # Log to main log file
        self.logger.info(f"""
================================================================================
LLM REQUEST START - {self.report_id}
TIMESTAMP: {datetime.now().isoformat()}
TYPE: generic
model: gpt-4
================================================================================

INPUT FIELDS:
{self._format_dict(input_fields)}
{'='*80}
""")

    def log_response(
        self,
        dspy_result: Any,
        raw_history: Optional[List[Dict]] = None,
    ) -> None:
        """
        Log the LLM response (both raw and formatted).
        
        Args:
            dspy_result: The result object from DSPy call
            raw_history: Raw LM history entries (from dspy.settings.lm.history)
        """
        end_time = datetime.now()
        latency_ms = 0
        if self._request_start_time:
            latency_ms = (end_time - self._request_start_time).total_seconds() * 1000
        
        # Extract output fields from DSPy result
        output_fields = {}
        if dspy_result:
            for attr in dir(dspy_result):
                if not attr.startswith('_') and not callable(getattr(dspy_result, attr)):
                    try:
                        value = getattr(dspy_result, attr)
                        if isinstance(value, (str, int, float, bool, list, dict)):
                            output_fields[attr] = value
                        else:
                            output_fields[attr] = str(value)
                    except Exception:
                        pass
        
        # Extract formatted prompt and raw response from history
        formatted_prompt = ""
        raw_response = ""
        token_usage = {}
        
        if raw_history:
            try:
                last_entry = raw_history[-1] if raw_history else {}
                
                # Extract messages (formatted prompt)
                if 'messages' in last_entry:
                    for msg in last_entry['messages']:
                        role = msg.get('role', 'unknown').upper()
                        content = msg.get('content', '')
                        formatted_prompt += f"[{role}]\n{content}\n\n"
                elif 'prompt' in last_entry:
                    formatted_prompt = last_entry['prompt']
                
                # Extract response
                if 'response' in last_entry:
                    resp = last_entry['response']
                    if hasattr(resp, 'text'):
                        raw_response = resp.text
                    elif hasattr(resp, 'choices') and resp.choices:
                        raw_response = resp.choices[0].message.content if hasattr(resp.choices[0], 'message') else str(resp.choices[0])
                    else:
                        raw_response = str(resp)
                elif 'outputs' in last_entry:
                    raw_response = str(last_entry['outputs'])
                
                # Token usage - ROBUST EXTRACTION (Fix for serialization error)
                if 'usage' in last_entry:
                    raw_usage = last_entry['usage']
                    # Handle Pydantic models / OpenAI objects
                    if hasattr(raw_usage, 'model_dump'):
                        token_usage = raw_usage.model_dump()
                    elif hasattr(raw_usage, 'dict'):
                        token_usage = raw_usage.dict()
                    elif isinstance(raw_usage, dict):
                        token_usage = raw_usage
                    else:
                        token_usage = {"raw": str(raw_usage)}
                        
                    # Recursive cleanup for any nested non-serializable objects
                    def clean_dict(d):
                        new_d = {}
                        for k, v in d.items():
                            if isinstance(v, dict):
                                new_d[k] = clean_dict(v)
                            elif hasattr(v, 'model_dump'):
                                new_d[k] = clean_dict(v.model_dump())
                            elif hasattr(v, 'dict'):
                                new_d[k] = clean_dict(v.dict())
                            elif isinstance(v, (str, int, float, bool, type(None))):
                                new_d[k] = v
                            else:
                                new_d[k] = str(v)
                        return new_d
                    
                    if isinstance(token_usage, dict):
                        token_usage = clean_dict(token_usage)

            except Exception as e:
                self.logger.debug(f"Error extracting history: {e}")
        
        # Log to main log file
        self.logger.info(f"""
--------------------------------------------------------------------------------
LLM RESPONSE - {self.report_id} ({latency_ms:.0f}ms)
--------------------------------------------------------------------------------
Latency:     {latency_ms:.0f}ms
Token Usage: {json.dumps(token_usage, default=str) if token_usage else 'N/A'}

OUTPUT FIELDS:
{self._format_dict(output_fields)}

{'─'*50}
FORMATTED PROMPT (DSPy):
{'─'*50}
{formatted_prompt[:5000] if formatted_prompt else '[No formatted prompt captured]'}

{'─'*50}
RAW API RESPONSE:
{'─'*50}
{raw_response[:2000] if raw_response else 'N/A'}
{'='*80}
""")
        
        # Reset
        self._request_start_time = None
        self._current_request = {}

    def _format_dict(self, d: Dict, indent: int = 2) -> str:
        """Format dictionary for readable logging."""
        lines = []
        for key, value in d.items():
            if isinstance(value, str) and len(value) > 200:
                # Truncate long strings for logs
                display_value = f"{value[:200]}... [{len(value)} chars]"
            else:
                display_value = value
            lines.append(f"{'  '*indent}{key}: {display_value}")
        return "\n".join(lines) if lines else "  (empty)"


def get_dspy_history(lm=None) -> List[Dict]:
    """
    Get the current DSPy LM history.
    
    Args:
        lm: Optional dspy.LM instance. If None, uses global dspy.settings.lm
        
    Returns:
        List of history entries
    """
    try:
        if lm is None:
            import dspy
            lm = dspy.settings.lm
            
        if lm and hasattr(lm, 'history') and lm.history:
            return list(lm.history)
    except Exception as e:
        logger.debug(f"Failed to get DSPy history: {e}")
    return []


def clear_dspy_history(lm=None) -> None:
    """Clear DSPy LM history to prevent accumulation."""
    try:
        if lm is None:
            import dspy
            lm = dspy.settings.lm
            
        if lm and hasattr(lm, 'history'):
            lm.history.clear()
    except Exception as e:
        logger.debug(f"Failed to clear DSPy history: {e}")
