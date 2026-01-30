"""
AeroGuardian Comprehensive Logging System
==========================================
Author: AeroGuardian Member
Date: 2026-01-19
Version: 1.0

Centralized logging with:
- Single daily log file (logs/YYYY-MM-DD.log)
- Detailed LLM input/output logging
- DSPy prompt optimization tracking
- Error tracking with full context
- Easy debugging with structured logs
"""

import logging
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import wraps
from dataclasses import dataclass

# =============================================================================
# Project Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Get current date for log file
def _get_log_file_path() -> Path:
    """Get the daily log file path."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return LOGS_DIR / f"{date_str}.log"


# =============================================================================
# Log Formatters
# =============================================================================

class ColorFormatter(logging.Formatter):
    """Colored console output formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'LLM': '\033[34m',       # Blue (custom)
        'DSPY': '\033[95m',      # Light Magenta (custom)
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        # Handle custom level names
        level_name = record.levelname
        if hasattr(record, 'custom_level'):
            level_name = record.custom_level
        
        color = self.COLORS.get(level_name, self.COLORS.get(record.levelname, self.RESET))
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class DetailedFormatter(logging.Formatter):
    """Detailed file formatter with context tracking."""
    
    def format(self, record):
        # Add separator for important entries
        if hasattr(record, 'section_start') and record.section_start:
            record.msg = f"\n{'='*100}\n{record.msg}\n{'='*100}"
        elif hasattr(record, 'subsection') and record.subsection:
            record.msg = f"\n{'─'*80}\n{record.msg}\n{'─'*80}"
        
        return super().format(record)


# =============================================================================
# Singleton Logger Manager
# =============================================================================

class AeroGuardianLogger:
    """
    Centralized logger manager for the entire AeroGuardian project.
    Ensures all components log to a single daily file.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if AeroGuardianLogger._initialized:
            return
        
        AeroGuardianLogger._initialized = True
        
        self.log_file = _get_log_file_path()
        self._loggers: Dict[str, logging.Logger] = {}
        self._file_handler = None
        self._console_handler = None
        
        self._setup_handlers()
        self._write_session_header()
    
    def _setup_handlers(self):
        """Set up shared handlers for all loggers."""
        # File handler - detailed format
        self._file_handler = logging.FileHandler(
            self.log_file, 
            encoding='utf-8',
            mode='a'
        )
        self._file_handler.setLevel(logging.DEBUG)
        self._file_handler.setFormatter(DetailedFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-25s | L%(lineno)-4d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        
        # Console handler - concise with colors
        self._console_handler = logging.StreamHandler(sys.stdout)
        self._console_handler.setLevel(logging.INFO)
        self._console_handler.setFormatter(ColorFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%H:%M:%S"
        ))
    
    def _write_session_header(self):
        """Write session start header to log file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n{'#'*100}\n")
            f.write(f"#  AEROGUARDIAN SESSION START\n")
            f.write(f"#  Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"#  Log File: {self.log_file}\n")
            f.write(f"{'#'*100}\n\n")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the standard configuration."""
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        # Remove any existing handlers
        logger.handlers.clear()
        
        # Add our shared handlers
        logger.addHandler(self._file_handler)
        logger.addHandler(self._console_handler)
        
        self._loggers[name] = logger
        return logger


# =============================================================================
# LLM Logger with Detailed I/O Tracking
# =============================================================================

@dataclass
class LLMCallRecord:
    """Record of a single LLM call."""
    timestamp: str
    call_id: int
    model: str
    signature_name: str
    input_fields: Dict[str, Any]
    output_fields: Dict[str, Any]
    latency_ms: float
    tokens_input: int = 0
    tokens_output: int = 0
    success: bool = True
    error: str = ""


class LLMLogger:
    """
    Comprehensive LLM logger that tracks:
    - Full input prompts and parameters
    - Full output responses
    - Latency and token usage
    - Error details with context
    """
    
    _call_counter = 0
    
    def __init__(self, name: str = "AeroGuardian.LLM"):
        self.logger = get_logger(name)
        self.records: List[LLMCallRecord] = []
    
    def _get_call_id(self) -> int:
        """Get unique call ID."""
        LLMLogger._call_counter += 1
        return LLMLogger._call_counter
    
    def log_request(
        self, 
        model: str, 
        prompt: str, 
        signature_name: str = "Unknown",
        input_fields: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Log an LLM request with full details.
        
        Args:
            model: Model name (e.g., gpt-4o-mini)
            prompt: Full prompt text
            signature_name: DSPy signature class name
            input_fields: Structured input fields
        """
        call_id = self._get_call_id()
        timestamp = datetime.now().isoformat()
        
        # Console log (concise)
        self.logger.info(f"📤 LLM Request #{call_id} | {model} | {signature_name}")
        
        # File log (detailed)
        self._write_to_log(f"""
{'='*100}
[LLM REQUEST #{call_id}]
{'='*100}
Timestamp:      {timestamp}
Model:          {model}
Signature:      {signature_name}
Prompt Length:  {len(prompt)} characters

{'─'*50}
INPUT FIELDS:
{'─'*50}
{json.dumps(input_fields or {}, indent=2, ensure_ascii=False, default=str)}

{'─'*50}
FULL PROMPT:
{'─'*50}
{prompt}
{'='*100}
""")
        
        return call_id, time.time()
    
    def log_response(
        self,
        call_id: int,
        start_time: float,
        model: str,
        response: Any,
        signature_name: str = "Unknown",
        output_fields: Dict[str, Any] = None,
        tokens_used: int = 0,
    ):
        """
        Log an LLM response with full details.
        
        Args:
            call_id: Call ID from log_request
            start_time: Start timestamp from log_request
            model: Model name
            response: Full response object or text
            signature_name: DSPy signature class name
            output_fields: Structured output fields
            tokens_used: Token count if available
        """
        latency_ms = (time.time() - start_time) * 1000
        timestamp = datetime.now().isoformat()
        
        # Format response
        if hasattr(response, '__dict__'):
            response_str = json.dumps(
                {k: str(v)[:500] for k, v in response.__dict__.items() if not k.startswith('_')},
                indent=2, ensure_ascii=False
            )
        else:
            response_str = str(response)[:2000]
        
        # Console log (concise)
        self.logger.info(f"📥 LLM Response #{call_id} | {latency_ms:.0f}ms | {len(response_str)} chars")
        
        # File log (detailed)
        self._write_to_log(f"""
{'─'*100}
[LLM RESPONSE #{call_id}]
{'─'*100}
Timestamp:      {timestamp}
Model:          {model}
Signature:      {signature_name}
Latency:        {latency_ms:.2f} ms
Tokens:         {tokens_used if tokens_used else 'N/A'}

{'─'*50}
OUTPUT FIELDS:
{'─'*50}
{json.dumps(output_fields or {}, indent=2, ensure_ascii=False, default=str)}

{'─'*50}
FULL RESPONSE:
{'─'*50}
{response_str}
{'='*100}
""")
        
        # Store record
        self.records.append(LLMCallRecord(
            timestamp=timestamp,
            call_id=call_id,
            model=model,
            signature_name=signature_name,
            input_fields=output_fields or {},
            output_fields=output_fields or {},
            latency_ms=latency_ms,
            tokens_output=tokens_used,
        ))
    
    def log_error(
        self,
        call_id: int,
        model: str,
        error: Exception,
        prompt: str = None,
        context: str = "",
    ):
        """Log an LLM error with full details."""
        timestamp = datetime.now().isoformat()
        error_type = type(error).__name__
        
        # Console log
        self.logger.error(f"❌ LLM Error #{call_id} | {error_type}: {error}")
        
        # File log (detailed)
        self._write_to_log(f"""
{'!'*100}
[LLM ERROR #{call_id}]
{'!'*100}
Timestamp:      {timestamp}
Model:          {model}
Error Type:     {error_type}
Error Message:  {error}
Context:        {context}

{'─'*50}
TRACEBACK:
{'─'*50}
{traceback.format_exc()}

{'─'*50}
FAILED PROMPT:
{'─'*50}
{prompt[:2000] if prompt else 'N/A'}
{'!'*100}
""")
    
    def _write_to_log(self, content: str):
        """Write content to the daily log file."""
        with open(_get_log_file_path(), 'a', encoding='utf-8') as f:
            f.write(content)


# =============================================================================
# DSPy Logger for Prompt Optimization Tracking
# =============================================================================

class DSPyLogger:
    """
    Specialized logger for DSPy operations including:
    - Signature definitions and fields
    - Compiled prompt templates
    - Optimization metrics
    - Module execution details
    """
    
    def __init__(self, name: str = "AeroGuardian.DSPy"):
        self.logger = get_logger(name)
    
    def log_signature(self, signature_class, description: str = ""):
        """
        Log a DSPy signature definition with all its fields.
        
        Args:
            signature_class: DSPy Signature class
            description: Optional description
        """
        sig_name = signature_class.__name__
        
        # Extract input and output fields
        input_fields = []
        output_fields = []
        
        if hasattr(signature_class, '__annotations__'):
            for field_name, field_type in signature_class.__annotations__.items():
                field_info = {
                    'name': field_name,
                    'type': str(field_type),
                }
                
                field_obj = None  # Initialize to avoid unbound error
                
                # Check if it's an InputField or OutputField
                if hasattr(signature_class, field_name):
                    field_obj = getattr(signature_class, field_name)
                    if hasattr(field_obj, 'desc'):
                        field_info['desc'] = field_obj.desc
                
                # Determine if input or output based on field object type
                if field_obj is not None and hasattr(field_obj, '__class__'):
                    class_name = field_obj.__class__.__name__
                    if 'Input' in class_name:
                        input_fields.append(field_info)
                    elif 'Output' in class_name:
                        output_fields.append(field_info)
        
        # Console log
        self.logger.info(f"🔧 DSPy Signature: {sig_name} | {len(input_fields)} inputs, {len(output_fields)} outputs")
        
        # File log (detailed)
        docstring = signature_class.__doc__ or "No docstring"
        
        self._write_to_log(f"""
{'='*100}
[DSPY SIGNATURE] {sig_name}
{'='*100}
Description: {description}

{'─'*50}
DOCSTRING (System Prompt):
{'─'*50}
{docstring}

{'─'*50}
INPUT FIELDS ({len(input_fields)}):
{'─'*50}
{json.dumps(input_fields, indent=2)}

{'─'*50}
OUTPUT FIELDS ({len(output_fields)}):
{'─'*50}
{json.dumps(output_fields, indent=2)}
{'='*100}
""")
    
    def log_predict_call(
        self,
        signature_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        latency_ms: float,
    ):
        """
        Log a DSPy Predict call with inputs and outputs.
        
        Args:
            signature_name: Name of the signature
            inputs: Input field values
            outputs: Output field values
            latency_ms: Execution time
        """
        timestamp = datetime.now().isoformat()
        
        # Console log
        self.logger.info(f"⚡ DSPy Predict: {signature_name} | {latency_ms:.0f}ms")
        
        # File log
        self._write_to_log(f"""
{'─'*80}
[DSPY PREDICT] {signature_name}
{'─'*80}
Timestamp: {timestamp}
Latency:   {latency_ms:.2f} ms

INPUTS:
{json.dumps(inputs, indent=2, ensure_ascii=False, default=str)[:3000]}

OUTPUTS:
{json.dumps(outputs, indent=2, ensure_ascii=False, default=str)[:3000]}
{'─'*80}
""")
    
    def log_compiled_prompt(
        self,
        signature_name: str,
        compiled_prompt: str,
        context: str = "",
    ):
        """
        Log the actual compiled prompt that DSPy sends to the LLM.
        
        This is crucial for understanding what DSPy optimizes.
        
        Args:
            signature_name: Name of the signature
            compiled_prompt: The actual prompt text
            context: Additional context
        """
        timestamp = datetime.now().isoformat()
        
        # Console log
        self.logger.info(f"📝 DSPy Compiled Prompt: {signature_name} | {len(compiled_prompt)} chars")
        
        # File log
        self._write_to_log(f"""
{'='*100}
[DSPY COMPILED PROMPT] {signature_name}
{'='*100}
Timestamp: {timestamp}
Context:   {context}
Length:    {len(compiled_prompt)} characters

{'─'*50}
COMPILED PROMPT:
{'─'*50}
{compiled_prompt}
{'='*100}
""")
    
    def log_optimization_metrics(
        self,
        signature_name: str,
        metrics: Dict[str, Any],
        optimizer_name: str = "Unknown",
    ):
        """
        Log DSPy optimization metrics.
        
        Args:
            signature_name: Name of the signature
            metrics: Optimization metrics dict
            optimizer_name: Name of the optimizer used
        """
        timestamp = datetime.now().isoformat()
        
        # Console log
        self.logger.info(f"📊 DSPy Optimization: {signature_name} | {optimizer_name}")
        
        # File log
        self._write_to_log(f"""
{'='*100}
[DSPY OPTIMIZATION METRICS] {signature_name}
{'='*100}
Timestamp:  {timestamp}
Optimizer:  {optimizer_name}

METRICS:
{json.dumps(metrics, indent=2, ensure_ascii=False, default=str)}
{'='*100}
""")
    
    def _write_to_log(self, content: str):
        """Write content to the daily log file."""
        with open(_get_log_file_path(), 'a', encoding='utf-8') as f:
            f.write(content)


# =============================================================================
# Error Logger with Full Context
# =============================================================================

def log_exception(
    logger: logging.Logger,
    error: Exception,
    context: str = "",
    include_locals: bool = False,
):
    """
    Log an exception with full traceback and context.
    
    Args:
        logger: Logger instance
        error: The exception
        context: Description of what was happening
        include_locals: Whether to include local variables
    """
    error_type = type(error).__name__
    timestamp = datetime.now().isoformat()
    
    # Console log
    logger.error(f"❌ Exception [{error_type}]: {error}")
    if context:
        logger.error(f"   Context: {context}")
    
    # Detailed file log
    tb = traceback.format_exc()
    
    content = f"""
{'!'*100}
[EXCEPTION] {error_type}
{'!'*100}
Timestamp:  {timestamp}
Error:      {error}
Context:    {context}

{'─'*50}
TRACEBACK:
{'─'*50}
{tb}
{'!'*100}
"""
    
    with open(_get_log_file_path(), 'a', encoding='utf-8') as f:
        f.write(content)


# =============================================================================
# Decorator for Function Logging
# =============================================================================

def log_function_call(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Decorator to log function entry, exit, and errors.
    
    Args:
        logger: Logger instance (auto-created if None)
        level: Logging level for entry/exit
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Log entry
            log.log(level, f"→ Entering {func_name}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start_time) * 1000
                log.log(level, f"← Exiting {func_name} ({elapsed:.1f}ms)")
                return result
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                log_exception(log, e, context=f"Error in {func_name} after {elapsed:.1f}ms")
                raise
        
        return wrapper
    return decorator


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

_logger_manager: Optional[AeroGuardianLogger] = None


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger that writes to the daily log file.
    
    Args:
        name: Logger name (e.g., 'AeroGuardian.LLM')
    
    Returns:
        Configured logger instance
    """
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = AeroGuardianLogger()
    return _logger_manager.get_logger(name)


def get_llm_logger(name: str = "AeroGuardian.LLM") -> LLMLogger:
    """Get an LLM logger instance with detailed I/O tracking."""
    return LLMLogger(name)


def get_dspy_logger(name: str = "AeroGuardian.DSPy") -> DSPyLogger:
    """Get a DSPy logger instance for prompt optimization tracking."""
    return DSPyLogger(name)


def setup_logging(name: str = "AeroGuardian", level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging (convenience function for compatibility).
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Configured logger
    """
    return get_logger(name)


# =============================================================================
# Initialize on Import
# =============================================================================

# Create the logger manager on first import
_logger_manager = AeroGuardianLogger()

# Default logger
default_logger = get_logger("AeroGuardian")
