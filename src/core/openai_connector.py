"""
OpenAI Connector - Standard API Interface
==========================================
Author: AeroGuardian Member
Date: 2026-01-18

A robust, debuggable OpenAI connector with:
- Proper error handling and retry logic
- Detailed logging for debugging
- Environment-based configuration
- Rate limiting protection
- Token usage tracking

Usage:
    from src.core.openai_connector import OpenAIConnector, get_openai
    
    client = get_openai()
    response = client.chat("What is UAV safety?")
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("AeroGuardian.OpenAI")

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OpenAIConfig:
    """OpenAI connector configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.1")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_TOKENS", "4096")))
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set in environment!")


@dataclass
class ChatResponse:
    """Structured response from OpenAI chat."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    latency_ms: float
    raw_response: Optional[Any] = None
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
        }


# =============================================================================
# OpenAI Connector Class
# =============================================================================

class OpenAIConnector:
    """
    Standard OpenAI API connector with robust error handling.
    
    Features:
    - Automatic retry with exponential backoff
    - Detailed logging for debugging
    - Token usage tracking
    - Rate limit handling
    
    Example:
        >>> connector = OpenAIConnector()
        >>> response = connector.chat("Hello!")
        >>> print(response.content)
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize OpenAI connector."""
        self.config = config or OpenAIConfig()
        self.client: Optional[OpenAI] = None
        self.is_ready = False
        
        # Usage tracking
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the OpenAI client."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI package not installed! Run: pip install openai")
            return
        
        if not self.config.api_key:
            logger.error("OPENAI_API_KEY not configured!")
            return
        
        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=0,  # We handle retries ourselves
            )
            self.is_ready = True
            logger.info(f"✓ OpenAI Connector initialized")
            logger.info(f"  Model: {self.config.model}")
            logger.info(f"  Temperature: {self.config.temperature}")
            logger.info(f"  Max Tokens: {self.config.max_tokens}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.is_ready = False
    
    def _log_request(self, messages: List[Dict], attempt: int = 1):
        """Log outgoing request details."""
        user_msg = messages[-1].get("content", "")[:100]
        logger.debug(f"[Request #{self.total_requests + 1}] Attempt {attempt}")
        logger.debug(f"  Model: {self.config.model}")
        logger.debug(f"  Messages: {len(messages)}")
        logger.debug(f"  Last message: {user_msg}...")
    
    def _log_response(self, response: ChatResponse):
        """Log response details."""
        logger.info(f"[Response] Model: {response.model}")
        logger.info(f"  Tokens: {response.usage}")
        logger.info(f"  Latency: {response.latency_ms:.0f}ms")
        logger.info(f"  Finish: {response.finish_reason}")
    
    def _log_error(self, error: Exception, attempt: int, max_attempts: int):
        """Log error details."""
        error_type = type(error).__name__
        logger.warning(f"[Error] Attempt {attempt}/{max_attempts}: {error_type}")
        logger.warning(f"  Message: {str(error)[:200]}")
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat message to OpenAI.
        
        Args:
            message: User message content
            system_prompt: Optional system prompt
            json_mode: If True, request JSON response format
            **kwargs: Additional parameters for the API
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            RuntimeError: If all retries fail
        """
        if not self.is_ready:
            raise RuntimeError("OpenAI connector not ready. Check API key.")
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        return self.chat_messages(messages, json_mode=json_mode, **kwargs)
    
    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        **kwargs
    ) -> ChatResponse:
        """
        Send multiple messages to OpenAI.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            json_mode: If True, request JSON response format
            **kwargs: Additional parameters for the API
            
        Returns:
            ChatResponse with content and metadata
        """
        if not self.is_ready:
            raise RuntimeError("OpenAI connector not ready. Check API key.")
        
        self.total_requests += 1
        last_error = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._log_request(messages, attempt)
                
                # Build request parameters
                # Note: GPT-5-mini uses max_completion_tokens instead of max_tokens
                # and only supports temperature=1 (default)
                params = {
                    "model": self.config.model,
                    "messages": messages,
                    "max_completion_tokens": self.config.max_tokens,
                    **kwargs
                }
                
                # Only add temperature for models that support it (not gpt-5-mini)
                if "gpt-5" not in self.config.model:
                    params["temperature"] = self.config.temperature
                
                if json_mode:
                    params["response_format"] = {"type": "json_object"}
                
                # Make request
                start_time = time.time()
                response = self.client.chat.completions.create(**params)
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract response
                choice = response.choices[0]
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                
                # Track usage
                self.total_tokens_used += usage["total_tokens"]
                
                result = ChatResponse(
                    content=choice.message.content,
                    model=response.model,
                    usage=usage,
                    finish_reason=choice.finish_reason,
                    latency_ms=latency_ms,
                    raw_response=response,
                )
                
                self._log_response(result)
                return result
                
            except Exception as e:
                last_error = e
                self._log_error(e, attempt, self.config.max_retries)
                
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** (attempt - 1))
                    logger.info(f"  Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        
        # All retries failed
        self.failed_requests += 1
        raise RuntimeError(f"OpenAI request failed after {self.config.max_retries} attempts: {last_error}")
    
    def chat_json(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Send a chat message and parse JSON response.
        
        Args:
            message: User message content
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON as dictionary
        """
        response = self.chat(message, system_prompt, json_mode=True, **kwargs)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw content: {response.content[:500]}")
            raise
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
            "total_tokens_used": self.total_tokens_used,
            "model": self.config.model,
        }
    
    def test_connection(self) -> bool:
        """Test the OpenAI connection with a simple request."""
        try:
            logger.info("Testing OpenAI connection...")
            response = self.chat("Say 'OK' if you can read this.", system_prompt="Respond with only 'OK'.")
            success = "ok" in response.content.lower()
            if success:
                logger.info("✓ Connection test passed!")
            else:
                logger.warning(f"Connection test unexpected response: {response.content}")
            return success
        except Exception as e:
            logger.error(f"✗ Connection test failed: {e}")
            return False


# =============================================================================
# Singleton Instance
# =============================================================================

_connector: Optional[OpenAIConnector] = None


def get_openai() -> OpenAIConnector:
    """Get singleton OpenAI connector instance."""
    global _connector
    if _connector is None:
        _connector = OpenAIConnector()
    return _connector


def reset_connector():
    """Reset the singleton connector (useful for testing)."""
    global _connector
    _connector = None


# =============================================================================
# Main - Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  OpenAI Connector Test")
    print("=" * 60)
    
    connector = get_openai()
    
    if connector.is_ready:
        # Test connection
        if connector.test_connection():
            print("\n✓ OpenAI connection working!")
            
            # Test a real query
            print("\nTesting with a UAV safety query...")
            response = connector.chat(
                "What are the top 3 causes of UAV crashes?",
                system_prompt="You are a UAV safety expert. Be concise."
            )
            print(f"\nResponse:\n{response.content}")
            print(f"\nStats: {connector.get_stats()}")
    else:
        print("✗ OpenAI connector not ready. Check your API key.")
    
    print("=" * 60)
