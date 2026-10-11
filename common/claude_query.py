"""
Claude Code Query Module - JUPYTER COMPATIBLE

A module for querying Claude Code with support for single and batch requests.
Fixed to work properly in Jupyter notebooks by handling async event loops correctly.

Author: Mahavir
Date: 2026-03-26
"""

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    CLINotFoundError,
    ProcessError
)
import asyncio
from typing import List, Optional
from dataclasses import dataclass
import logging

# Configure logging - only show errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Suppress INFO logs from claude_agent_sdk
logging.getLogger('claude_agent_sdk').setLevel(logging.ERROR)


def _run_async_safe(coro):
    """
    Run async coroutine in a way that works both in Jupyter and regular Python.
    
    In Jupyter notebooks, there's already a running event loop, so asyncio.run()
    fails. This function detects that and handles it properly.
    """
    try:
        # Check if there's already a running loop (Jupyter case)
        loop = asyncio.get_running_loop()
        
        # We're in Jupyter - create task and run synchronously
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(coro)
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)
    except ImportError:
        # nest_asyncio not installed
        logger.error(
            "nest_asyncio is required for Jupyter notebooks. "
            "Install it with: pip install nest_asyncio"
        )
        raise


@dataclass
class QueryResponse:
    """Represents a query response."""
    prompt: str
    response: str
    error: Optional[str] = None
    success: bool = True


class ClaudeCodeClient:
    """
    Client for interacting with Claude Code.
    
    Works in both Jupyter notebooks and regular Python scripts.
    """
    
    def __init__(
        self,
        default_model: Optional[str] = None,
        default_system_prompt: Optional[str] = None,
        permission_mode: str = "bypassPermissions"
    ):
        """
        Initialize the Claude Code client.
        
        Args:
            default_model: Default model to use (e.g., "claude-sonnet-4-20250514")
            default_system_prompt: Default system prompt for all queries
            permission_mode: Permission mode ("bypassPermissions", "acceptEdits", "plan")
        """
        self.default_model = default_model
        self.default_system_prompt = default_system_prompt
        self.permission_mode = permission_mode
    
    async def _query_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Internal async query method.
        
        Args:
            prompt: The question or task to send to Claude
            system_prompt: Optional custom system prompt
            model: Optional model selection
            
        Returns:
            Claude's response as a string
        """
        _system_prompt = system_prompt or self.default_system_prompt
        _model = model or self.default_model
        
        options_dict = {"permission_mode": self.permission_mode}
        if _system_prompt:
            options_dict["system_prompt"] = _system_prompt
        if _model:
            options_dict["model"] = _model
            
        options = ClaudeAgentOptions(**options_dict)
        
        response_text = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
        
        return response_text
    
    def ask(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Synchronous query to Claude Code.
        Works in Jupyter notebooks and regular Python scripts.
        
        Args:
            prompt: The question or task to send to Claude
            system_prompt: Optional custom system prompt
            model: Optional model selection
            
        Returns:
            Claude's response as a string
        """
        try:
            return _run_async_safe(
                self._query_async(prompt, system_prompt, model)
            )
        except Exception as e:
            logger.error(f"Error in ask(): {e}")
            raise
    
    async def ask_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Asynchronous query to Claude Code.
        Use this with 'await' in Jupyter notebooks or async contexts.
        
        Args:
            prompt: The question or task to send to Claude
            system_prompt: Optional custom system prompt
            model: Optional model selection
            
        Returns:
            Claude's response as a string
        """
        return await self._query_async(prompt, system_prompt, model)
    
    async def batch_query_async(
        self,
        prompts: List[str],
        system_prompts: Optional[List[Optional[str]]] = None,
        models: Optional[List[Optional[str]]] = None,
        max_concurrent: int = 10
    ) -> List[QueryResponse]:
        """
        Execute multiple queries concurrently.
        
        Args:
            prompts: List of prompts to send
            system_prompts: Optional list of system prompts (same length as prompts)
            models: Optional list of models (same length as prompts)
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of QueryResponse objects in the same order as prompts
        """
        num_prompts = len(prompts)
        
        if system_prompts is None:
            system_prompts = [None] * num_prompts
        elif len(system_prompts) != num_prompts:
            raise ValueError(
                f"system_prompts length ({len(system_prompts)}) must match "
                f"prompts length ({num_prompts})"
            )
        
        if models is None:
            models = [None] * num_prompts
        elif len(models) != num_prompts:
            raise ValueError(
                f"models length ({len(models)}) must match "
                f"prompts length ({num_prompts})"
            )
        
        if max_concurrent > 10:
            logger.warning(f"max_concurrent={max_concurrent} exceeds recommended limit of 10")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _query_with_semaphore(
            prompt: str,
            system_prompt: Optional[str],
            model: Optional[str]
        ) -> QueryResponse:
            async with semaphore:
                try:
                    response = await self._query_async(prompt, system_prompt, model)
                    return QueryResponse(
                        prompt=prompt,
                        response=response,
                        success=True
                    )
                except CLINotFoundError:
                    error_msg = "Claude Code CLI not found. Please install it first."
                    logger.error(error_msg)
                    return QueryResponse(
                        prompt=prompt,
                        response="",
                        error=error_msg,
                        success=False
                    )
                except ProcessError as e:
                    error_msg = f"Process failed with exit code {e.exit_code}"
                    logger.error(f"Error for prompt '{prompt[:50]}...': {error_msg}")
                    return QueryResponse(
                        prompt=prompt,
                        response="",
                        error=error_msg,
                        success=False
                    )
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error for prompt '{prompt[:50]}...': {error_msg}")
                    return QueryResponse(
                        prompt=prompt,
                        response="",
                        error=error_msg,
                        success=False
                    )
        
        tasks = [
            _query_with_semaphore(prompt, sys_prompt, model)
            for prompt, sys_prompt, model in zip(prompts, system_prompts, models)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return results
    
    def batch_query(
        self,
        prompts: List[str],
        system_prompts: Optional[List[Optional[str]]] = None,
        models: Optional[List[Optional[str]]] = None,
        max_concurrent: int = 10
    ) -> List[QueryResponse]:
        """
        Synchronous wrapper for batch_query_async.
        Works in Jupyter notebooks and regular Python scripts.
        
        Args:
            prompts: List of prompts to send
            system_prompts: Optional list of system prompts
            models: Optional list of models
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of QueryResponse objects
        """
        return _run_async_safe(
            self.batch_query_async(prompts, system_prompts, models, max_concurrent)
        )


# Convenience functions

def ask_claude(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Simple function to query Claude Code.
    Works in Jupyter notebooks and regular Python scripts.
    
    Args:
        prompt: The question or task
        system_prompt: Optional system prompt
        model: Optional model selection
        
    Returns:
        Claude's response as a string
    """
    client = ClaudeCodeClient()
    return client.ask(prompt, system_prompt, model)


def batch_ask_claude(
    prompts: List[str],
    system_prompts: Optional[List[Optional[str]]] = None,
    models: Optional[List[Optional[str]]] = None,
    max_concurrent: int = 10
) -> List[QueryResponse]:
    """
    Simple function to query Claude Code with multiple prompts.
    Works in Jupyter notebooks and regular Python scripts.
    
    Args:
        prompts: List of prompts
        system_prompts: Optional list of system prompts
        models: Optional list of models
        max_concurrent: Maximum concurrent requests
        
    Returns:
        List of QueryResponse objects
    """
    client = ClaudeCodeClient()
    return client.batch_query(prompts, system_prompts, models, max_concurrent)


if __name__ == "__main__":
    print("=== Example 1: Simple Query ===")
    response = ask_claude("What is the capital of India?")
    print(f"Response: {response}\n")
    
    print("=== Example 2: Batch Queries ===")
    prompts = ["What is 25 * 47?", "What is the square root of 144?"]
    responses = batch_ask_claude(prompts)
    
    for resp in responses:
        if resp.success:
            print(f"Q: {resp.prompt}")
            print(f"A: {resp.response}\n")