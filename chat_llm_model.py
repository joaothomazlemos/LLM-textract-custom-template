"""Pydantic models for LLM chat functionality."""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    
    message: str = Field(
        ..., 
        description="The user's message to send to the LLM",
        min_length=1,
        max_length=10000
    )
    streaming: bool = Field(
        default=False,
        description="Whether to use streaming response"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for this conversation"
    )


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    
    response: str = Field(
        ...,
        description="The LLM's response in markdown format"
    )
    cache_metrics: Optional[dict] = Field(
        default=None,
        description="Cache usage metrics if available"
    )
    model_id: str = Field(
        ...,
        description="The model ID used for the response"
    )


class ChatMetrics(BaseModel):
    """Model for cache and token usage metrics."""
    
    cache_read_tokens: int = Field(default=0, description="Tokens read from cache")
    cache_write_tokens: int = Field(default=0, description="Tokens written to cache") 
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens generated")
