"""Routes for LLM chat functionality."""

import boto3
from fastapi import APIRouter, HTTPException, status
from botocore.config import Config

from ..config import settings
from ..models.llm import ChatRequest, ChatResponse
from ..services.llm_service import NovaLLMService

router = APIRouter(prefix="/llm", tags=["LLM Chat"])


def get_bedrock_client():
    """Get configured Bedrock client for Nova Premier."""
    bedrock_config = Config(
        read_timeout=1000,
        connect_timeout=5,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    return boto3.client(
        "bedrock-runtime",  # Use bedrock-runtime for model invocation
        region_name=settings.AWS_REGION,
        config=bedrock_config,
    )


def get_llm_service(system_prompt: str | None = None) -> NovaLLMService:
    """Create and configure NovaLLMService instance."""
    client = get_bedrock_client()
    
    # Default system prompt if none provided
    default_system_prompt = """You are a helpful AI assistant specialized in contract analysis and document processing. 
    You provide clear, accurate, and professional responses. When analyzing contracts or legal documents, 
    focus on key terms, obligations, and important clauses."""
    
    return NovaLLMService(
        model_id=settings.NOVA_MODEL_ID or "us.amazon.nova-premier-v1:0",  # Use the configured Nova Premier model
        client=client,
        system_prompt=system_prompt or default_system_prompt,
        max_tokens=4000,
        temperature=0.7,
        top_p=0.9,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with the LLM using Nova Premier.
    
    This endpoint allows users to send messages to the LLM and receive responses.
    Supports both streaming and standard responses.
    """
    try:
        # Create LLM service instance
        llm_service = get_llm_service(request.system_prompt)
        
        # Get response from LLM
        response_text = llm_service.invoke_model(
            input_str=request.message,
            streaming=request.streaming
        )
        
        return ChatResponse(
            response=response_text,
            model_id=llm_service.model_id,
            cache_metrics=None  # Will be extracted from response in the future
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for LLM service."""
    try:
        # Simple test to verify Bedrock connectivity
        client = get_bedrock_client()
        # You could add a simple model list call here to verify connectivity
        return {"status": "healthy", "service": "llm"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service unavailable: {str(e)}"
        )
