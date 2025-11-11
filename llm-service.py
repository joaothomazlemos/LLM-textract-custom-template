import json
import re
from typing import Any, Dict

import boto3


import logging

logger = logging.getLogger(__name__)


class AnthropicLLMService:
    def __init__(
        self,
        model_id: str,
        model_version: str,
        client: Any,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        budget_tokens: int,
        reasoning: bool = True,
    ) -> None:
        self.model_id = model_id
        self.model_version = model_version
        self.client = client
        self.system_prompt = system_prompt
        self.MAX_TOKENS = max_tokens
        self.TEMPERATURE = temperature
        self.BUDGET_TOKENS = budget_tokens
        self.REASONING = reasoning

    def _config_body(self, input_str: str) -> Dict[str, Any]:
        body = {
            "anthropic_version": self.model_version,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
            "system": self.system_prompt,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": input_str}]},
            ],
        }
        if self.REASONING:
            body["thinking"] = {"type": "enabled", "budget_tokens": self.BUDGET_TOKENS}
        return body

    def _safe_extract_json(self, raw_text: str) -> dict:
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL
        ) or re.search(r"\{.*?\}", raw_text, re.DOTALL)

        if not match:
            raise ValueError("Nenhum JSON encontrado na resposta do modelo.")

        json_str = match.group(1) if match.lastindex else match.group(0)
        json_str = json_str.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON inválido detectado: {e}")
            cleaned = re.sub(r"^[^\{]*", "", json_str)
            cleaned = re.sub(r"[^\}]*$", "", cleaned)
            return json.loads(cleaned)

    def invoke_model(self, input_str: str) -> dict:
        response = self.client.invoke_model(
            body=json.dumps(self._config_body(input_str)), modelId=self.model_id
        )
        body_str = response["body"].read().decode("utf-8")
        if not body_str.strip():
            raise ValueError("Empty body from Bedrock model")

        response_body = json.loads(body_str)

        text_parts = [
            block.get("text", "")
            for block in response_body.get("content", [])
            if block.get("type") == "text"
        ]
        raw_text_response = "\n".join(text_parts)
        parsed_json = self._safe_extract_json(raw_text_response)

        return parsed_json
