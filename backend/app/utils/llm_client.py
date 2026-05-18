import re
import json
from typing import List, Dict, Any, Optional
from .retry import retry_with_backoff


class LLMClient:
    """LLM API Client with chat interface"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: int = 120
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> str:
        """Send chat request and return response text"""
        import requests
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            url, 
            headers=headers, 
            json=data, 
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def extract_tags(self, content: str) -> List[str]:
        """
        Extract thinking tags from content
        
        Args:
            content: Text content to extract from
            
        Returns:
            List of thinking tags
        """
        # Remove think tags
        content = re.sub(r'<think>[\s\S]*?</think>', '', content)
        return content.strip()

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send chat request and return JSON

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Maximum token count

        Returns:
            Parsed JSON object
        """
        # Build prompt to ensure JSON response
        json_prompt_added = False
        for msg in messages:
            if msg["role"] == "user" and not json_prompt_added:
                msg["content"] = msg["content"] + "\n\nPlease only return JSON format, no other content."
                json_prompt_added = True

        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Clean markdown code block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format from LLM: {cleaned_response[:200]}...")
