"""
LLM Client - 使用 Codemax API (Anthropic 兼容格式)
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Union

import httpx

from ..config import Config


class LLMClient:
    """
    LLM 客户端
    使用 Codemax API (Anthropic 兼容格式)
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM API key not configured")

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        发送请求到 LLM API

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数

        Returns:
            LLM 生成的文本
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Codemax API 不支持 system role，将其合并到用户消息中
        processed_messages = self._convert_system_messages(messages)

        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": processed_messages,
            "temperature": temperature,
            **kwargs
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=body
            )

        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")

        data = response.json()

        # 提取文本内容
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "")

        return ""

    def _convert_system_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        将 system 消息转换为 user 消息（兼容 Codemax API）

        Codemax API 不支持 'system' role，将系统提示合并到第一个用户消息中。
        """
        system_prompt = ""
        new_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n\n"
            else:
                new_messages.append(msg)

        if system_prompt and new_messages:
            # 将系统提示添加到第一个用户消息的开头
            first_msg = new_messages[0]
            first_msg["content"] = system_prompt + first_msg["content"]

        return new_messages

    def call(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: str = None
    ) -> str:
        """
        简单的 prompt 调用

        Args:
            prompt: 用户 prompt
            temperature: 温度参数
            max_tokens: 最大 token 数
            system: 系统提示

        Returns:
            LLM 生成的文本
        """
        messages = []
        if system:
            messages.append({"role": "user", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self._make_request(messages, temperature, max_tokens)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """
        对话接口

        Args:
            messages: 消息列表 [{"role": "user/assistant", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            LLM 生成的文本
        """
        return self._make_request(messages, temperature, max_tokens)

    def extract_content(self, content: str) -> str:
        """
        清理 LLM 返回的内容，去除思考标记等

        Args:
            content: 原始内容

        Returns:
            清理后的内容
        """
        # 移除 <thinking>...</thinking> 标签
        content = re.sub(r'<thinking>[\s\S]*?</thinking>', '', content)
        # 移除 <think>...</think> 标签
        content = re.sub(r'<think>[\s\S]*?</think>', '', content)
        return content.strip()

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回 JSON

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            解析后的 JSON 对象
        """
        # 构建提示词，确保返回 JSON
        json_prompt_added = False
        for msg in messages:
            if msg["role"] == "user" and not json_prompt_added:
                msg["content"] = msg["content"] + "\n\n请只返回 JSON 格式，不要包含其他内容。"
                json_prompt_added = True

        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 清理 markdown 代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response[:200]}...")
