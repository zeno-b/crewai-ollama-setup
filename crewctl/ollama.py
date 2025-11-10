"""HTTP client for interacting with Ollama's REST API."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

import requests
from requests import Response


class OllamaClientError(RuntimeError):
    """Raised when Ollama API operations fail."""


class OllamaClient:
    def __init__(self, base_url: str, *, timeout: int = 30, verify: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify = verify
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _checked(self, response: Response) -> Response:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = self._extract_error(response)
            raise OllamaClientError(detail) from exc
        return response

    def _extract_error(self, response: Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict) and "error" in payload:
                return str(payload["error"])
        except ValueError:
            pass
        return f"HTTP {response.status_code} calling {response.url}"

    def list_models(self) -> List[Dict[str, Any]]:
        response = self._checked(
            self.session.get(
                self._url("/api/tags"),
                timeout=self.timeout,
                verify=self.verify,
            )
        )
        payload = response.json()
        if isinstance(payload, dict):
            models = payload.get("models")
            if isinstance(models, list):
                return models
        raise OllamaClientError("Unexpected response format from /api/tags")

    def show_model(self, name: str) -> Optional[Dict[str, Any]]:
        for model in self.list_models():
            if model.get("name") == name:
                return model
        return None

    def pull_model(self, name: str) -> Iterator[Dict[str, Any]]:
        payload = {"name": name, "stream": True}
        response = self._checked(
            self.session.post(
                self._url("/api/pull"),
                json=payload,
                timeout=self.timeout,
                verify=self.verify,
                stream=True,
            )
        )
        yield from self._iter_events(response)

    def create_model(self, name: str, modelfile: str, *, quantize: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "name": name,
            "modelfile": modelfile,
            "stream": True,
        }
        if quantize:
            payload["quantize"] = quantize
        response = self._checked(
            self.session.post(
                self._url("/api/create"),
                json=payload,
                timeout=self.timeout,
                verify=self.verify,
                stream=True,
            )
        )
        yield from self._iter_events(response)

    def show_info(self) -> Dict[str, Any]:
        response = self._checked(
            self.session.get(
                self._url("/api/version"),
                timeout=self.timeout,
                verify=self.verify,
            )
        )
        return response.json()

    def _iter_events(self, response: Response) -> Iterator[Dict[str, Any]]:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                event = {"status": raw_line}
            yield event
