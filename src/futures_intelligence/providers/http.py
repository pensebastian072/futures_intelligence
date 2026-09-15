from __future__ import annotations

from typing import Any

import requests


class SafeHttpClient:
    def __init__(self, *, timeout: float = 45.0, session: requests.Session | None = None):
        self.timeout = timeout
        self.session = session or requests.Session()

    def get(self, url: str, *, headers: dict[str, str] | None = None,
            params: dict[str, Any] | None = None, auth=None) -> requests.Response:
        response = self.session.get(
            url, headers=headers, params=params, auth=auth, timeout=self.timeout
        )
        response.raise_for_status()
        return response

    def post(self, url: str, *, headers: dict[str, str] | None = None,
             data: dict[str, Any] | None = None, auth=None) -> requests.Response:
        response = self.session.post(
            url, headers=headers, data=data, auth=auth, timeout=self.timeout
        )
        response.raise_for_status()
        return response

