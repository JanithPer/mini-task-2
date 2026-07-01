from __future__ import annotations

import httpx

GEMINI_CACHE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TTL_SECONDS = 7200


async def create_gemini_cache(
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> str | None:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{GEMINI_CACHE_BASE_URL}/cachedContents",
                params={"key": api_key},
                json={
                    "model": f"models/{model}",
                    "contents": [
                        {
                            "parts": [{"text": user_content}],
                            "role": "user",
                        }
                    ],
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}],
                    },
                    "ttl": f"{ttl_seconds}s",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["name"]
        except Exception:
            return None
