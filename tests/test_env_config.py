from __future__ import annotations

import os
import unittest

from memory_nav.common.env import resolve_model_environment, resolve_task_num_ctx


class EnvConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("NAV_", "ST_NAV_"))
            or key in {"OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GMAPS_KEY", "GMAPS_API_KEY"}
        }
        for key in list(self._saved_env):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in list(os.environ):
            if key.startswith(("NAV_", "ST_NAV_")) or key in {
                "OPENAI_API_KEY",
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
                "GMAPS_KEY",
                "GMAPS_API_KEY",
            }:
                os.environ.pop(key, None)
        os.environ.update(self._saved_env)

    def test_short_profile_env_names_resolve_model_settings(self) -> None:
        os.environ.update(
            {
                "NAV_PROFILE": "ollama",
                "NAV_OLLAMA_MODEL": "gemma4:31b",
                "NAV_OLLAMA_BASE": "http://127.0.0.1:11434/v1",
                "NAV_OLLAMA_KEY": "ollama",
                "NAV_OLLAMA_API": "chat_completions",
                "NAV_OLLAMA_CTX": "16384",
                "NAV_OLLAMA_TEMP": "0",
                "NAV_OLLAMA_TIMEOUT": "180",
            }
        )

        settings = resolve_model_environment(default_api_kind="responses")

        self.assertEqual(settings.active_profile, "ollama")
        self.assertEqual(settings.provider, "ollama")
        self.assertEqual(settings.model_name, "gemma4:31b")
        self.assertEqual(settings.api_base, "http://127.0.0.1:11434/v1")
        self.assertEqual(settings.api_key, "ollama")
        self.assertEqual(settings.api_kind, "chat_completions")
        self.assertEqual(settings.num_ctx, 16384)
        self.assertEqual(settings.temperature, 0.0)
        self.assertEqual(settings.request_timeout, 180.0)

    def test_legacy_profile_env_names_still_resolve(self) -> None:
        os.environ.update(
            {
                "ST_NAV_ACTIVE_PROFILE": "gemini",
                "ST_NAV_PROFILE_GEMINI_MODEL_PROVIDER": "gemini",
                "ST_NAV_PROFILE_GEMINI_MODEL_NAME": "gemma-4-31b-it",
                "ST_NAV_PROFILE_GEMINI_API_KEY": "secret",
                "ST_NAV_PROFILE_GEMINI_API_KIND": "responses",
                "ST_NAV_PROFILE_GEMINI_REQUEST_TIMEOUT": "300",
            }
        )

        settings = resolve_model_environment(default_api_kind="chat_completions")

        self.assertEqual(settings.provider, "gemini")
        self.assertEqual(settings.model_name, "gemma-4-31b-it")
        self.assertEqual(settings.api_key, "secret")
        self.assertEqual(settings.api_kind, "responses")
        self.assertEqual(settings.request_timeout, 300.0)


    def test_gemini_profile_prefers_gemini_standard_key(self) -> None:
        os.environ.update(
            {
                "NAV_PROFILE": "gemini",
                "NAV_GEMINI_MODEL": "gemma-4-31b-it",
                "OPENAI_API_KEY": "openai-secret",
                "GEMINI_API_KEY": "gemini-secret",
            }
        )

        settings = resolve_model_environment()

        self.assertEqual(settings.provider, "gemini")
        self.assertEqual(settings.api_key, "gemini-secret")

    def test_short_task_context_aliases_resolve(self) -> None:
        os.environ.update(
            {
                "NAV_PARSE_CTX": "8192",
                "NAV_LOCALIZE_CTX": "16384",
            }
        )

        self.assertEqual(resolve_task_num_ctx("parse_instruction"), 8192)
        self.assertEqual(resolve_task_num_ctx("localization"), 16384)


if __name__ == "__main__":
    unittest.main()
