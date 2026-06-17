from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:
    TestClient = None

if TestClient is not None:
    from memory_nav.web import MemoryGuidanceConfig, PanoExportConfig, WebConfig, create_app


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are not installed")
class WebServerTests(unittest.TestCase):
    def make_client(self, tmpdir: str):
        pano_dir = Path(tmpdir) / "pano"
        pano_dir.mkdir()
        (pano_dir / "index.html").write_text("<h1>Pano</h1>", encoding="utf-8")
        (pano_dir / "viewer_data.json").write_text("{}", encoding="utf-8")
        (pano_dir / "trajectory.js").write_text("window.PanoTrajectory = {};", encoding="utf-8")
        config = WebConfig(
            pano_export="never",
            memory_guidance=MemoryGuidanceConfig(upload_dir=str(Path(tmpdir) / "uploads")),
            pano_viewer=PanoExportConfig(output_dir=str(pano_dir)),
        )
        return TestClient(create_app(config))

    def test_healthz_and_tool_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(tmpdir)

            health = client.get("/healthz")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ok")

            memory_page = client.get("/memory-guidance/")
            self.assertEqual(memory_page.status_code, 200)
            self.assertIn("RAG", memory_page.text)

            pano_data = client.get("/pano-viewer/viewer_data.json")
            self.assertEqual(pano_data.status_code, 200)
            self.assertEqual(pano_data.json(), {})

            trajectory_script = client.get("/pano-viewer/trajectory.js")
            self.assertEqual(trajectory_script.status_code, 200)
            self.assertIn("PanoTrajectory", trajectory_script.text)

    def test_memory_guidance_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(tmpdir)
            response = client.post("/memory-guidance/api/guide", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action_request"], "missing_target")

    def test_copy_pano_viewer_includes_trajectory_parser(self) -> None:
        from memory_nav.web.pano_export import copy_viewer

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "viewer"
            output_dir.mkdir()
            copy_viewer(output_dir)

            self.assertTrue((output_dir / "trajectory.js").is_file())
            index_html = (output_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn('<script src="./trajectory.js"></script>', index_html)


if __name__ == "__main__":
    unittest.main()
