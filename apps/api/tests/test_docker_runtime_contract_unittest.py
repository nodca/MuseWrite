from pathlib import Path
import unittest


class DockerRuntimeContractTestCase(unittest.TestCase):
    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def test_local_dockerfile_uses_optional_export_path(self) -> None:
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
        dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

        self.assertIn("ARG PRELOAD_RERANKER_ONNX=false", dockerfile)
        self.assertIn("COPY requirements-docker.txt ./requirements.txt", dockerfile)
        self.assertIn("COPY requirements-export.txt ./requirements-export.txt", dockerfile)
        self.assertNotIn("COPY models ./models", dockerfile)
        self.assertIn("pip install --no-cache-dir -r requirements-export.txt", dockerfile)
        self.assertIn("python scripts/export_reranker_onnx.py", dockerfile)
        self.assertIn("models/", dockerignore)

    def test_runtime_requirements_exclude_export_time_dependencies(self) -> None:
        runtime_requirements = Path("requirements-docker.txt").read_text(encoding="utf-8")
        export_requirements = Path("requirements-export.txt").read_text(encoding="utf-8")

        self.assertIn("onnxruntime>=1.20.0", runtime_requirements)
        self.assertIn("transformers>=4.46.0", runtime_requirements)
        self.assertIn("sentencepiece>=0.2.0", runtime_requirements)
        self.assertNotIn("torch>=", runtime_requirements)
        self.assertNotIn("\nonnx>=", f"\n{runtime_requirements}")
        self.assertIn("torch>=2.2.0", export_requirements)
        self.assertIn("onnx>=1.20.0", export_requirements)

    def test_delivery_assets_exist_for_baked_model_release(self) -> None:
        delivery_dockerfile = Path("Dockerfile.delivery").read_text(encoding="utf-8")
        delivery_compose = (self._repo_root() / "docker-compose.delivery.yml").read_text(encoding="utf-8")
        default_compose = (self._repo_root() / "docker-compose.yml").read_text(encoding="utf-8")
        root_dockerignore = (self._repo_root() / ".dockerignore").read_text(encoding="utf-8")

        self.assertIn("COPY apps/api/models ./models", delivery_dockerfile)
        self.assertIn("COPY apps/api/requirements-docker.txt ./requirements.txt", delivery_dockerfile)
        self.assertNotIn("requirements-export.txt", delivery_dockerfile)
        self.assertIn("context: .", delivery_compose)
        self.assertIn("dockerfile: apps/api/Dockerfile.delivery", delivery_compose)
        self.assertIn("image: novel-platform-api:delivery", delivery_compose)
        self.assertNotIn("./apps/api/models:/app/models:ro", delivery_compose)
        self.assertIn(".git/", root_dockerignore)
        self.assertIn("**/node_modules/", root_dockerignore)
        for compose_content in (default_compose, delivery_compose):
            self.assertIn("context: ./docker/neo4j-gds", compose_content)
            self.assertIn("image: novel-platform-neo4j-gds:latest", compose_content)
            self.assertIn("NEO4J_dbms_security_procedures_unrestricted: gds.*", compose_content)
            self.assertIn("RETURN gds.version() AS version", compose_content)
            self.assertNotIn("NEO4J_PLUGINS:", compose_content)
            self.assertNotIn("neo4j_plugins:", compose_content)


if __name__ == "__main__":
    unittest.main()
