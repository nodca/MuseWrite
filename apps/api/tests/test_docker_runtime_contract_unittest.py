from pathlib import Path
import unittest


class DockerRuntimeContractTestCase(unittest.TestCase):
    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def test_local_dockerfile_uses_uv(self) -> None:
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY --from=ghcr.io/astral-sh/uv:latest", dockerfile)
        self.assertIn("uv pip install", dockerfile)
        self.assertIn("COPY requirements-docker.txt", dockerfile)
        # Should NOT contain ONNX reranker references.
        self.assertNotIn("PRELOAD_RERANKER_ONNX", dockerfile)
        self.assertNotIn("export_reranker_onnx", dockerfile)
        self.assertNotIn("CONTEXT_COMPRESSION_RERANKER_ONNX", dockerfile)

    def test_runtime_requirements_include_core_deps(self) -> None:
        runtime_requirements = Path("requirements-docker.txt").read_text(encoding="utf-8")
        self.assertIn("fastapi>=", runtime_requirements)
        self.assertIn("sqlmodel>=", runtime_requirements)
        self.assertIn("neo4j>=", runtime_requirements)
        self.assertIn("graphiti-core>=", runtime_requirements)
        self.assertIn("instructor>=", runtime_requirements)
        # ONNX deps should be gone.
        self.assertNotIn("onnxruntime", runtime_requirements)
        self.assertNotIn("transformers", runtime_requirements)
        self.assertNotIn("sentencepiece", runtime_requirements)

    def test_delivery_dockerfile_uses_uv(self) -> None:
        delivery_dockerfile = Path("Dockerfile.delivery").read_text(encoding="utf-8")
        self.assertIn("COPY --from=ghcr.io/astral-sh/uv:latest", delivery_dockerfile)
        self.assertIn("uv pip install", delivery_dockerfile)
        # Should NOT contain ONNX references.
        self.assertNotIn("PRELOAD_RERANKER_ONNX", delivery_dockerfile)

    def test_docker_compose_neo4j_gds(self) -> None:
        delivery_compose = (self._repo_root() / "docker-compose.delivery.yml").read_text(encoding="utf-8")
        default_compose = (self._repo_root() / "docker-compose.yml").read_text(encoding="utf-8")
        root_dockerignore = (self._repo_root() / ".dockerignore").read_text(encoding="utf-8")

        self.assertIn(".git/", root_dockerignore)
        self.assertIn("**/node_modules/", root_dockerignore)
        for compose_content in (default_compose, delivery_compose):
            self.assertIn("context: ./docker/neo4j-gds", compose_content)
            self.assertIn("image: novel-platform-neo4j-gds:latest", compose_content)
            self.assertIn("NEO4J_dbms_security_procedures_unrestricted: gds.*", compose_content)
            self.assertIn("RETURN gds.version() AS version", compose_content)

    def test_docker_compose_has_llm_proxy(self) -> None:
        compose = (self._repo_root() / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("llm-proxy:", compose)
        self.assertIn("context: ./docker/llm-proxy", compose)

    def test_env_example_exists_and_has_tiers(self) -> None:
        env_example = (self._repo_root() / ".env.example").read_text(encoding="utf-8")
        self.assertIn("Tier 1", env_example)
        self.assertIn("LLM_BASE_URL", env_example)
        self.assertIn("LLM_API_KEY", env_example)
        self.assertIn("LLM_MODEL", env_example)


if __name__ == "__main__":
    unittest.main()
