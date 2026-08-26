"""Regression tests for executable development documentation."""

import unittest
from pathlib import Path


class DevelopmentDocumentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[1]
        cls.primary_documents = {
            name: (project_root / name).read_text()
            for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md")
        }
        cls.examples_readme = (project_root / "examples" / "README").read_text()

    def test_primary_documents_share_setup_test_and_build_commands(self):
        canonical_workflow = """uv python install 3.14
uv sync --locked --extra torch-cpu --extra hdf5
uv run --no-sync python -m unittest discover -v
uv build"""
        for name, contents in self.primary_documents.items():
            with self.subTest(document=name):
                self.assertIn(canonical_workflow, contents)

    def test_examples_and_mpi_use_the_uv_environment(self):
        readme = self.primary_documents["README.md"]
        self.assertIn("uv run --no-sync python examples/air2d.py", readme)
        self.assertIn("uv run --no-sync mpiexec", readme)
        self.assertIn(
            "uv run --no-sync python examples/<example file name>", self.examples_readme
        )
        self.assertIn("uv run --no-sync mpiexec", self.examples_readme)
        self.assertNotIn("$ python examples/", self.examples_readme)

    def test_lock_migration_and_native_prerequisites_are_documented(self):
        readme = self.primary_documents["README.md"]
        contributing = self.primary_documents["CONTRIBUTING.md"]
        agents = self.primary_documents["AGENTS.md"]
        self.assertIn("uv lock --upgrade", readme)
        self.assertIn("PEP 735", readme)
        self.assertIn("build-essential", readme)
        self.assertIn("xcode-select --install", readme)
        self.assertIn("libopenmpi-dev openmpi-bin", readme)
        self.assertIn("brew install open-mpi", readme)
        self.assertIn("contiguous-indexing fallback", readme)
        self.assertIn("tests/test_packaging.py", contributing)
        self.assertIn("uv lock --upgrade", agents)


if __name__ == "__main__":
    unittest.main()
