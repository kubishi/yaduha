"""Tests for MCP Language Development Server."""

import tempfile
from pathlib import Path

import pytest

from yaduha.mcp_server import LanguageDevelopmentServer


class TestLanguageDevelopmentServer:
    """Test LanguageDevelopmentServer functionality."""

    @pytest.fixture
    def temp_language_dir(self):
        """Create a temporary language package directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create basic language package structure
            (tmppath / "tests").mkdir()
            (tmppath / "tests" / "__init__.py").write_text("")
            (tmppath / "tests" / "test_language.py").write_text(
                """
import pytest

def test_basic():
    assert True
"""
            )

            # Create main module
            (tmppath / "my_language").mkdir()
            (tmppath / "my_language" / "__init__.py").write_text(
                """
from pydantic import BaseModel

class SimpleSentence(BaseModel):
    text: str

    def __str__(self) -> str:
        return self.text
"""
            )

            # Create pyproject.toml
            (tmppath / "pyproject.toml").write_text(
                """
[project]
name = "yaduha-test-lang"
version = "0.1.0"
"""
            )

            # Create README
            (tmppath / "README.md").write_text("# Test Language\n")

            yield tmppath

    def test_init_with_valid_path(self, temp_language_dir):
        """Test server initialization with valid path."""
        server = LanguageDevelopmentServer(str(temp_language_dir))
        assert server.language_path == temp_language_dir
        assert server.language_code is not None

    def test_init_with_invalid_path(self):
        """Test server initialization with invalid path."""
        with pytest.raises(ValueError, match="does not exist"):
            LanguageDevelopmentServer("/nonexistent/path")

    def test_read_file(self, temp_language_dir):
        """Test reading a file."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        content = server.read_file("README.md")
        assert "Test Language" in content

    def test_read_file_not_found(self, temp_language_dir):
        """Test reading a nonexistent file."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        result = server.read_file("nonexistent.py")
        assert "Error" in result

    def test_read_file_outside_package(self, temp_language_dir):
        """Test that reading outside package is prevented."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        result = server.read_file("../../etc/passwd")
        assert "outside" in result.lower()

    def test_write_file(self, temp_language_dir):
        """Test writing a file."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        result = server.write_file("my_language/vocab.py", "VOCAB = {}")
        assert "Success" in result
        assert (temp_language_dir / "my_language" / "vocab.py").exists()

    def test_write_protected_file_pyproject(self, temp_language_dir):
        """Test that protected files cannot be written."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        result = server.write_file("pyproject.toml", "[bad]")
        assert "protected" in result.lower()

    def test_write_protected_file_tests(self, temp_language_dir):
        """Test that test files cannot be written."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        result = server.write_file("tests/test_hack.py", "bad")
        assert "protected" in result.lower()

    def test_write_file_outside_package(self, temp_language_dir):
        """Test that writing outside package is prevented."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        result = server.write_file("../../etc/evil", "bad")
        assert "outside" in result.lower()

    def test_delete_file(self, temp_language_dir):
        """Test deleting a file."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        # Create a user file
        user_file = temp_language_dir / "custom_module.py"
        user_file.write_text("# user defined")

        result = server.delete_file("custom_module.py")
        assert "Success" in result or "Deleted" in result
        assert not user_file.exists()

    def test_delete_protected_file_init(self, temp_language_dir):
        """Test that core files cannot be deleted."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        result = server.delete_file("my_language/__init__.py")
        assert "protected" in result.lower()

    def test_delete_tests_directory(self, temp_language_dir):
        """Test that tests directory cannot be deleted."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        result = server.delete_file("tests")
        assert "protected" in result.lower()

    def test_list_package_structure(self, temp_language_dir):
        """Test listing package structure."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        structure = server.list_package_structure()
        assert "README.md" in structure
        assert "pyproject.toml" in structure

    def test_show_git_status(self, temp_language_dir):
        """Test git status command."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=str(temp_language_dir), capture_output=True)

        result = server.show_git_status()
        # Result might be empty or show files
        assert isinstance(result, str)

    def test_run_tests(self, temp_language_dir):
        """Test running pytest."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        result = server.run_tests()
        assert "All tests passed" in result or "Tests failed" in result or "Error" in result

    def test_commit_changes(self, temp_language_dir):
        """Test git commit."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=str(temp_language_dir), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(temp_language_dir),
            capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=str(temp_language_dir),
            capture_output=True
        )

        # Make a change
        server.write_file("test.txt", "hello")

        # Try to commit
        result = server.commit_changes("Initial commit")
        assert "Committed" in result or "Error" in result

    def test_revert_file(self, temp_language_dir):
        """Test reverting a file."""
        server = LanguageDevelopmentServer(str(temp_language_dir))

        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=str(temp_language_dir), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(temp_language_dir),
            capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=str(temp_language_dir),
            capture_output=True
        )

        # Create and commit a file
        test_file = temp_language_dir / "tracked.txt"
        test_file.write_text("original")
        subprocess.run(
            ["git", "add", "."],
            cwd=str(temp_language_dir),
            capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=str(temp_language_dir),
            capture_output=True
        )

        # Modify it
        test_file.write_text("modified")

        # Revert
        result = server.revert_file("tracked.txt")
        assert "Reverted" in result or "Error" in result
