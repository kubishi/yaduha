"""Language Development MCP Server.

Implements Model Context Protocol server for interactive language development
with Claude or other LLMs.
"""

import subprocess
from pathlib import Path


class LanguageDevelopmentServer:
    """MCP Server for language package development.

    Exposes tools for reading, writing, testing, and validating language packages.
    """

    def __init__(self, language_path: str) -> None:
        """Initialize the server.

        Args:
            language_path: Path to the language package directory
        """
        self.language_path = Path(language_path).resolve()
        if not self.language_path.exists():
            raise ValueError(f"Language path does not exist: {language_path}")

        # Load language metadata
        self._load_language()

    def _load_language(self) -> None:
        """Load language package metadata."""
        try:
            # Try to load from pyproject.toml
            pyproject_path = self.language_path / "pyproject.toml"
            if pyproject_path.exists():
                try:
                    import tomllib  # Python 3.11+
                except ImportError:
                    import tomli as tomllib  # type: ignore
                with open(pyproject_path, "rb") as f:
                    config = tomllib.load(f)
                    self.language_name = config.get("project", {}).get("name", "unknown")
                    self.language_code = self.language_path.name.replace("yaduha-", "")
            else:
                self.language_code = self.language_path.name
                self.language_name = self.language_path.name
        except Exception:
            self.language_code = self.language_path.name
            self.language_name = self.language_path.name

    def read_file(self, path: str) -> str:
        """Read a file from the language package.

        Args:
            path: Relative path from language package root

        Returns:
            File contents
        """
        file_path = (self.language_path / path).resolve()

        # Prevent reading outside language package
        if not str(file_path).startswith(str(self.language_path)):
            return "Error: Cannot read files outside language package"

        if not file_path.exists():
            return f"Error: File not found: {path}"

        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, path: str, content: str) -> str:
        """Write to a file in the language package.

        Args:
            path: Relative path from language package root
            content: File contents to write

        Returns:
            Success/error message
        """
        file_path = (self.language_path / path).resolve()

        # Prevent writing outside language package
        if not str(file_path).startswith(str(self.language_path)):
            return "Error: Cannot write files outside language package"

        # Check protected files
        protected = {"tests", "pyproject.toml", "README.md", ".git", ".gitignore"}
        if file_path.name in protected or any(p in file_path.parts for p in protected):
            return f"Error: Cannot modify protected file: {path}"

        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            # Automatically run tests after write
            test_result = self.run_tests()

            return f"Success: File written to {path}\n\nTest Results:\n{test_result}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def delete_file(self, path: str) -> str:
        """Delete a file from the language package.

        Args:
            path: Relative path from language package root

        Returns:
            Success/error message
        """
        file_path = (self.language_path / path).resolve()

        # Prevent deleting outside language package
        if not str(file_path).startswith(str(self.language_path)):
            return "Error: Cannot delete files outside language package"

        # Only allow deleting user-defined files (not core files)
        protected = {
            "__init__.py", "pyproject.toml", "README.md", ".git",
            ".gitignore", "tests", "setup.py"
        }
        if file_path.name in protected or any(p in file_path.parts for p in protected):
            return f"Error: Cannot delete protected file: {path}"

        try:
            if file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)
            else:
                file_path.unlink()
            return f"Success: Deleted {path}"
        except Exception as e:
            return f"Error deleting file: {str(e)}"

    def list_package_structure(self) -> str:
        """Show directory structure of the language package.

        Returns:
            Directory tree as string
        """
        try:
            result = subprocess.run(
                ["find", str(self.language_path), "-type", "f", "-not", "-path", "*/.*"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return f"Error: {result.stderr}"

            files = sorted(result.stdout.strip().split("\n"))
            # Make paths relative
            rel_files = [str(Path(f).relative_to(self.language_path)) for f in files if f]
            return "\n".join(rel_files)
        except Exception as e:
            return f"Error listing structure: {str(e)}"

    def show_git_status(self) -> str:
        """Show git status of the language package.

        Returns:
            Git status output
        """
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(self.language_path),
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return f"Error: {result.stderr}"

            return result.stdout or "No changes"
        except Exception as e:
            return f"Error getting git status: {str(e)}"

    def run_tests(self) -> str:
        """Execute pytest for the language package.

        Returns:
            Test results (pass/fail with output)
        """
        try:
            result = subprocess.run(
                ["pytest", "tests/", "-v", "--tb=short"],
                cwd=str(self.language_path),
                capture_output=True,
                text=True,
                timeout=30
            )

            output = result.stdout + result.stderr
            if result.returncode == 0:
                return f"✓ All tests passed\n\n{output}"
            else:
                return f"✗ Tests failed\n\n{output}"
        except FileNotFoundError:
            return "Error: pytest not found. Install with: pip install pytest"
        except subprocess.TimeoutExpired:
            return "Error: Tests timed out (>30 seconds)"
        except Exception as e:
            return f"Error running tests: {str(e)}"

    def translate_text(self, text: str) -> str:
        """Translate a single text using the language package.

        Args:
            text: Text to translate

        Returns:
            Translation or error message
        """
        # Feature not yet implemented in core server
        return "translate_text: Feature not yet fully implemented"

    def validate_language(self) -> str:
        """Run Yaduha's validation system.

        Returns:
            Validation errors or success message
        """
        # Feature not yet implemented in core server
        return "validate_language: Feature not yet fully implemented"

    def commit_changes(self, message: str) -> str:
        """Commit changes to git.

        Args:
            message: Commit message

        Returns:
            Success/error message
        """
        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(self.language_path),
                capture_output=True,
                timeout=5
            )

            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(self.language_path),
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return f"✓ Committed: {message}\n{result.stdout}"
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error committing: {str(e)}"

    def revert_file(self, path: str) -> str:
        """Revert a file to its previous git state.

        Args:
            path: Relative path from language package root

        Returns:
            Success/error message
        """
        try:
            result = subprocess.run(
                ["git", "checkout", path],
                cwd=str(self.language_path),
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return f"✓ Reverted {path}"
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error reverting: {str(e)}"

    def run(self, host: str = "localhost", port: int = 5000) -> None:
        """Run the MCP server.

        Args:
            host: Server host
            port: Server port
        """
        print(f"Starting Yaduha Language Development Server...")
        print(f"Language: {self.language_code} ({self.language_name})")
        print(f"Path: {self.language_path}")
        print(f"Listening on {host}:{port}")
        print()
        print("Server is ready for MCP connections.")
        print("Use this with: claude --mcp-server <connection-info>")
        print()
        print("Available tools:")
        print("  - read_file(path)")
        print("  - write_file(path, content) [auto-runs tests]")
        print("  - delete_file(path)")
        print("  - list_package_structure()")
        print("  - show_git_status()")
        print("  - run_tests()")
        print("  - translate_text(text)")
        print("  - validate_language()")
        print("  - commit_changes(message)")
        print("  - revert_file(path)")
        print()
        print("Press Ctrl+C to stop the server.")
        print()

        # Keep the server running
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nServer stopped.")
