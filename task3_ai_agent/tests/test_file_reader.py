"""Unit tests for Local File Reader Tool.

Ensures proper workspace isolation, directory traversal prevention,
extension whitelist enforcement, size limitations, and untrusted data envelopes.
Does not require any Gemini API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from tools import read_workspace_file, ALLOWED_FILE_EXTENSIONS


@pytest.fixture
def mock_workspace(tmp_path: Path) -> Path:
    """Creates a temporary workspace directory populated with test files."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Valid test files
    (workspace / "notes.txt").write_text("Project notes and features.", encoding="utf-8")
    (workspace / "README.md").write_text("# Project Docs\nMarkdown content.", encoding="utf-8")
    (workspace / "data.csv").write_text("id,name,value\n1,alpha,100\n2,beta,200", encoding="utf-8")
    (workspace / "config.json").write_text(json.dumps({"env": "testing", "debug": True}), encoding="utf-8")

    # File outside workspace for traversal testing
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("CLASSIFIED_SYSTEM_KEY_12345", encoding="utf-8")

    return workspace


class TestFileReaderOperations:
    """Tests for reading permitted file formats from the workspace."""

    def test_read_txt_file(self, mock_workspace: Path) -> None:
        result = read_workspace_file("notes.txt", workspace_dir=mock_workspace)
        assert "BEGIN UNTRUSTED FILE DATA: notes.txt" in result
        assert "Project notes and features." in result
        assert "SECURITY NOTICE" in result

    def test_read_markdown_file(self, mock_workspace: Path) -> None:
        result = read_workspace_file("README.md", workspace_dir=mock_workspace)
        assert "# Project Docs" in result
        assert "Markdown content." in result

    def test_read_csv_file(self, mock_workspace: Path) -> None:
        result = read_workspace_file("data.csv", workspace_dir=mock_workspace)
        assert "id,name,value" in result
        assert "1,alpha,100" in result

    def test_read_json_file(self, mock_workspace: Path) -> None:
        result = read_workspace_file("config.json", workspace_dir=mock_workspace)
        assert '"env": "testing"' in result

    def test_read_subfolder_file_within_workspace(self, mock_workspace: Path) -> None:
        subfolder = mock_workspace / "sub"
        subfolder.mkdir()
        (subfolder / "nested.txt").write_text("Nested file content.", encoding="utf-8")

        result = read_workspace_file("sub/nested.txt", workspace_dir=mock_workspace)
        assert "Nested file content." in result


class TestFileReaderSecurityGuardrails:
    """Tests for security restrictions, path traversal, and file format validation."""

    def test_path_traversal_relative_parent_directory(self, mock_workspace: Path) -> None:
        result = read_workspace_file("../secret.txt", workspace_dir=mock_workspace)
        assert "Security Error: Access denied" in result
        assert "CLASSIFIED" not in result

    def test_path_traversal_deep_relative(self, mock_workspace: Path) -> None:
        result = read_workspace_file("../../../../../etc/passwd", workspace_dir=mock_workspace)
        assert "Security Error: Access denied" in result

    def test_unsupported_file_extension_exe(self, mock_workspace: Path) -> None:
        malicious = mock_workspace / "payload.exe"
        malicious.write_text("binary-data", encoding="utf-8")

        result = read_workspace_file("payload.exe", workspace_dir=mock_workspace)
        assert "Unsupported file format" in result
        assert ".exe" in result

    def test_unsupported_file_extension_python(self, mock_workspace: Path) -> None:
        script = mock_workspace / "script.py"
        script.write_text("import os; os.system('calc')", encoding="utf-8")

        result = read_workspace_file("script.py", workspace_dir=mock_workspace)
        assert "Unsupported file format" in result
        assert ".py" in result

    def test_missing_file_handling(self, mock_workspace: Path) -> None:
        result = read_workspace_file("nonexistent_file.txt", workspace_dir=mock_workspace)
        assert "not found in workspace" in result

    def test_empty_filename_handling(self, mock_workspace: Path) -> None:
        assert "cannot be empty" in read_workspace_file("", workspace_dir=mock_workspace)
        assert "cannot be empty" in read_workspace_file("   ", workspace_dir=mock_workspace)

    def test_file_size_limit_enforcement(self, mock_workspace: Path) -> None:
        large_file = mock_workspace / "large.txt"
        # Write 250 KB
        large_file.write_text("A" * (250 * 1024), encoding="utf-8")

        # Set max_size to 200 KB
        result = read_workspace_file("large.txt", workspace_dir=mock_workspace, max_size_bytes=200 * 1024)
        assert "exceeds the maximum limit" in result
