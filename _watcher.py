#!/usr/bin/env python3
"""
File watcher that automatically formats and lints Python files on save.

Uses ruff for both formatting and linting, and bandit for security checks.
Provides clickable links to warnings and errors for easy navigation.
"""

import json
import logging
import subprocess  # nosec: S404
import time
from pathlib import Path
from typing import ClassVar

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class ColoredFormatter(logging.Formatter):

    """Custom formatter with colors for different log levels."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class PythonFileHandler(FileSystemEventHandler):

    """Handler for Python file changes."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the file handler."""
        self.project_root = project_root
        self.last_modified: dict[Path, float] = {}
        self.debounce_delay = 0.5  # seconds

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only process Python files
        if file_path.suffix != ".py":
            return

        # Skip if file is in common ignore directories
        if any(part in str(file_path) for part in ["__pycache__", ".git", ".venv", "venv", "node_modules"]):
            return

        # Debounce rapid file changes
        current_time = time.time()
        if file_path in self.last_modified and current_time - self.last_modified[file_path] < self.debounce_delay:
            return

        self.last_modified[file_path] = current_time

        # Run ruff format and lint
        self._run_ruff(file_path)

        # Run bandit security check
        self._run_bandit(file_path)

    def _run_ruff(self, file_path: Path) -> None:
        """Run ruff format and lint on the specified file."""
        try:
            logging.info("Processing %s", file_path.relative_to(self.project_root))

            # Run ruff format
            format_result = subprocess.run(  # nosec: S603
                ["uv", "run", "ruff", "format", str(file_path)],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if format_result.returncode == 0:
                logging.info("✅ Formatted %s", file_path.name)
            else:
                logging.warning("⚠️  Format issues in %s: %s", file_path.name, format_result.stderr)

            # Run ruff lint
            lint_result = subprocess.run(  # nosec: S603
                ["uv", "run", "ruff", "check", str(file_path), "--fix"],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if lint_result.returncode == 0:
                logging.info("✅ Linted %s", file_path.name)
            else:
                # Parse and log lint issues with clickable links
                lint_issues = self._parse_ruff_output(lint_result.stderr, file_path)
                if lint_issues:
                    self._log_issues_with_links(lint_issues, "Lint")
                else:
                    # Try parsing stdout if stderr is empty
                    lint_issues = self._parse_ruff_output(lint_result.stdout, file_path)
                    if lint_issues:
                        self._log_issues_with_links(lint_issues, "Lint")
                    else:
                        # Debug: show what we got
                        if lint_result.stderr.strip():
                            logging.warning("⚠️  Lint issues in %s: %s", file_path.name, lint_result.stderr)
                        if lint_result.stdout.strip():
                            logging.warning("⚠️  Lint stdout in %s: %s", file_path.name, lint_result.stdout)

        except Exception:
            logging.exception("❌ Error processing %s", file_path.name)

    def _run_bandit(self, file_path: Path) -> None:
        """Run bandit security check on the specified file."""
        try:
            # Run bandit with JSON output for easier parsing
            bandit_result = subprocess.run(  # nosec: S603 # pylint: disable=subprocess-run-check
                ["uv", "run", "bandit", "-f", "json", str(file_path)],  # pylint: disable=subprocess-run-check
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if bandit_result.returncode == 0:
                logging.info("✅ Bandit security check passed for %s", file_path.name)
            else:
                # Parse bandit JSON output
                try:
                    bandit_data = json.loads(bandit_result.stdout)
                    bandit_issues = self._parse_bandit_output(bandit_data, file_path)
                    if bandit_issues:
                        self._log_issues_with_links(bandit_issues, "Security")
                    else:
                        logging.warning("⚠️  Bandit issues in %s: %s", file_path.name, bandit_result.stderr)
                except json.JSONDecodeError:
                    logging.warning(
                        "⚠️  Bandit issues in %s (could not parse JSON): %s",
                        file_path.name,
                        bandit_result.stderr,
                    )

        except Exception:
            logging.exception("❌ Error running bandit on %s", file_path.name)

    def _create_clickable_link(self, file_path: Path, line_number: int, column_number: int | None) -> str:
        """Create a clickable file:// URL for the IDE."""
        file_url = f"file://{file_path.absolute()}"
        if line_number:
            file_url += f":{line_number}"
            if column_number:
                file_url += f":{column_number}"
        return file_url

    def _parse_ruff_output(self, output: str, file_path: Path) -> list[dict]:
        """Parse ruff output to extract warnings and errors."""
        if not output.strip():
            return []

        def parse_location_line(line: str):
            # Example: "  --> watcher.py:23:14"
            location_part = line.split("  --> ")[1].strip()
            if ":" in location_part:
                _, line_col = location_part.split(":", 1)
                if ":" in line_col:
                    line_num, col_num = line_col.split(":", 1)
                    return int(line_num), int(col_num)
            return None, None

        def find_code_and_message(lines, idx):
            # Look for the error code and message in previous lines
            for prev_line in reversed(lines[:idx]):
                if prev_line.strip() and not prev_line.startswith("  "):
                    parts = prev_line.split(" ", 1)
                    if len(parts) == 2:
                        return parts[0], parts[1]
            return "UNKNOWN", "Unknown error"

        issues = []
        lines = output.strip().split("\n")
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            if "  --> " in line:
                line_num, col_num = parse_location_line(line)
                code, message = find_code_and_message(lines, idx)
                issues.append(
                    {
                        "file": file_path,
                        "line": line_num,
                        "column": col_num,
                        "code": code,
                        "message": message,
                        "clickable_link": self._create_clickable_link(file_path, line_num, col_num),
                    },
                )
            elif ":" in line and not line.startswith("  "):
                # Example: watcher.py:23:14: F401: message
                parts = line.split(":", 4)
                count = 5
                if len(parts) >= count:
                    _, line_num, col_num, code, message = parts
                    try:
                        line_num = int(line_num)
                        col_num = int(col_num)
                    except Exception:
                        continue
                    issues.append(
                        {
                            "file": file_path,
                            "line": line_num,
                            "column": col_num,
                            "code": code.strip(),
                            "message": message.strip(),
                            "clickable_link": self._create_clickable_link(file_path, line_num, col_num),
                        },
                    )
        return issues

    def _parse_bandit_output(self, bandit_data: dict, file_path: Path) -> list[dict]:
        """Parse bandit JSON output to extract security issues."""
        issues = []

        if "results" in bandit_data:
            for result in bandit_data["results"]:
                line_number = result.get("line_number", 1)
                column_number = result.get("col_offset", 1)
                test_id = result.get("test_id", "Unknown")
                issue_severity = result.get("issue_severity", "Unknown")
                issue_confidence = result.get("issue_confidence", "Unknown")
                issue_text = result.get("issue_text", "No description available")

                message = f"[{issue_severity}/{issue_confidence}] {issue_text}"

                issues.append(
                    {
                        "file": file_path,
                        "line": line_number,
                        "column": column_number,
                        "code": test_id,
                        "message": message,
                        "clickable_link": self._create_clickable_link(file_path, line_number, column_number),
                    },
                )

        return issues

    def _log_issues_with_links(self, issues: list[dict], issue_type: str) -> None:
        """Log issues with clickable links."""
        if not issues:
            return

        for issue in issues:
            relative_path = issue["file"].relative_to(self.project_root)
            logging.warning(
                "🔍 %s issue in %s:%d:%d - %s %s\n🔗 Click to open: %s",
                issue_type,
                relative_path,
                issue["line"],
                issue["column"],
                issue["code"],
                issue["message"],
                issue["clickable_link"],
            )


def main() -> None:
    """Start the file watcher."""
    project_root = Path(__file__).parent.absolute()

    # Configure logging with colors
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console_handler)

    logging.info("🚀 Starting file watcher for %s", project_root.name)
    logging.info("🔧 Auto-formatting and linting python files with ruff")
    logging.info("🔒 Running security checks with bandit")
    logging.info("Press Ctrl+C to stop")
    logging.info("-" * 50)

    # Create event handler
    event_handler = PythonFileHandler(project_root)

    # Create observer
    observer = Observer()
    observer.schedule(event_handler, str(project_root), recursive=True)

    # Start watching
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("🛑 Stopping file watcher...")
        observer.stop()

    observer.join()
    logging.info("✅ File watcher stopped")


if __name__ == "__main__":
    main()
