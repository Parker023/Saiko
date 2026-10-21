"""The UI must be an HTTP client, not a second copy of the backend.

Unusual tests - they assert on source code rather than behaviour. Worth
having because the shortcut is one import away and would work perfectly:
`from jobagency.analyse import overview` in ui.py runs fine, and silently
makes the whole API decorative. Nothing else would ever catch it.
"""

import ast
from pathlib import Path

import pytest

UI_FILES = ("jobagency/ui.py", "jobagency/ui_client.py")

# The UI may import its own client and nothing else from the package.
FORBIDDEN = {
    "jobagency.analyse",
    "jobagency.api",
    "jobagency.chart",
    "jobagency.fetch",
    "jobagency.main",
    "jobagency.models",
    "jobagency.normalise",
}


def _imported_modules(path: Path) -> set[str]:
    """Every fully-qualified module name imported by this file."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            # `from jobagency import ui_client` -> jobagency.ui_client
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
            names.add(node.module)
    return names


@pytest.mark.parametrize("filename", UI_FILES)
def test_ui_does_not_import_the_backend(filename):
    imported = _imported_modules(Path(filename))
    leaked = imported & FORBIDDEN
    assert not leaked, (
        f"{filename} imports {sorted(leaked)} directly. The UI must reach the "
        f"backend over HTTP via ui_client, or FastAPI is decorative."
    )


def test_only_the_client_module_uses_requests():
    """One place knows the API exists, so one place handles it being down."""
    assert "requests" in _imported_modules(Path("jobagency/ui_client.py"))
    assert "requests" not in _imported_modules(Path("jobagency/ui.py"))


def test_ui_reaches_the_backend_only_through_ui_client():
    """Whitelist, not blacklist: a NEW backend module would also be caught.

    `from jobagency.ui_client import ApiError` yields a name that is a class,
    not a module, so compare only the first two dotted parts.
    """
    imported = _imported_modules(Path("jobagency/ui.py"))
    modules = {
        ".".join(name.split(".")[:2])
        for name in imported
        if name.startswith("jobagency.")
    }
    assert modules == {"jobagency.ui_client"}


def test_api_base_is_configurable():
    """So the same UI can point at a deployed backend without a code change."""
    source = Path("jobagency/ui_client.py").read_text()
    assert "environ" in source, "API_BASE should be overridable by env var"
