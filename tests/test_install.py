from __future__ import annotations

import subprocess
import sys

import install


def test_fresh_install_enables_lifecycle_observer(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(install, "_hermes_executable", lambda: "/fake/hermes")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: calls.append((args, kwargs)) or subprocess.CompletedProcess(args, 0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["install.py", "--hermes-home", str(tmp_path / ".hermes")],
    )

    assert install.main() == 0
    assert calls[0][0] == [
        "/fake/hermes",
        "plugins",
        "enable",
        "consolidating_local",
        "--no-allow-tool-override",
    ]


def test_update_preserves_existing_enablement_and_grants(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "plugins" / "consolidating_local").mkdir(parents=True)

    def unexpected_enable(*args, **kwargs):
        raise AssertionError("an update must not re-enable the plugin")

    monkeypatch.setattr(install, "_hermes_executable", lambda: "/fake/hermes")
    monkeypatch.setattr(subprocess, "run", unexpected_enable)
    monkeypatch.setattr(
        sys,
        "argv",
        ["install.py", "--hermes-home", str(home)],
    )

    assert install.main() == 0
