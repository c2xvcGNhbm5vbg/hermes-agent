"""The managed-install venv console shim is a launch-context artifact, not a durable launcher.

The install.sh layout ships BOTH a git checkout (``~/.hermes/hermes-agent``, with
``apps/desktop`` and the packaged ``release/linux-unpacked/Hermes``) and a managed
venv generation (``~/.hermes/installs/<hash>/environments/<hash>/venv``). The venv
generation's ``workspace`` is a Python build snapshot with NO ``apps/`` tree, so
``hermes desktop`` from it can never build or launch the GUI.

The desktop-update hand-off puts ``<checkout>/venv/bin`` at the front of PATH; a
managed-venv-context launch then resolves the venv console script as the "primary".
Because that script is NOT inside the checkout, the external-primary-wins rule used
to persist it as the ``Exec=`` target — pinning the entry to a shim that can never
launch the GUI, and re-pinning it on every subsequent launch/update (self-
perpetuating clobber of a hand-fixed entry).

A managed-venv shim must be classified as a launch-context artifact (like a
checkout-internal candidate) and skipped in favor of the durable wrapper probe.
A GENUINE external primary (``/opt/.../bin/hermes`` from another install) must
still win — the external-primary-wins rule is preserved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hermes_cli import linux_desktop_entry as lde


@pytest.fixture
def managed_venv_home(tmp_path, monkeypatch) -> Path:
    """A managed-venv generation under an isolated HERMES_HOME."""
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(lde.sys, "platform", "linux")
    gen = hermes_home / "installs" / "d7c46095785b5dfd" / "environments" / "6b1e4cd1aeb646b49b904ea657c0e2d2"
    venv_bin = gen / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    shim = venv_bin / "hermes"
    shim.write_text("#!/bin/sh\nexec true\n", encoding="utf-8")
    shim.chmod(0o755)
    return hermes_home


def _make_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "hermes-agent"
    icon = root / "apps" / "desktop" / "assets" / "icon.png"
    icon.parent.mkdir(parents=True)
    icon.write_bytes(b"\x89PNG fake")
    return root


def _durable_wrapper(tmp_path: Path, root: Path) -> Path:
    wrapper = tmp_path / ".local" / "bin" / "hermes"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/usr/bin/env bash\nexec true {root} "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def test_managed_venv_shim_on_path_is_skipped_for_durable_wrapper(
    tmp_path, managed_venv_home, monkeypatch
):
    """A managed-venv console shim on PATH (front) must NOT be persisted.

    The resolver's primary is the venv shim (a per-generation build artifact,
    not a durable launcher). It is not inside the checkout, so the old
    external-primary-wins rule returned it immediately. It must instead be
    skipped and the durable wrapper (``~/.local/bin/hermes``) returned.
    """
    root = _make_checkout(tmp_path)
    wrapper = _durable_wrapper(tmp_path, root)
    venv_shim = managed_venv_home / "installs" / "d7c46095785b5dfd" / "environments" / "6b1e4cd1aeb646b49b904ea657c0e2d2" / "venv" / "bin" / "hermes"

    # The hand-off context: the venv shim is the primary (front of PATH).
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: str(venv_shim))
    monkeypatch.setattr(sys, "argv", [str(venv_shim), "desktop"])
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    resolved = lde._resolve_hermes_bin_for_desktop_entry(checkout_root=root)
    assert resolved == str(wrapper)
    assert str(venv_shim) not in (resolved or "")


def test_genuine_external_primary_still_wins(tmp_path, managed_venv_home, monkeypatch):
    """A genuine external launcher (``/opt/.../bin/hermes``) is NOT a managed-venv
    shim — the external-primary-wins rule is preserved (no silent switch to a
    different installation, #94443 review case 3).
    """
    root = _make_checkout(tmp_path)
    _durable_wrapper(tmp_path, root)
    external = tmp_path / "opt-hermes" / "bin" / "hermes"
    external.parent.mkdir(parents=True)
    external.write_text("#!/bin/sh\nexec true\n", encoding="utf-8")
    external.chmod(0o755)

    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: str(external))
    monkeypatch.setattr(sys, "argv", [str(external), "desktop"])
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    resolved = lde._resolve_hermes_bin_for_desktop_entry(checkout_root=root)
    assert resolved == str(external)
