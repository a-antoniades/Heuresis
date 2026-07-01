"""Tests for the in-sandbox DNS resolver bind-mount in bwrap.

The sandbox bind-mounts /etc by default. That covers a regular-file
/etc/resolv.conf. When /etc/resolv.conf is a symlink into /run/* (systemd-
resolved on some GCE VMs, dnsmasq runtime dirs, etc.), the /etc bind keeps
the symlink but its target is absent inside the sandbox, breaking DNS.
``_mount_dns_resolver`` resolves the symlink at command-construction time
and binds the target's parent directory when it lives outside /etc.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from heuresis import _bwrap


def test_regular_file_resolv_conf_is_a_noop(monkeypatch, tmp_path: Path) -> None:
    """A non-symlink resolv.conf needs no extra mount; the /etc bind covers it."""
    fake = tmp_path / "resolv.conf"
    fake.write_text("nameserver 8.8.8.8\n")
    monkeypatch.setattr("heuresis._bwrap._DNS_CONFIG", fake)

    cmd: list[str] = []
    _bwrap._mount_dns_resolver(cmd)

    assert cmd == []


def test_systemd_resolved_symlink_binds_parent_dir(monkeypatch, tmp_path: Path) -> None:
    """The original GCE systemd-resolved layout: symlink → /run/systemd/resolve."""
    target_dir = tmp_path / "run-systemd-resolve"
    target_dir.mkdir()
    target = target_dir / "resolv.conf"
    target.write_text("nameserver 169.254.169.254\n")

    symlink = tmp_path / "etc-resolv.conf"
    symlink.symlink_to(target)
    monkeypatch.setattr("heuresis._bwrap._DNS_CONFIG", symlink)

    cmd: list[str] = []
    _bwrap._mount_dns_resolver(cmd)

    assert cmd == ["--ro-bind", str(target_dir), str(target_dir)]


def test_non_systemd_resolver_symlink_also_binds_parent(monkeypatch, tmp_path: Path) -> None:
    """A symlink target outside /etc that is NOT /run/systemd/resolve must
    still be handled — e.g. dnsmasq, NetworkManager, custom resolvers."""
    target_dir = tmp_path / "run-NetworkManager"
    target_dir.mkdir()
    target = target_dir / "resolv.conf"
    target.write_text("nameserver 1.1.1.1\n")

    symlink = tmp_path / "etc-resolv.conf"
    symlink.symlink_to(target)
    monkeypatch.setattr("heuresis._bwrap._DNS_CONFIG", symlink)

    cmd: list[str] = []
    _bwrap._mount_dns_resolver(cmd)

    assert cmd == ["--ro-bind", str(target_dir), str(target_dir)]


def test_symlink_target_already_inside_etc_is_a_noop(monkeypatch, tmp_path: Path) -> None:
    """If /etc/resolv.conf points to another file already inside /etc (e.g.
    /etc/resolv.conf → /etc/resolvconf/resolv.conf.d/original), the /etc
    bind already covers the target, so no extra mount is needed."""
    fake_etc = tmp_path / "etc"
    fake_etc.mkdir()
    target = fake_etc / "resolv.conf.real"
    target.write_text("nameserver 8.8.8.8\n")

    symlink = fake_etc / "resolv.conf"
    symlink.symlink_to(target)

    monkeypatch.setattr("heuresis._bwrap._DNS_CONFIG", symlink)
    monkeypatch.setattr("heuresis._bwrap._ETC_DIR", fake_etc)

    cmd: list[str] = []
    _bwrap._mount_dns_resolver(cmd)

    assert cmd == []


def test_dangling_symlink_raises(monkeypatch, tmp_path: Path) -> None:
    """A symlink whose target does not exist must fail loudly. Silently
    launching agents with a broken DNS path was the original GCE failure
    mode and is the worst-case outcome."""
    symlink = tmp_path / "etc-resolv.conf"
    symlink.symlink_to(tmp_path / "missing" / "resolv.conf")
    monkeypatch.setattr("heuresis._bwrap._DNS_CONFIG", symlink)

    cmd: list[str] = []
    with pytest.raises(RuntimeError, match="does not exist"):
        _bwrap._mount_dns_resolver(cmd)


def test_multi_hop_symlink_chain_raises(monkeypatch, tmp_path: Path) -> None:
    """A symlink whose immediate target path includes another symlink as
    one of its parent components is a multi-hop chain. The kernel inside
    the sandbox walks the path component-by-component and would have to
    follow a symlink we have not bound (e.g. /var/run -> /run on Debian-
    derived hosts). Refuse to launch with a clear error rather than
    silently break DNS."""
    real_run = tmp_path / "real-run"
    (real_run / "resolver").mkdir(parents=True)
    target_file = real_run / "resolver" / "resolv.conf"
    target_file.write_text("nameserver 8.8.8.8\n")

    var_run = tmp_path / "var-run"
    var_run.symlink_to(real_run)  # /var/run -> /run pattern

    etc_link = tmp_path / "etc-resolv.conf"
    etc_link.symlink_to(var_run / "resolver" / "resolv.conf")
    monkeypatch.setattr("heuresis._bwrap._DNS_CONFIG", etc_link)

    cmd: list[str] = []
    with pytest.raises(RuntimeError, match="multi-hop|symlink chain|ancestor"):
        _bwrap._mount_dns_resolver(cmd)


def test_build_command_invokes_dns_helper(monkeypatch, tmp_path: Path) -> None:
    """End-to-end: build_command() must wire _mount_dns_resolver in.
    Catches the regression of someone removing the call site without
    catching it via helper-only tests (Codex 2026-05-01 review)."""
    target_dir = tmp_path / "run-resolver"
    target_dir.mkdir()
    target = target_dir / "resolv.conf"
    target.write_text("nameserver 1.1.1.1\n")

    symlink = tmp_path / "etc-resolv.conf"
    symlink.symlink_to(target)
    monkeypatch.setattr("heuresis._bwrap._DNS_CONFIG", symlink)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cmd = _bwrap.build_command(workspace=workspace, inner_cmd=["echo", "hi"])

    # The DNS bind for the resolver target dir must appear in the bwrap argv.
    flat = " ".join(cmd)
    assert f"--ro-bind {target_dir} {target_dir}" in flat
