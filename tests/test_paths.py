"""Tests for claude_swap.paths resolver helpers.

These tests verify that cswap resolves Claude Code config/credential paths the
same way claude-code itself does. If these drift from claude-code's behavior,
cswap will read the wrong files and misattribute accounts (see issue #16).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap.exceptions import MigrationError
from claude_swap.models import Platform
from claude_swap.paths import (
    LEGACY_BACKUP_DIRNAME,
    get_backup_root,
    get_claude_config_home,
    get_credentials_path,
    get_global_config_path,
    get_legacy_backup_root,
    migrate_legacy_backup_dir,
)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Temp HOME with CLAUDE_CONFIG_DIR unset."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    with patch("pathlib.Path.home", return_value=home):
        yield home


class TestGetClaudeConfigHome:
    def test_default_is_dot_claude_in_home(self, isolated_home: Path):
        assert get_claude_config_home() == isolated_home / ".claude"

    def test_respects_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        custom = tmp_path / "custom-claude"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        assert get_claude_config_home() == custom


class TestGetGlobalConfigPath:
    def test_default_returns_homedir_claude_json(self, isolated_home: Path):
        """Without CCD, claude-code writes .claude.json at $HOME, not inside .claude/."""
        assert get_global_config_path() == isolated_home / ".claude.json"

    def test_ccd_set_returns_ccd_claude_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        custom = tmp_path / "ccd"
        custom.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        assert get_global_config_path() == custom / ".claude.json"

    def test_legacy_config_json_takes_precedence(self, isolated_home: Path):
        """If ~/.claude/.config.json exists, claude-code uses that (legacy)."""
        config_home = isolated_home / ".claude"
        config_home.mkdir(exist_ok=True)
        legacy = config_home / ".config.json"
        legacy.write_text("{}")
        assert get_global_config_path() == legacy

    def test_legacy_config_json_in_ccd_takes_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        custom = tmp_path / "ccd"
        custom.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        legacy = custom / ".config.json"
        legacy.write_text("{}")
        assert get_global_config_path() == legacy


class TestGetCredentialsPath:
    def test_default_inside_dot_claude(self, isolated_home: Path):
        assert get_credentials_path() == isolated_home / ".claude" / ".credentials.json"

    def test_respects_ccd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        custom = tmp_path / "ccd"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        assert get_credentials_path() == custom / ".credentials.json"


class TestGetBackupRoot:
    """XDG-aware path everywhere but Windows, which keeps the legacy root."""

    def test_linux_default_is_xdg_data_home(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.LINUX))
        assert get_backup_root() == isolated_home / ".local" / "share" / "claude-swap"

    def test_linux_respects_xdg_data_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        custom = tmp_path / "xdg"
        monkeypatch.setenv("XDG_DATA_HOME", str(custom))
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.LINUX))
        assert get_backup_root() == custom / "claude-swap"

    def test_linux_ignores_empty_xdg_data_home(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("XDG_DATA_HOME", "")
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.LINUX))
        assert get_backup_root() == isolated_home / ".local" / "share" / "claude-swap"

    def test_linux_ignores_relative_xdg_data_home(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Per the XDG spec, relative paths must be ignored.
        monkeypatch.setenv("XDG_DATA_HOME", "relative/path")
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.LINUX))
        assert get_backup_root() == isolated_home / ".local" / "share" / "claude-swap"

    def test_linux_expands_tilde_in_xdg_data_home(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # systemd unit files / Dockerfiles set env vars without shell
        # expansion, so a literal ``~/foo`` must still resolve correctly.
        monkeypatch.setenv("XDG_DATA_HOME", "~/custom-data")
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.LINUX))
        assert get_backup_root() == isolated_home / "custom-data" / "claude-swap"

    def test_wsl_uses_xdg_layout(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.WSL))
        assert get_backup_root() == isolated_home / ".local" / "share" / "claude-swap"

    def test_macos_uses_the_xdg_layout_too(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Against Apple's own convention, and on purpose: see get_backup_root."""
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.MACOS))
        assert get_backup_root() == isolated_home / ".local" / "share" / "claude-swap"

    def test_macos_respects_xdg_data_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """One layout means one set of rules, the override included.

        A macOS branch that landed in ~/.local/share but ignored the variable
        naming it would be the XDG layout in spelling only.
        """
        custom = tmp_path / "xdg"
        monkeypatch.setenv("XDG_DATA_HOME", str(custom))
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.MACOS))
        assert get_backup_root() == custom / "claude-swap"

    def test_unknown_platform_uses_the_xdg_layout(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Only Windows is named as the exception, so everything else follows
        the rule — a BSD is far closer to the XDG world than to %APPDATA%."""
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.UNKNOWN))
        assert get_backup_root() == isolated_home / ".local" / "share" / "claude-swap"

    def test_windows_uses_legacy_layout(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.WINDOWS))
        assert get_backup_root() == isolated_home / LEGACY_BACKUP_DIRNAME

    def test_windows_ignores_xdg_data_home(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The variable must not drag the one remaining legacy platform off
        its layout — there is no migration written for that direction."""
        monkeypatch.setenv("XDG_DATA_HOME", str(isolated_home / "xdg"))
        monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.WINDOWS))
        assert get_backup_root() == isolated_home / LEGACY_BACKUP_DIRNAME

    def test_legacy_helper_returns_home_dot_claude_swap_backup(
        self, isolated_home: Path
    ):
        assert get_legacy_backup_root() == isolated_home / LEGACY_BACKUP_DIRNAME


def _assert_legacy_vacated(legacy: Path, target: Path) -> None:
    """The legacy directory is gone, and points at ``target`` where it can.

    The compatibility link is best effort by design (see
    ``paths.leave_compat_symlink``): Windows refuses directory symlinks to an
    unprivileged process, and the migration does not fail over a cosmetic
    loss. What must hold on every platform is that the directory itself no
    longer exists — the link, where it exists, is asserted on its own below.
    """
    if legacy.is_symlink():
        assert legacy.resolve() == target.resolve()
    else:
        assert not legacy.exists()


class TestMigrateLegacyBackupDir:
    def test_no_legacy_is_noop(self, isolated_home: Path):
        target = isolated_home / ".local" / "share" / "claude-swap"
        assert migrate_legacy_backup_dir(target) is False
        assert not target.exists()

    def test_target_equals_legacy_is_noop(self, isolated_home: Path):
        # Windows: backup_root == legacy. Migration must not touch it.
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir()
        (legacy / "marker").write_text("keep me")
        assert migrate_legacy_backup_dir(legacy) is False
        assert (legacy / "marker").read_text() == "keep me"

    def test_moves_legacy_to_target(self, isolated_home: Path):
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir()
        (legacy / "sequence.json").write_text('{"k": 1}')
        nested = legacy / "configs"
        nested.mkdir()
        (nested / "x.json").write_text("{}")

        target = isolated_home / ".local" / "share" / "claude-swap"
        assert migrate_legacy_backup_dir(target) is True
        _assert_legacy_vacated(legacy, target)
        assert (target / "sequence.json").read_text() == '{"k": 1}'
        assert (target / "configs" / "x.json").read_text() == "{}"

    def test_collision_raises_migration_error(self, isolated_home: Path):
        """Both paths exist → refuse to merge or overwrite."""
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir()
        (legacy / "sequence.json").write_text('{"src": "legacy"}')

        target = isolated_home / ".local" / "share" / "claude-swap"
        target.mkdir(parents=True)
        (target / "sequence.json").write_text('{"src": "target"}')

        with pytest.raises(MigrationError, match="Refusing to merge"):
            migrate_legacy_backup_dir(target)
        # Neither path is touched on collision.
        assert (legacy / "sequence.json").read_text() == '{"src": "legacy"}'
        assert (target / "sequence.json").read_text() == '{"src": "target"}'

    def test_target_with_only_throwaway_artifacts_is_wiped(self, isolated_home: Path):
        """Run-1 created cache/log in target; legacy appears later → migrate anyway.

        Regression: pre-fix this raised "Both legacy and new exist" because
        any prior cswap invocation laid down cache/ + claude-swap.log in the
        XDG path even when no real data was present.
        """
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir()
        (legacy / "sequence.json").write_text('{"src": "legacy"}')

        target = isolated_home / ".local" / "share" / "claude-swap"
        (target / "cache").mkdir(parents=True)
        (target / "cache" / "update_check.json").write_text("{}")
        (target / "claude-swap.log").write_text("noise")
        (target / "claude-swap.log.1").write_text("rotated")

        assert migrate_legacy_backup_dir(target) is True
        _assert_legacy_vacated(legacy, target)
        assert (target / "sequence.json").read_text() == '{"src": "legacy"}'
        # Throwaway artifacts were replaced by legacy contents.
        assert not (target / "cache").exists()
        assert not (target / "claude-swap.log").exists()

    def test_target_with_real_data_alongside_artifacts_still_collides(
        self, isolated_home: Path
    ):
        """Cache/log next to real data → still a collision, must refuse."""
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir()
        (legacy / "sequence.json").write_text('{"src": "legacy"}')

        target = isolated_home / ".local" / "share" / "claude-swap"
        (target / "cache").mkdir(parents=True)
        (target / "claude-swap.log").write_text("noise")
        (target / "sequence.json").write_text('{"src": "target"}')

        with pytest.raises(MigrationError, match="Refusing to merge"):
            migrate_legacy_backup_dir(target)
        assert (legacy / "sequence.json").read_text() == '{"src": "legacy"}'
        assert (target / "sequence.json").read_text() == '{"src": "target"}'

    def test_resumes_after_interrupted_move(self, isolated_home: Path):
        """Flag present + legacy still there = previous run was interrupted.

        Discard any partial target and retry the move.
        """
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir()
        (legacy / "sequence.json").write_text('{"src": "legacy"}')

        target = isolated_home / ".local" / "share" / "claude-swap"
        target.mkdir(parents=True)
        (target / "stale-partial.json").write_text("garbage")
        flag = target.parent / f".{target.name}.migrating"
        flag.touch()

        assert migrate_legacy_backup_dir(target) is True
        _assert_legacy_vacated(legacy, target)
        assert not flag.exists()
        assert not (target / "stale-partial.json").exists()
        assert (target / "sequence.json").read_text() == '{"src": "legacy"}'

    def test_cleans_stale_flag_after_completed_move(self, isolated_home: Path):
        """Flag present + legacy gone = move completed but flag wasn't unlinked.

        Just clean the flag; do NOT touch the (complete) target.
        """
        target = isolated_home / ".local" / "share" / "claude-swap"
        target.mkdir(parents=True)
        (target / "sequence.json").write_text('{"complete": true}')
        flag = target.parent / f".{target.name}.migrating"
        flag.touch()

        assert migrate_legacy_backup_dir(target) is False
        assert not flag.exists()
        # Target is untouched.
        assert (target / "sequence.json").read_text() == '{"complete": true}'

    def test_oserror_is_wrapped(self, isolated_home, monkeypatch):
        """Filesystem errors surface as MigrationError, not raw OSError."""
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir()
        (legacy / "sequence.json").write_text("{}")
        target = isolated_home / ".local" / "share" / "claude-swap"

        def exploding_move(*args, **kwargs):
            raise PermissionError("simulated EACCES")

        monkeypatch.setattr("claude_swap.paths.shutil.move", exploding_move)

        with pytest.raises(MigrationError, match="failed"):
            migrate_legacy_backup_dir(target)
        # Legacy untouched (the real shutil.move was never called).
        assert (legacy / "sequence.json").read_text() == "{}"

    @pytest.mark.skipif(
        os.name == "nt", reason="Windows refuses directory symlinks unprivileged"
    )
    def test_leaves_a_compat_symlink_at_the_vacated_path(self, isolated_home: Path):
        """The old address keeps working after the move.

        A live session shell carries the old path in ``CLAUDE_CONFIG_DIR``,
        and a PyPI-installed copy of cswap sharing this $HOME still resolves
        the legacy root — without the link the first breaks and the second
        silently starts a second store.
        """
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir()
        (legacy / "sequence.json").write_text('{"src": "legacy"}')

        target = isolated_home / ".local" / "share" / "claude-swap"
        assert migrate_legacy_backup_dir(target) is True

        assert legacy.is_symlink()
        assert legacy.resolve() == target.resolve()
        # Readable THROUGH the link, not merely present as one.
        assert (legacy / "sequence.json").read_text() == '{"src": "legacy"}'

    @pytest.mark.skipif(
        os.name == "nt", reason="Windows refuses directory symlinks unprivileged"
    )
    def test_second_run_is_a_noop_through_the_symlink(self, isolated_home: Path):
        """Idempotence rests on the link, so it has to be tested through it.

        Once the link is in place ``legacy.resolve()`` answers the target,
        which is exactly what the ``same_path`` guard reads — and a second
        migration that did anything at all would be moving the store into
        itself.
        """
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir()
        (legacy / "sequence.json").write_text('{"src": "legacy"}')
        target = isolated_home / ".local" / "share" / "claude-swap"
        assert migrate_legacy_backup_dir(target) is True

        assert migrate_legacy_backup_dir(target) is False
        assert (target / "sequence.json").read_text() == '{"src": "legacy"}'
        assert legacy.is_symlink()
        assert not (target.parent / f".{target.name}.migrating").exists()

    def test_unlinkable_legacy_path_does_not_fail_the_migration(
        self, isolated_home: Path, monkeypatch
    ):
        """The data is already home by the time the link is attempted.

        Windows without developer mode and a read-only $HOME both land here;
        neither is a reason to report a failed migration over a store that
        moved cleanly.
        """
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir()
        (legacy / "sequence.json").write_text('{"src": "legacy"}')
        target = isolated_home / ".local" / "share" / "claude-swap"

        def refuse(*args, **kwargs):
            raise OSError("symlink privilege not held")

        monkeypatch.setattr(Path, "symlink_to", refuse)

        assert migrate_legacy_backup_dir(target) is True
        assert (target / "sequence.json").read_text() == '{"src": "legacy"}'
        assert not legacy.exists()

    def test_preserves_file_modes(self, isolated_home: Path):
        if os.name == "nt":
            pytest.skip("POSIX file modes not meaningful on Windows")
        legacy = isolated_home / LEGACY_BACKUP_DIRNAME
        legacy.mkdir(mode=0o700)
        cred = legacy / "credentials" / ".creds-1-user@example.com.enc"
        cred.parent.mkdir(mode=0o700)
        cred.write_text("data")
        os.chmod(cred, 0o600)

        target = isolated_home / ".local" / "share" / "claude-swap"
        assert migrate_legacy_backup_dir(target) is True
        moved = target / "credentials" / ".creds-1-user@example.com.enc"
        assert moved.stat().st_mode & 0o777 == 0o600
        assert (target / "credentials").stat().st_mode & 0o777 == 0o700
