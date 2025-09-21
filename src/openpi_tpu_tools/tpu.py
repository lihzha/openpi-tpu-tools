from __future__ import annotations

import base64
from dataclasses import dataclass
import os
import re
import shlex
from typing import Literal

from .config import TPUEnvConfig
from .ssh import SSHOptions
from .ssh import gcloud_tpu_ssh
from .ssh import gcloud_tpu_ssh_stream
from .ssh import run_streaming
from .ssh import run_with_timeout


def _ts() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


DescribeRC = Literal[0, 1, 2]


def _gcloud_describe_state(project: str, zone: str, name: str, timeout_s: int) -> tuple[DescribeRC, str]:
    proc = run_with_timeout(
        timeout_s,
        int(os.environ.get("SSH_KILL_AFTER", 5)),
        [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            name,
            "--zone",
            zone,
            "--project",
            project,
            "--format",
            "value(state)",
        ],
    )
    if proc.returncode == 0:
        return 0, (proc.stdout.strip() or "UNKNOWN")
    out = (proc.stderr or proc.stdout or "").lower()
    if re.search(r"not\s*found|404", out):
        return 0, "NOT_FOUND"
    if re.search(r"permission_denied|forbidden|403", out):
        return 0, "PERMISSION_DENIED"
    if re.search(r"invalid value for \[--zone\]|argument --zone", out):
        return 2, "INVALID_ZONE"
    return 1, out.strip().splitlines()[-1] if out else "ERROR"


@dataclass
class TPUManager:
    env: TPUEnvConfig
    ssh: SSHOptions = SSHOptions()
    describe_timeout_s: int = int(os.environ.get("DESCRIBE_TIMEOUT", 20))
    sleep_secs: int = int(os.environ.get("SLEEP_SECS", 20))

    def _zone_for(self, version: Literal["v4", "v5", "v6"]) -> str:
        return {
            "v4": self.env.tpu_zone_v4,
            "v5": self.env.tpu_zone_v5,
            "v6": self.env.tpu_zone_v6,
        }[version]

    def _bucket_for(self, version: Literal["v4", "v5", "v6"]) -> str:
        return {
            "v4": self.env.tpu_bucket_v4,
            "v5": self.env.tpu_bucket_v5,
            "v6": self.env.tpu_bucket_v6,
        }[version]

    def describe(self, version: Literal["v4", "v5", "v6"]) -> str:
        rc, state = _gcloud_describe_state(
            self.env.tpu_project, self._zone_for(version), self.env.tpu_name, self.describe_timeout_s
        )
        if rc == 2:
            raise RuntimeError(f"Invalid zone for {version}: {self._zone_for(version)}")
        if rc != 0:
            print(f"{_ts()} - Describe error: {state}")
            return "ERROR"
        return state

    def delete(self, version: Literal["v4", "v5", "v6"]) -> bool:
        zone = self._zone_for(version)
        rc = run_streaming(
            [
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "tpu-vm",
                "delete",
                self.env.tpu_name,
                "--zone",
                zone,
                "--project",
                self.env.tpu_project,
                "--quiet",
            ]
        )
        return rc == 0

    def stop(self, version: Literal["v4", "v5", "v6"]) -> bool:
        zone = self._zone_for(version)
        rc = run_streaming(
            [
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "tpu-vm",
                "stop",
                self.env.tpu_name,
                "--zone",
                zone,
                "--project",
                self.env.tpu_project,
            ]
        )
        return rc == 0

    def start(self, version: Literal["v4", "v5", "v6"]) -> bool:
        zone = self._zone_for(version)
        rc = run_streaming(
            [
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "tpu-vm",
                "start",
                self.env.tpu_name,
                "--zone",
                zone,
                "--project",
                self.env.tpu_project,
            ]
        )
        return rc == 0

    def create(self, version: Literal["v4", "v5", "v6"], *, tpu_num: int, topology: str | None = None) -> bool:
        zone = self._zone_for(version)
        common = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "create",
            self.env.tpu_name,
            "--zone",
            zone,
            "--project",
            self.env.tpu_project,
            "--service-account",
            self.env.tpu_service_account,
            "--spot",
        ]
        if version == "v4":
            if not topology:
                raise ValueError("topology is required for v4")
            args = [*common, "--type", "v4", "--topology", topology, "--version", "tpu-ubuntu2204-base"]
        elif version == "v5":
            accel = {16: "v5litepod-16", 32: "v5litepod-32", 64: "v5litepod-64"}.get(tpu_num)
            if not accel:
                raise ValueError("Unsupported TPU_NUM for v5: expected 16/32/64")
            args = [*common, "--accelerator-type", accel, "--version", "v2-alpha-tpuv5-lite"]
        else:  # v6
            args = [*common, "--accelerator-type", f"v6e-{tpu_num}", "--version", "v2-alpha-tpuv6e"]

        rc = run_streaming(args)
        return rc == 0

    def tmux(self, version: Literal["v4", "v5", "v6"], *, cmd: str, session: str = "tpu") -> bool:
        # Ensure tmux exists and start/send in a session across all workers
        line = f"set -eo pipefail; export PYTHONUNBUFFERED=1; {cmd} 2>&1 | tee -a $LOG"
        remote = (
            "command -v tmux >/dev/null || (sudo apt-get update && sudo apt-get install -y tmux);"
            f"mkdir -p $HOME/{self.env.gh_repo_name}/logs;"
            "TS=$(date +%Y%m%d-%H%M%S);"
            f"LOG=$HOME/{self.env.gh_repo_name}/logs/{session}_$TS.log;"
            f"if ! tmux has-session -t {shlex.quote(session)} 2>/dev/null; then "
            f"    tmux new-session -ds {shlex.quote(session)} -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK -e LOG=$LOG; "
            "else "
            f"    tmux set-environment -t {shlex.quote(session)} LOG $LOG; "
            "fi;"
            # Send the command and execute it
            f"tmux send-keys -t {shlex.quote(session)} {shlex.quote(line)} Enter"
        )
        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.env.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command=remote,
                ssh=self.ssh,
            )
            == 0
        )

    def raw(self, version: Literal["v4", "v5", "v6"], *, cmd: str, worker: str | None = "all") -> int:
        """Run a raw command on TPU worker(s) without tmux.

        Mirrors `v4 "<cmd>"` style helpers from ~/.tpu_funcs.sh.
        """
        return gcloud_tpu_ssh_stream(
            tpu_name=self.env.tpu_name,
            project=self.env.tpu_project,
            zone=self._zone_for(version),
            worker=worker if worker is not None else None,
            command=cmd,
            ssh=self.ssh,
        )

    def attach(self, version: Literal["v4", "v5", "v6"], *, session: str = "tpu", worker: int = 0) -> int:
        # Use exec with `tmux new -As` to attach-or-create without running extra commands afterward
        return gcloud_tpu_ssh_stream(
            tpu_name=self.env.tpu_name,
            project=self.env.tpu_project,
            zone=self._zone_for(version),
            worker=str(worker),
            command=(
                "command -v tmux >/dev/null || (sudo apt-get update && sudo apt-get install -y tmux); "
                f"exec tmux new -As {shlex.quote(session)}"
            ),
            ssh=self.ssh,
            allocate_tty=True,
            no_shell_rc=True,
        )

    def tmux_ls(self, version: Literal["v4", "v5", "v6"]) -> bool:
        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.env.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command="tmux ls || true",
                ssh=self.ssh,
            )
            == 0
        )

    def tail_log(self, version: Literal["v4", "v5", "v6"], *, worker: int = 0) -> int:
        # Prefer tmux's LOG environment for the current session; fallback to newest in logs dir.
        # Use -f to follow like the shell helper's v4_tail.
        session = "tpu"
        cmd = (
            f"SESSION={shlex.quote(session)}; "
            'LOG_FILE="$(tmux show-environment -t "$SESSION" LOG 2>/dev/null | sed -n "s/^LOG=//p")"; '
            '[ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ] && { tail -n 200 -f "$LOG_FILE"; exit $?; }; '
            'LOG_DIR="${LOG_FILE%/*}"; '
            f'[ -n "$LOG_DIR" ] || LOG_DIR=$HOME/{self.env.gh_repo_name}/logs; '
            'test -d "$LOG_DIR" || { echo "[ERROR] Logs dir not found: $LOG_DIR"; exit 1; }; '
            'F="$(ls -1t "$LOG_DIR" | head -n1 || true)"; '
            '[ -n "$F" ] || { echo "[ERROR] No log files in $LOG_DIR"; exit 1; }; '
            'tail -n 200 -f "$LOG_DIR/$F"'
        )
        rc = gcloud_tpu_ssh_stream(
            tpu_name=self.env.tpu_name,
            project=self.env.tpu_project,
            zone=self._zone_for(version),
            worker=str(worker),
            command=cmd,
            ssh=self.ssh,
        )
        return rc

    def tmux_kill_all(self, version: Literal["v4", "v5", "v6"]) -> bool:
        remote = (
            "set -euo pipefail;"
            "if command -v tmux >/dev/null 2>&1; then "
            "tmux ls >/dev/null 2>&1 && tmux kill-server || true; "
            "rm -rf /tmp/tmux-$(id -u) 2>/dev/null || true; fi"
        )
        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.env.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command=remote,
                ssh=self.ssh,
            )
            == 0
        )

    def kill_jax(self, version: Literal["v4", "v5", "v6"]) -> bool:
        remote = (
            "set -euo pipefail;"
            "PIDS=$(pgrep -u $USER -f python || true);"
            "for pid in $PIDS; do "
            "if tr '\\0' '\\n' </proc/$pid/environ 2>/dev/null | grep -qE '(^(JAX_|XLA_|TPU_|LIBTPU))'; then "
            "kill -TERM $pid 2>/dev/null || true; fi; done;"
            "sleep 2;"
            "for pid in $(pgrep -u $USER -f python || true); do "
            "if tr '\\0' '\\n' </proc/$pid/environ 2>/dev/null | grep -qE '(^(JAX_|XLA_|TPU_|LIBTPU))'; then "
            "kill -0 $pid 2>/dev/null && kill -KILL $pid 2>/dev/null || true; fi; done;"
            "pgrep -a -u $USER -f python || true"
        )
        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.env.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command=remote,
                ssh=self.ssh,
            )
            == 0
        )

    def clean_jax_tmp(self, version: Literal["v4", "v5", "v6"]) -> bool:
        remote = (
            'echo "[INFO] Cleaning /tmp…";'
            "find /tmp -maxdepth 1 -user $USER "
            "( -name 'jax*' -o -name '.jax*' -o -name 'pjrt*' -o -name 'xla*' "
            "-o -name 'libtpu*' -o -name 'tpu*' -o -name 'coordination-*' -o -name 'jax-mp-*' ) "
            "-print -exec rm -rf {} + 2>/dev/null || true;"
            'echo "[INFO] Cleaning /dev/shm…";'
            "find /dev/shm -maxdepth 1 -user $USER "
            "( -name 'sem.*' -o -name 'psm_*' -o -name 'jax*' -o -name 'xla*' -o -name 'pjrt*' ) "
            "-print -exec rm -f {} + 2>/dev/null || true"
        )
        return (
            gcloud_tpu_ssh_stream(
                tpu_name=self.env.tpu_name,
                project=self.env.tpu_project,
                zone=self._zone_for(version),
                worker="all",
                command=remote,
                ssh=self.ssh,
            )
            == 0
        )

    def nuke_all(self, version: Literal["v4", "v5", "v6"]) -> bool:
        ok = self.tmux_kill_all(version)
        ok = self.kill_jax(version) and ok
        ok = self.clean_jax_tmp(version) and ok
        return ok

    def list(self, version: Literal["v4", "v5", "v6"]) -> int:
        zone = self._zone_for(version)
        rc = run_streaming(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "list",
                "--zone",
                zone,
            ]
        )
        return rc

    def delete_by_name(self, version: Literal["v4", "v5", "v6"], name: str) -> int:
        zone = self._zone_for(version)
        rc = run_streaming(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "delete",
                name,
                "--project",
                self.env.tpu_project,
                "--zone",
                zone,
            ]
        )
        return rc

    def check_activity(self, version: Literal["v4", "v5", "v6"]) -> bool:
        """Return True if busy, False if idle. Failures => busy (conservative)."""
        probe = r"""
        set -e -o pipefail
        uid=$(id -u)
        for m in /proc/*/maps; do
        [ -r "$m" ] || continue
        if grep -qE 'libtpu|libxla|_xla_extension|libdevice' "$m" 2>/dev/null; then
            pid=${m%/maps}; pid=${pid#/proc/}
            puid=$(awk '/^Uid:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo -1)
            if [ "$puid" = "$uid" ]; then
            echo busy; exit 0
            fi
        fi
        done
        if pgrep -af '(^|/)python([0-9.])?' >/dev/null 2>&1; then echo busy; exit 0; fi
        echo idle
        """
        encoded = base64.b64encode(probe.encode()).decode().replace("\n", "")
        # Use non-streaming so we can parse the result.
        proc = gcloud_tpu_ssh(
            tpu_name=self.env.tpu_name,
            project=self.env.tpu_project,
            zone=self._zone_for(version),
            worker="all",
            command=f"bash -lc 'echo {encoded} | base64 -d | bash -s'",
            ssh=self.ssh,
        )
        if proc.returncode != 0:
            print(f"{_ts()} - SSH probe failed (rc={proc.returncode}); treating as busy.")
            return True
        return bool(re.search(r"(^|\r?\n)busy(\r?\n|$)", proc.stdout))
