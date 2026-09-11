#!/usr/bin/env python3
"""SSH helper for the worker: key first, then WORKER_PASS from .env / env."""
from __future__ import annotations

import argparse
import os
import sys

import pexpect


def load_dotenv(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            v = os.path.expandvars(v)
            os.environ.setdefault(k.strip(), v)


def env_file_from_argv(argv: list[str]) -> str:
    """Return the --env-file value from argv (both "--env-file X" and
    "--env-file=X" forms) so the file can be loaded before argument
    defaults are evaluated. Empty string when absent."""
    for i, a in enumerate(argv):
        if a == "--env-file" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--env-file="):
            return a.split("=", 1)[1]
    return ""


def ssh_cmd(
    host: str,
    user: str,
    remote: str,
    password: str | None,
    identity: str | None,
    timeout: int,
) -> int:
    dest = f"{user}@{host}"
    identity = os.path.expandvars(os.path.expanduser(identity or ""))
    base = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
    ]
    # Prefer key when present
    if identity and os.path.isfile(os.path.expanduser(identity)):
        key = os.path.expanduser(identity)
        cmd = base + [
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-i",
            key,
            dest,
            remote,
        ]
        # timeout=None (or <=0): wait forever — needed for docker pull / rsync.
        child = pexpect.spawn(
            cmd[0], cmd[1:], timeout=(None if timeout <= 0 else timeout), encoding="utf-8"
        )
        # Drain output until EOF; reset idle timer on each chunk so long pulls
        # with sparse progress lines don't die at a fixed wall clock.
        buf: list[str] = []
        while True:
            i = child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=(None if timeout <= 0 else timeout))
            if child.before:
                sys.stdout.write(child.before)
                sys.stdout.flush()
                buf.append(child.before)
            if i == 0:
                break
            # TIMEOUT with no EOF: still alive but quiet — keep waiting
            continue
        child.close()
        if child.exitstatus == 0:
            return 0
        # Key auth ran but remote command failed — don't fall through to password.
        if child.exitstatus is not None:
            return child.exitstatus

    if not password:
        print(
            "SSH key auth failed and WORKER_PASS is unset/empty.",
            file=sys.stderr,
        )
        return 255

    cmd = base + [
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "NumberOfPasswordPrompts=1",
        dest,
        remote,
    ]
    to = None if timeout <= 0 else timeout
    child = pexpect.spawn(cmd[0], cmd[1:], timeout=to, encoding="utf-8")
    i = child.expect(
        [r"(?i)password:", r"(?i)permission denied", pexpect.EOF, pexpect.TIMEOUT]
    )
    if i != 0:
        print(
            "SSH password prompt not received "
            f"(got index={i}). Check WORKER_USER/WORKER_HOST and that "
            "password auth is enabled on the worker.",
            file=sys.stderr,
        )
        sys.stderr.write(child.before or "")
        child.close()
        return 255
    child.sendline(password)
    while True:
        j = child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=to)
        if child.before:
            sys.stdout.write(child.before)
            sys.stdout.flush()
        if j == 0:
            break
    child.close()
    if child.exitstatus not in (0, None):
        print(
            f"SSH to {dest} failed (exit {child.exitstatus}). "
            "Update WORKER_PASS in .env or install the shared SSH key on the worker.",
            file=sys.stderr,
        )
        return child.exitstatus or 255
    return 0


def main() -> int:
    # Load the env file BEFORE the parser captures os.environ-based defaults,
    # so SSH_IDENTITY / WORKER_HOST / WORKER_USER / REMOTE_TIMEOUT from .env
    # take effect (previously they were only honored if exported in the shell).
    load_dotenv(env_file_from_argv(sys.argv))

    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default="")
    ap.add_argument("--host", default=os.environ.get("WORKER_HOST", "10.0.0.2"))
    ap.add_argument("--user", default=os.environ.get("WORKER_USER") or os.environ.get("USER", "spark"))
    ap.add_argument(
        "--identity",
        default=os.environ.get("SSH_IDENTITY", "~/.ssh/id_ed25519_shared"),
    )
    # Seconds of idle with no output before abort. 0 = wait forever.
    # Default 600: short commands finish fast; docker pull needs a high value
    # (start.sh passes --timeout for long ops).
    ap.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("REMOTE_TIMEOUT", "600")),
    )
    ap.add_argument("remote")
    args = ap.parse_args()
    password = os.environ.get("WORKER_PASS") or None
    return ssh_cmd(
        args.host,
        args.user,
        args.remote,
        password,
        args.identity,
        args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
