#!/usr/bin/env python3
"""Offline launcher tests: no Docker, SSH, sysfs or external model paths."""
import importlib.util
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_sitecustomize():
    spec = importlib.util.spec_from_file_location("dsv41_test_sitecustomize", ROOT / "adapter/sitecustomize.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"DSV41_SOURCE": ""}):
        spec.loader.exec_module(module)
    return module


class FakeLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.should_async_load = lambda weight: True


def test_serial_weight_load_opt_in_and_default():
    sc = load_sitecustomize()
    before = list(sys.meta_path)
    for value, expected in (("1", False), ("0", True), (None, True)):
        module = SimpleNamespace(__name__="sglang.srt.model_loader.utils")
        with patch.dict(os.environ):
            os.environ.pop("DSV41_SERIAL_WEIGHT_LOAD", None)
            if value is not None:
                os.environ["DSV41_SERIAL_WEIGHT_LOAD"] = value
            sc.EngramLoader(FakeLoader()).exec_module(module)
        assert module.should_async_load(object()) is expected
    assert sys.meta_path == before


def test_serial_finder_does_not_intercept_when_disabled():
    sc = load_sitecustomize()
    with patch.dict(os.environ, {"DSV41_SERIAL_WEIGHT_LOAD": "0"}), patch.object(sc.importlib.machinery.PathFinder, "find_spec") as finder:
        assert sc.EngramFinder().find_spec("sglang.srt.model_loader.utils") is None
        finder.assert_not_called()


def test_serial_loader_rejects_incompatible_upstream():
    sc = load_sitecustomize()
    loader = SimpleNamespace(exec_module=lambda module: None)
    with patch.dict(os.environ, {"DSV41_SERIAL_WEIGHT_LOAD": "1"}):
        try:
            sc.EngramLoader(loader).exec_module(SimpleNamespace(__name__="sglang.srt.model_loader.utils"))
        except RuntimeError as exc:
            assert "should_async_load" in str(exc)
        else:
            raise AssertionError("incompatible loader accepted")


def launcher_source():
    source = (ROOT / "start.sh").read_text()
    # Run real defaults and function definitions; omit env sourcing and dispatch.
    # Every filesystem operation in this setup targets the fixture below.
    source = source[source.index('HEAD_IP='):source.index('case "$CMD" in')]
    return "set -euo pipefail\n" + source


def shell(code, directory, settings=None):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "ROOT": str(ROOT)}
    for name in ("MODEL_DIR", "COMMON_MODEL", "SSH_IDENTITY", "NCCL_HOST_DIR", "STATE_DIR", "LOG_DIR", "WORKER_DIR", "WORKER_ENGRAM_DIR", "ENGRAM_DIR"):
        env[name] = str(directory / name)
    Path(env["MODEL_DIR"]).mkdir(exist_ok=True)
    Path(env["MODEL_DIR"], "config.json").write_text("{}")
    env.update(settings or {})
    return subprocess.run(["bash", "-c", launcher_source() + "\n" + code], env=env, cwd=ROOT, text=True, capture_output=True, timeout=20)


def check(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def pairs(args, flag):
    return [args[i + 1] for i, item in enumerate(args[:-1]) if item == flag]


def test_head_worker_env_defaults_and_quoted_opt_in():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        for ring in (False, True):
            settings = {"SERVED_MODEL_NAME": "model name; $(false) 'quoted'", "DSV41_SERIAL_WEIGHT_LOAD": "1" if ring else "0"}
            if ring:
                settings.update(NCCL_SWITCHLESS_RING_ONLY="1", NCCL_MIN_NCHANNELS="4", NCCL_MAX_NCHANNELS="4")
            for rank in (1, 2, 3):
                code = f'''
head=()
docker_common_args head "$HEAD_IP" 3
printf '%s\\0' "${{head[@]}}"
printf 'SPLIT\\0'
capture() {{ printf '%s\\0' "$@"; }}
eval "capture $(worker_env_lines 10.0.0.2 3 {rank})
end-of-command"
'''
                if ring:
                    libdir = d / "NCCL_HOST_DIR"
                    libdir.mkdir(exist_ok=True)
                    (libdir / "libnccl.so.2").write_text("SWITCHLESS_RING_ONLY")
                out = check(shell(code, d, settings)).split("\0")
                split = out.index("SPLIT")
                head, worker = out[:split], out[split + 1:]
                he, we = pairs(head, "-e"), pairs(worker, "-e")
                for value in ("DSPARK_BLOCK_SIZE=3", "SGLANG_DSV41_REASONING_EFFORT=75", "DSV41_MAX_NEW_TOKENS=32768", "DSV41_LOOP_ABORT=1", "SERVED_MODEL_NAME=" + settings["SERVED_MODEL_NAME"]):
                    assert value in he and value in we, value
                assert f"NODE_RANK={rank}" in we
                assert "end-of-command" in worker
                assert ("NCCL_SWITCHLESS_RING_ONLY=1" in he) == ring
                assert ("NCCL_SWITCHLESS_RING_ONLY=1" in we) == ring
                if ring:
                    assert [x for x in he if x.startswith("NCCL_") and x != "NCCL_IB_GID_INDEX=3"] == [x for x in we if x.startswith("NCCL_") and x != "NCCL_IB_GID_INDEX=3"]
                    assert not any(x.startswith("LD_LIBRARY_PATH=") for x in he)
                    assert any("site-packages/nvidia/nccl" in x for x in pairs(head, "-v"))


def test_nccl_library_selection_and_worker_path_quoting():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        libdir = d / "custom nccl ' $(false)"
        libdir.mkdir()
        (libdir / "libnccl.so.2").write_text("SWITCHLESS_RING_ONLY")
        for overlay in ("0", "1"):
            result = shell('''
eval "$(nccl_worker_settings)"
a=(); nccl_mount_args a
printf '%s\\0' "${a[@]}"
''', d, {"NCCL_HOST_DIR": str(libdir), "NCCL_OVERLAY_PIP": overlay})
            args = check(result).split("\0")
            if overlay == "1":
                assert pairs(args, "-v") == [str(libdir / "libnccl.so.2") + ":/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:ro"]
                assert not pairs(args, "-e")
            else:
                assert pairs(args, "-v") == [str(libdir) + ":/nccl:ro"]
                assert pairs(args, "-e") == ["LD_LIBRARY_PATH=/nccl"]
        (libdir / "libnccl.so.2.30.7").write_text("preferred")
        assert check(shell('nccl_library "$NCCL_HOST_DIR"', d, {"NCCL_HOST_DIR": str(libdir)})).strip().endswith("libnccl.so.2.30.7")
        assert shell('a=(); nccl_mount_args a', d, {"NCCL_OVERLAY_PIP": "1"}).returncode != 0
        output = check(shell('NCCL_HOST_DIR="$HOME/custom build"; nccl_worker_settings', d))
        assert 'NCCL_HOST_DIR="$HOME"/custom\\ build' in output
        output = check(shell('NCCL_WORKER_DIR="/custom worker"; nccl_worker_settings', d))
        assert 'NCCL_HOST_DIR=/custom\\ worker' in output


def make_fabric(d):
    for name in ("rocep1s0f0", "rocep1s0f1"):
        port = d / name / "ports/1"
        (port / "gid_attrs/types").mkdir(parents=True)
        (port / "gids").mkdir()
        (port / "state").write_text("4: ACTIVE\n")
        (port / "gid_attrs/types/3").write_text("RoCE v2\n")
        (port / "gids/3").write_text("0000:0000:0000:0000:0000:ffff:0a0a:000a\n")
    return port


def test_ring_probes_all_selected_ports_and_explicit_gid():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        port = make_fabric(d / "fabric")
        command = "ring_gid_index " + shlex.quote(str(d / "fabric"))
        assert check(shell(command, d)).strip() == "3"
        assert shell(command, d, {"NCCL_IB_GID_INDEX": "2"}).returncode != 0
        (port / "state").write_text("2: INIT\n")
        assert shell(command, d).returncode != 0
        (port / "state").write_text("4: ACTIVE\n")
        (port / "gid_attrs/types/3").write_text("RoCE v1\n")
        assert shell(command, d).returncode != 0


def test_ring_config_requires_safe_overlay_and_ring_algorithm():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        settings = dict(NCCL_SWITCHLESS_RING_ONLY="1", TP_SIZE="4", EP_SIZE="4", WORKER_IPS="10.0.0.2 10.0.0.3 10.0.0.4")
        check(shell("nccl_validate_config", d, settings))
        for key, value in (("NCCL_OVERLAY_PIP", "0"), ("NCCL_ALGO", "Tree"), ("NCCL_MIN_NCHANNELS", "33"), ("NCCL_IB_DISABLE", "1"), ("DSV41_SERIAL_WEIGHT_LOAD", "yes")):
            assert shell("nccl_validate_config", d, {**settings, key: value}).returncode != 0


def test_preflight_rejects_stock_nccl_and_missing_pip_path():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        libdir = d / "NCCL_HOST_DIR"
        libdir.mkdir()
        library = libdir / "libnccl.so.2"
        library.write_text("stock")
        settings = {"NCCL_SWITCHLESS_RING_ONLY": "1"}
        code = 'docker() { return 0; }; ring_gid_index() { echo 3; }; nccl_preflight'
        assert shell(code, d, settings).returncode != 0
        library.write_text("SWITCHLESS_RING_ONLY")
        assert check(shell(code, d, settings)).strip() == "3"
        assert shell(code.replace('docker() { return 0;', 'docker() { return 1;'), d, settings).returncode != 0
        assert shell(code.replace('ring_gid_index() { echo 3;', 'ring_gid_index() { return 1;'), d, settings).returncode != 0


def test_serve_fails_before_replacing_containers_on_last_rank_preflight():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        code = '''
cmd_doctor() { :; }; _busy_gpu() { return 1; }; ln() { :; }
docker() { [[ "$1 $2" == 'image inspect' ]] || { echo MUTATION; return 1; }; }
remote_ok_on() { return 1; }
cmd_build() { echo BUILD; }
nccl_preflight() { echo 3; }
remote_on() { echo "CHECK:$1" >&2; [[ "$1" != 10.0.0.4 ]] || return 1; echo 3; }
cmd_share() { echo MUTATION; }
cmd_serve
'''
        result = shell(code, d, dict(NCCL_SWITCHLESS_RING_ONLY="1", TP_SIZE="4", EP_SIZE="4", WORKER_IPS="10.0.0.2 10.0.0.3 10.0.0.4"))
        assert result.returncode != 0
        assert "BUILD" in result.stdout
        assert "CHECK:10.0.0.4" in result.stderr
        assert "MUTATION" not in result.stdout


def test_local_weights_never_trigger_nfs_or_teardown_when_missing():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        result = shell('''
cmd_doctor() { :; }; _busy_gpu() { return 1; }; ln() { :; }
docker() { [[ "$1 $2" == 'image inspect' ]] || { echo MUTATION; return 1; }; }
remote_ok_on() { return 0; }; nfs_worker_has_model() { return 1; }
local_model_has_weights() { return 0; }
cmd_share() { echo MUTATION; }
cmd_serve
''', d, {"NFS_SHARE": "0"})
        assert result.returncode != 0 and "local volume" in result.stderr
        assert "MUTATION" not in result.stdout
        check(shell('nfs_ensure_server() { return 99; }; cmd_share', d, {"NFS_SHARE": "0"}))


def test_worker_launch_script_executes_in_default_and_ring_modes():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        libdir = d / "nccl space ' $(false)"
        libdir.mkdir()
        (libdir / "libnccl.so.2").write_text("SWITCHLESS_RING_ONLY")
        for ring in ("0", "1"):
            result = shell(r'''
cmd_doctor() { :; }; _busy_gpu() { return 1; }; ln() { :; }
preflight_all_nodes() { RING_GID_HEAD=3; RING_WORKER_GIDS=(3 3 3); }
nfs_worker_has_model() { return 0; }; remote_ok_on() { return 0; }
local_model_has_weights() { return 0; }
gid_index_local() { echo 3; }; gid_index_remote() { echo 3; }
push_spec_tables() { :; }
docker_common_args() { exit 0; }
docker() {
  if [[ "$1 $2" == 'run -d' ]]; then printf '%s\0' "$@";
  elif [[ "$1 $2" != 'image inspect' && "$1 $2" != 'volume inspect' && "$1 $2" != 'rm -f' ]]; then return 99; fi
}
test() { [[ "$*" == '-d /dev/infiniband' ]] || builtin test "$@"; }
mkdir() { :; }
remote_on() {
  if [[ "$2" == *'docker run -d'* ]]; then
    bash -c "set -euo pipefail
$(declare -f docker test mkdir)
$2"
  fi
}
cmd_serve
''', d, {"NFS_SHARE": "0", "NCCL_SWITCHLESS_RING_ONLY": ring, "TP_SIZE": "4", "EP_SIZE": "4", "WORKER_IPS": "10.0.0.2 10.0.0.3 10.0.0.4", "NCCL_HOST_DIR": str(libdir), "API_KEY": "fixture ' $(false)", "EXTRA_SGLANG_ARGS": "--name 'quoted ; $(false)'"})
            args = check(result).split("\0")
            for rank in (1, 2, 3):
                assert f"NODE_RANK={rank}" in args
            assert args.count("API_KEY=fixture ' $(false)") == 3
            assert args.count("EXTRA_SGLANG_ARGS=--name 'quoted ; $(false)'") == 3
            assert args.count("NCCL_SWITCHLESS_RING_ONLY=1") == (3 if ring == "1" else 0)
            assert args.count("LD_LIBRARY_PATH=/nccl") == (0 if ring == "1" else 3)


def test_weight_status_probe_checks_contents_without_pull_or_creation():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        volume = d / "weights"
        volume.mkdir()
        code = r'''
remote_on() { eval "${@: -1}"; }
docker() {
  if [[ "$1 $2" == 'volume inspect' ]]; then [[ "$VOLUME_PRESENT" == 1 ]]; return; fi
  [[ "$1" == run ]] || return 90
  [[ " $* " == *' --pull=never '* && " $* " == *' --network none '* ]] || return 91
  while [[ "$1" != -c ]]; do shift; done
  local check="${2//\/m\//$FIXTURE_VOLUME/}"
  bash -c "$check" sh 2
}
nfs_worker_has_model worker
'''
        settings = {"VOLUME_PRESENT": "0", "FIXTURE_VOLUME": str(volume)}
        assert shell(code, d, settings).returncode != 0
        settings["VOLUME_PRESENT"] = "1"
        assert shell(code, d, settings).returncode != 0
        (volume / "config.json").write_text("{}")
        (volume / "model-1-of-2.safetensors").write_text("fixture")
        assert shell(code, d, settings).returncode != 0
        (volume / "model-2-of-2.safetensors").write_text("fixture")
        check(shell(code, d, settings))
        (volume / "model-2-of-2.safetensors").write_text("")
        assert shell(code, d, settings).returncode != 0


def test_local_head_requires_nonempty_complete_shards():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        assert shell("local_model_has_weights", d, {"EXPECTED_SHARDS": "1"}).returncode != 0
        shard = d / "MODEL_DIR/model-1-of-1.safetensors"
        shard.write_text("fixture")
        check(shell("local_model_has_weights", d, {"EXPECTED_SHARDS": "1"}))
        shard.write_text("")
        assert shell("local_model_has_weights", d, {"EXPECTED_SHARDS": "1"}).returncode != 0


def test_nfs_root_and_reused_subdirectory_are_preserved():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
        d = Path(tmp)
        export = d / "export" / "dsv41-native"
        export.mkdir(parents=True)
        (export / "config.json").write_text("{}")
        (export / "model-1-of-1.safetensors").write_text("fixture")
        output = check(shell('''
NFS_REUSE_EXPORT=0; nfs_publish_model; printf 'DEVICE=%s\\n' "$NFS_DEVICE"
NFS_REUSE_EXPORT=1; nfs_publish_model; printf 'DEVICE=%s\\n' "$NFS_DEVICE"
''', d, {"HF_EXPORT_ROOT": str(d / "export"), "EXPECTED_SHARDS": "1"}))
        assert "DEVICE=:/\n" in output
        assert "DEVICE=:/dsv41-native\n" in output


def test_tp4_profile_ring_flags_are_commented_and_base_defaults_preserved():
    text = (ROOT / ".env.tp4.example").read_text()
    values = dict(re.findall(r"(?m)^([A-Z][A-Z0-9_]*)=(.*)$", text))
    assert values["DSV41_SERIAL_WEIGHT_LOAD"] == "1"
    for key in ("NCCL_SWITCHLESS_RING_ONLY", "NCCL_ALGO", "NCCL_MIN_NCHANNELS", "NCCL_P2P_LEVEL"):
        assert key not in values
    assert values["NCCL_MAX_NCHANNELS"] == "8"
    for key, value in {"DSPARK_BLOCK_SIZE": "3", "SGLANG_DSV41_REASONING_EFFORT": "75", "DSV41_MAX_NEW_TOKENS": "32768", "DSV41_LOOP_ABORT": "1"}.items():
        assert values[key] == value


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[OK]   {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(bool(failed))
