"""CPU checks of adapter/moe_b12x_next.py's host-side logic (no GPU, no engine, no b12x_next build).

- hook order: install_method -> install_runner and install_runner -> install_method both import
  b12x_next through ``_b12x()`` (pinned-commit and drift checks), and a wrong pin or a drifted
  b12x source refuses in both orders (fake ``b12x_next`` / ``sglang`` packages on disk).
- a layer that reaches the fused function without a conversion refuses instead of running the
  stock FlashInfer kernel on checkpoint-layout weights.
- chunk selection: the chunk is the largest prepared capacity one launch can run; with
  CHUNKED_PREFILL_SIZE=65536 (ladder top 65536 above the E=384 launch limit of 61,717 tokens) the
  65536 plan is dropped, the chunk is 4096, and ``forward`` never hands a plan more rows than its
  capacity for any M up to 140k; ``_run`` refuses an oversized batch.
- CUDA_GRAPH_MAX_BS_DECODE above the adapter's graph list names the uncovered row counts.
- DSV41_MOE_B12X_NEXT_DETERMINISTIC=1 races nothing. With DSV41_MOE_B12X_NEXT_DET_TRITON=0 it moves
  the <= 8 row plan to the internal route planner; by default (DET_TRITON=1) the pinned plan keeps the
  Triton route planner, and ``_b12x()`` falls back to internal (with a WARNING) when the runtime lacks
  scripts/b12x_next-det-triton-planner.patch (probed on fake patched / stock / absent sources).

usage: python3 tests/test_moe_b12x_next_cpu.py   (needs torch; runs each scenario in a subprocess)
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "adapter")

FAKE_FILES = {
    "b12x_next/__init__.py": "",
    "b12x_next/preparation/__init__.py": "class PreparationSession:\n    pass\n",
    "b12x_next/moe/__init__.py": "",
    "b12x_next/moe/fused_moe/__init__.py": "def run(binding):\n    pass\n",
    "b12x_next/moe/fused_moe/_preparation.py": textwrap.dedent("""\
        class _FusedMoeState:
            def bind(self, **kwargs):
                kwargs["experts"] = self.experts._impl
                b = self.scratch.bind(**kwargs)
                return b, dict(compact_launches=self.compact_launches), "unit_scale_contract", "_w4a16_launches"
        """),
    "b12x_next/moe/fused_moe/api.py": textwrap.dedent("""\
        from dataclasses import replace
        def bind(state, plan, **kwargs):
            return replace(state.bind(**kwargs), plan=plan)
        """),
    "sglang/__init__.py": "",
    "sglang/srt/__init__.py": "",
    "sglang/srt/distributed/__init__.py": "def get_tp_group():\n    return None\n",
    "sglang/srt/distributed/device_communicators/__init__.py": "",
    "sglang/srt/distributed/device_communicators/pynccl_allocator.py": "def use_symmetric_memory(g):\n    pass\n",
    "sglang/srt/layers/__init__.py": "",
    "sglang/srt/layers/dp_attention.py": "def is_allocation_symmetric():\n    return False\n",
    "sglang/srt/layers/quantization/__init__.py": "",
    "sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py": textwrap.dedent("""\
        import torch
        # the stock method dispatches to fused_experts_none_to_flashinfer_mxfp4
        class Mxfp4FlashinferCutlassMoEMethod:
            load_up_proj_weight_first = True
            def create_weights(self, layer, e, h, n, dt, **kw):
                assert n % 128 == 0
                self._fp8.create_weights(layer, e, h, n, dt, fp4_scale_dtype=torch.float8_e8m0fnu, **kw)
            def process_weights_after_loading(self, layer):
                self._fp8.process_weights_after_loading(layer)
                block_scale_interleave(layer)
            def create_moe_runner(self, layer, cfg):
                pass
            def apply(self, layer, d):
                pass
        """),
    "sglang/srt/layers/moe/__init__.py": "",
    "sglang/srt/layers/moe/topk.py": "class TopKOutputChecker:\n    pass\n",
    "sglang/srt/layers/moe/token_dispatcher/__init__.py": "",
    "sglang/srt/layers/moe/token_dispatcher/standard.py": "class StandardCombineInput:\n    pass\n",
    "sglang/srt/layers/moe/moe_runner/__init__.py": "",
    "sglang/srt/layers/moe/moe_runner/base.py": textwrap.dedent("""\
        def fused_experts_none_to_flashinfer_mxfp4(dispatch_output, quant_info, runner_config):
            return "stock"
        class FusedOpPool:
            _fused_funcs = {("none", "flashinfer_mxfp4"): fused_experts_none_to_flashinfer_mxfp4}
        """),
    "sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py": "",
    "sglang/srt/layers/moe/fused_moe_triton/__init__.py": "",
    "sglang/srt/layers/moe/fused_moe_triton/layer.py": textwrap.dedent("""\
        class FusedMoE:
            def _load_w13(self, loaded_weight, shard_dim):
                if getattr(self.quant_method, "load_up_proj_weight_first", False):
                    pass
                shard_size = loaded_weight.shape[shard_dim] // self.moe_tp_size
                start = shard_size
            def _load_w2(self, loaded_weight, shard_dim):
                shard_size = loaded_weight.shape[shard_dim] // self.moe_tp_size
        """),
}

# det-triton probe targets (moe_b12x_next._det_triton_supported reads these two functions' source)
DET_SOURCES = {
    "patched": {
        "b12x_next/moe/fused_moe/_impl.py": textwrap.dedent("""\
            def _dynamic_external_route_plan_supported(*, deterministic_output, dynamic_route_mode):
                return bool(dynamic_route_mode == "grouped")
            """),
        "b12x_next/moe/fused_moe/_tuning.py": textwrap.dedent("""\
            def _compact_w4a8_query(query):
                return True
            def validate_moe_decode_config(config, query):
                return _compact_w4a8_query(query)
            """),
    },
    "stock": {
        "b12x_next/moe/fused_moe/_impl.py": textwrap.dedent("""\
            def _dynamic_external_route_plan_supported(*, deterministic_output, dynamic_route_mode):
                return bool(dynamic_route_mode == "grouped"
                            and not deterministic_output)
            """),
        "b12x_next/moe/fused_moe/_tuning.py": textwrap.dedent("""\
            def _compact_w4a8_query(query):
                return True
            def validate_moe_decode_config(config, query):
                return _compact_w4a8_query(query) and not query.deterministic_output
            """),
    },
}


def make_tree(root, commit, drift=False, det=None):
    for rel, text in {**FAKE_FILES, **DET_SOURCES.get(det, {})}.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if drift and rel.endswith("_preparation.py"):
            text = text.replace("unit_scale_contract", "unit_scale")
        with open(path, "w") as f:
            f.write(text)
    with open(os.path.join(root, "b12x_next", "SOURCE_COMMIT"), "w") as f:
        f.write(commit + "\n")


def run(code, env=None, path=None):
    """Runs ``code`` in a fresh interpreter with the adapter (and a fake tree) importable; returns stdout."""
    e = dict(os.environ)
    for k in list(e):
        # PYTHONPATH: the image's adapter dir carries sitecustomize.py, whose import hooks would
        # install the adapter on the fake modules' import and take the hook order out of our hands
        if k == "PYTHONPATH" or k.startswith(("DSV41_", "CUDA_GRAPH_MAX_BS_DECODE", "CHUNKED_PREFILL_SIZE",
                                               "DSPARK_", "B12X_")):
            del e[k]
    e.update({"DSV41_MOE_B12X_NEXT": "1", "STATE_PATH": tempfile.gettempdir()}, **(env or {}))
    pre = f"import sys; sys.path[:0] = {[p for p in (path, ADAPTER) if p]!r}\n"
    r = subprocess.run([sys.executable, "-c", pre + textwrap.dedent(code)], env=e, capture_output=True, text=True,
                       timeout=300)
    if r.returncode != 0:
        raise AssertionError(f"scenario failed:\n{code}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr[-3000:]}")
    return r.stdout


HOOKS = """
import moe_b12x_next as ad
import sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe as mq
import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as fic
steps = {"method": lambda: ad.install_method(mq), "runner": lambda: ad.install_runner(fic)}
out = []
for name in ORDER:
    try:
        steps[name]()
        out.append([name, "ok", sorted(ad._B), sorted(ad._ENG)])
    except RuntimeError as exc:
        out.append([name, "refused", str(exc)[:200]])
        break
print("RESULT " + __import__("json").dumps(out))
"""


def hooks(tree, order):
    out = run(f"ORDER = {order!r}\n" + HOOKS, path=tree)
    return json.loads(out.split("RESULT ", 1)[1])


def check_hook_order():
    pinned = run("import moe_b12x_next as ad; print(ad.PINNED_COMMIT)").strip()
    with tempfile.TemporaryDirectory() as good, tempfile.TemporaryDirectory() as bad, \
            tempfile.TemporaryDirectory() as drift:
        make_tree(good, pinned)
        make_tree(bad, "0" * 40)
        make_tree(drift, pinned, drift=True)
        for order in (["method", "runner"], ["runner", "method"]):
            res = hooks(good, order)
            assert [r[1] for r in res] == ["ok", "ok"], res
            for r in res:                       # b12x modules present from the first hook on
                assert {"fm", "prep", "pkg"} <= set(r[2]), (order, r)
            assert {"get_tp_group", "TopKOutputChecker"} <= set(res[-1][3]), res
            assert not {"fm", "prep"} & set(res[-1][3]) and "get_tp_group" not in res[-1][2], res
            for tree, why in ((bad, "is built from 0000"), (drift, "drifted")):
                res = hooks(tree, order)
                assert res[0][1] == "refused" and why in res[0][2], (order, why, res)
        print("hook order: method->runner and runner->method both run the pin + drift checks; "
              "a wrong pin / drifted source refuses in both orders")


def check_unconverted_layer():
    out = run("""
        import torch, types, moe_b12x_next as ad
        qi = types.SimpleNamespace(w13_weight=torch.zeros(2, 4, dtype=torch.uint8))
        ad._state["orig_fused"] = lambda *a: "stock kernel ran"
        try:
            ad._fused_b12x_next(None, qi, None)
            print("RAN")
        except RuntimeError as exc:
            print("REFUSED", exc)
    """)
    assert out.startswith("REFUSED") and "without a b12x_next conversion" in out, out
    print("unconverted layer: refuses instead of running the stock kernel")


CHUNK = """
import torch, types, moe_b12x_next as ad
LIMIT = 61717                      # _safe_dynamic_token_chunk at E=384, K=5120, N=576, top-6
g = ad._Geometry.__new__(ad._Geometry)
caps = g.capacities()
# dynamic plans launch up to LIMIT tokens; b12x's micro plan (tiny caps) launches its capacity
limits = {c: (c if c <= 8 else LIMIT) for c in caps}
chunk, kept, dropped = ad.select_chunk(limits)
g.caps, g.chunk, g.direct = kept, chunk, True
g.table = [None] + [g._lookup(m) for m in range(1, ad.EXACT_MAX + 1)]
seen, worst = set(), 0
def fake_run(cap, x, ids, w, out, impl):
    global worst
    assert x.shape[0] <= cap and cap in kept, (x.shape[0], cap)
    worst = max(worst, x.shape[0] - cap)
    seen.add(cap)
g._run = fake_run
for m in list(range(1, 5000)) + [8191, 8192, 61717, 61718, 65535, 65536, 65537, 140000]:
    x = torch.empty(m, 0)
    ids = torch.empty(m, 6, dtype=torch.int32)
    w = torch.empty(m, 6, dtype=torch.float32)
    g.forward(None, x, ids, w, x)
# the real _run refuses a batch larger than the plan's capacity before touching the plan
try:
    ad._Geometry._run(g, 4096, torch.empty(4097, 0), None, None, None, None)
    raised = False
except RuntimeError:
    raised = True
# a launch limit between two ladder steps (Fable M2): chunk = the largest capacity under it
c2 = ad.select_chunk({c: (c if c <= 8 else 3000) for c in caps})
try:
    ad.select_chunk({4096: 100})
    none_ok = True
except RuntimeError:
    none_ok = False
print("RESULT " + __import__("json").dumps(dict(ladder=ad.LADDER, caps=caps, chunk=chunk, kept_top=kept[-3:],
      dropped=dropped, raised=raised, c2=[c2[0], c2[2]], none_ok=none_ok, seen_top=max(seen))))
"""


def check_chunk_selection():
    res = json.loads(run(CHUNK, env={"CHUNKED_PREFILL_SIZE": "65536"}).split("RESULT ", 1)[1])
    assert res["ladder"][-1] == 65536 and 65536 in res["caps"], res
    assert res["chunk"] == 4096 and res["dropped"] == [65536] and res["seen_top"] == 4096, res
    assert res["raised"] and res["c2"] == [2048, [4096, 65536]] and not res["none_ok"], res
    base = json.loads(run(CHUNK, env={"CHUNKED_PREFILL_SIZE": "4096"}).split("RESULT ", 1)[1])
    assert base["chunk"] == 4096 and base["dropped"] == [] and base["caps"] == res["caps"][:-1], base
    print(f"chunk selection: CHUNKED_PREFILL_SIZE=65536 -> capacity 65536 dropped (limit 61717), chunk "
          f"{res['chunk']}; every chunk of M in 1..140000 fits its plan; oversized _run refuses; limit 3000 -> "
          f"chunk {res['c2'][0]}; production ladder (4096) unchanged")


GRAPH = """
import moe_b12x_next as ad
tuned = sorted(ad._Geometry.tuned_caps(type("G", (), {"topk": 6})()))
print("RESULT " + __import__("json").dumps(dict(rows=ad.uncovered_rows(), planner=ad.SMALL_PLANNER, bs=ad.GRAPH_BS,
                                                tuned=tuned)))
"""


def graph(env):
    return json.loads(run(GRAPH, env=env).split("RESULT ", 1)[1])


def check_graph_bs_and_planner():
    assert graph({})["rows"] == [] and graph({"CUDA_GRAPH_MAX_BS_DECODE": "16"})["rows"] == []
    r = graph({"CUDA_GRAPH_MAX_BS_DECODE": "32"})
    want = sorted({k * b for k in (5, 6) for b in range(17, 33)})
    assert r["rows"] == want and r["bs"][-1] == 16, r
    r = graph({"CUDA_GRAPH_MAX_BS_DECODE": "32", "DSV41_MOE_B12X_NEXT_GRAPH_BS": ",".join(map(str, range(1, 33)))})
    assert r["rows"] == [], r
    r = graph({"CUDA_GRAPH_MAX_BS_DECODE": "8"})
    assert r["rows"] == [] and r["bs"] == [1, 2, 3, 4, 5, 6, 7, 8], r
    d = graph({})
    assert d["planner"] == "triton" and d["tuned"] == [6 * b for b in d["bs"]], d
    det = graph({"DSV41_MOE_B12X_NEXT_DETERMINISTIC": "1", "DSV41_MOE_B12X_NEXT_DET_TRITON": "0"})
    assert det["planner"] == "internal" and det["tuned"] == [], det
    det = graph({"DSV41_MOE_B12X_NEXT_DETERMINISTIC": "1"})     # DET_TRITON default 1
    assert det["planner"] == "triton" and det["tuned"] == [], det
    assert graph({"DSV41_MOE_B12X_NEXT_SMALL_PLAN": "internal:none:16"})["planner"] == "internal"
    print(f"graph bs: CUDA_GRAPH_MAX_BS_DECODE=32 names {len(want)} uncovered row counts ({want[0]}..{want[-1]}), "
          f"none at 16 or with a covering list; DETERMINISTIC=1: no race, triton planner kept (DET_TRITON=0 -> internal)")


TABLE = """
import moe_b12x_next as ad
print("RESULT " + __import__("json").dumps(dict(
    table=ad.PLAN_TABLE, p6=[ad.table_plan(c, 6) for c in (1, 6, 8, 9, 96, 4096)],
    p3=[ad.table_plan(c, 3) for c in (5, 40, 80)])))
"""


def table(env):
    return json.loads(run(TABLE, env=env).split("RESULT ", 1)[1])


def check_plan_table():
    d = table({})
    assert d["p6"] == [["triton", 48, 16]] * 3 + [None] * 3 and d["p3"] == [["triton", 48, 16], None, None], d
    det = table({"DSV41_MOE_B12X_NEXT_DETERMINISTIC": "1", "DSV41_MOE_B12X_NEXT_DET_TRITON": "0"})
    assert det["p6"][:3] == [["internal", 48, 16]] * 3 and det["p6"][3:] == [None] * 3, det
    det = table({"DSV41_MOE_B12X_NEXT_DETERMINISTIC": "1"})
    assert det["p6"] == d["p6"] and det["p3"] == d["p3"], det
    assert table({"DSV41_MOE_B12X_NEXT_SMALL_PLAN": "internal:none:16"})["p6"][0] == ["internal", None, 16]
    t = table({"DSV41_MOE_B12X_NEXT_DETERMINISTIC": "1", "DSV41_MOE_B12X_NEXT_DET_TRITON": "0",
               "DSV41_MOE_B12X_NEXT_PLAN_TABLE": "1-8=triton:48:16, 40-80@k3=internal:32:16, 96=heur, 9-96=internal:none:16"})
    assert t["p6"] == [["internal", 48, 16]] * 3 + [["internal", None, 16], None, None], t
    assert t["p3"] == [["internal", 48, 16], ["internal", 32, 16], ["internal", 32, 16]], t
    for bad in ("1-8=cutlass:48:16", "8-1=internal:48:16", "x=heur", "1-8=internal:48"):
        try:
            run(TABLE, env={"DSV41_MOE_B12X_NEXT_PLAN_TABLE": bad})
            raise RuntimeError(f"{bad!r} was accepted")
        except AssertionError as exc:
            assert "DSV41_MOE_B12X_NEXT_PLAN_TABLE: bad" in str(exc), (bad, str(exc)[-300:])
    print("plan table: default = <= 8 row pin only (unchanged plans), SMALL_PLAN still honoured, top-k scoping, "
          "first match wins, heur entries, deterministic triton kept (DET_TRITON=0 -> internal), malformed entries "
          "refuse to load")


PROBE = """
import moe_b12x_next as ad
before = [ad.table_plan(c, 6) for c in (1, 8, 9)]
ad._b12x()
print("RESULT " + __import__("json").dumps(dict(before=before, after=[ad.table_plan(c, 6) for c in (1, 8, 9)],
                                                p3=ad.table_plan(5, 3))))
"""


def check_det_triton_probe():
    pinned = run("import moe_b12x_next as ad; print(ad.PINNED_COMMIT)").strip()
    tri, intl = [["triton", 48, 16]] * 2 + [None], [["internal", 48, 16]] * 2 + [None]
    det = {"DSV41_MOE_B12X_NEXT_DETERMINISTIC": "1"}
    for kind, env, want, warn in (("patched", det, tri, False), ("stock", det, intl, True),
                                  (None, det, intl, True), ("stock", {}, tri, False),
                                  ("patched", dict(det, DSV41_MOE_B12X_NEXT_DET_TRITON="0"), intl, False)):
        with tempfile.TemporaryDirectory() as tree:
            make_tree(tree, pinned, det=kind)
            out = run(PROBE, env=env, path=tree)
            res = json.loads(out.split("RESULT ", 1)[1])
            assert res["after"] == want and res["p3"] == want[0], (kind, env, res)
            assert ("lacks the det-triton-planner patch" in out) == warn, (kind, env, out[-500:])
    print("det-triton probe: patched runtime keeps the triton planner under DETERMINISTIC; stock / absent "
          "sources fall back to internal with a WARNING; non-deterministic and DET_TRITON=0 unaffected")


def main():
    check_hook_order()
    check_unconverted_layer()
    check_chunk_selection()
    check_graph_bs_and_planner()
    check_plan_table()
    check_det_triton_probe()
    print("moe_b12x_next CPU OK")


if __name__ == "__main__":
    main()
