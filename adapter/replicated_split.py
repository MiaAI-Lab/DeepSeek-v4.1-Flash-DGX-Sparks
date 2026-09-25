"""Column-split of chosen ReplicatedLinear layers over the TP group, bit-identical or not at all.

DSV41_REPLICATED_SPLIT=wqkv_a (comma list of prefix suffixes). A ReplicatedLinear makes every TP
rank stream the whole weight for the same output. Here each rank runs the SAME quantized linear on
its slice of 128-row weight tiles (tiles spread as evenly as possible, e.g. 4/4/3/3 of 14) and the
columns are all-gathered (RoCE at small M, a byte copy).

Uneven splits: every rank's GEMM is as wide as the widest slice. A narrower rank runs its GEMM on
a window of that width over the real weight that contains its own tiles (for wqkv_a, 14 tiles:
rank 2 rows 1024..1535, rank 3 rows 1280..1791; views, no copy), so its output is already the
gather's padded width: no pad fill + copy kernel before the all-gather (they made ranks 2/3 arrive
last at 94 % of the wqkv_a gathers). The extra columns are real (a neighbour's) and are dropped
after the gather. DSV41_REPLICATED_SPLIT_WINDOW=0 restores the exact-slice GEMM + pad.

Exactness is checked, not assumed: at the first eager call with M <= DSV41_REPLICATED_SPLIT_MAX_M,
each rank runs the stock layer and its slice on the same random inputs and compares the columns bit
for bit; the ranks agree over the TP group (MIN) and either all use the split for that layer or
all keep the stock path. Graph capture only ever sees the decided path.

Compact gather (DSV41_SPLIT_COMPACT_GATHER, default 0 = the gather + reorder above, unchanged).
The gather above moves world x width columns per row (the widest slice for every rank) and then
reorders them: the TP group's dim-0 gather, a movedim copy into [M, world * width], and a cat that
drops the padding / neighbour columns (wqkv_a: 1792 of 2048). With the gate on:
  - roce: if the call would go to RoCEnante (pynccl enabled, i.e. CUDA graph warm-up / capture /
    replay, and DSV41_ROCE_GATHER admits the size), each rank sends only its own columns (widths
    512/512/384/384 = 64/64/48/48 16-byte packs, rank 3 reading its view at column 128 of the
    window output) and the kernel writes them at columns 0/512/1024/1408 (packs 0/64/128/176, row
    stride 224 packs) of the [M, 1792] output: no pad, no movedim, no cat (runtime/b12x
    RoceOneshotAllReduce.all_gather(columns=...));
  - minimal (every other group or size, and =minimal): the dim-0 gather into [world, M, width]
    (views only) and one cat of the own-column views.
The first eager call of each variant per layer and row count M runs it AND the old path on the same
shard, compares every byte, and agrees over TP (MIN): on any difference (or a rank that cannot
compile / align the RoCE variant) that layer keeps the old path for that M. SGLang's warm-up forwards
before each capture are eager, so every captured M is checked before it is captured; a (variant, M)
not checked yet at a capture takes the old path. The check adds one gather to the first warm-up
forward of each M; the L2 prefetch plan is relearned from the second one.
"""
import copy
import os

import torch

SPEC = [s.strip() for s in os.environ.get("DSV41_REPLICATED_SPLIT", "").split(",") if s.strip()]
MAX_M = int(os.environ.get("DSV41_REPLICATED_SPLIT_MAX_M", "96"))
WINDOW = os.environ.get("DSV41_REPLICATED_SPLIT_WINDOW", "1").strip().lower() not in ("0", "off", "false", "")
_COMPACT = os.environ.get("DSV41_SPLIT_COMPACT_GATHER", "0").strip().lower()
COMPACT = _COMPACT in ("1", "on", "true", "roce", "minimal")
COMPACT_ROCE = _COMPACT in ("1", "on", "true", "roce")
TILE = 128
PACK = 16
_LOG = {"built": 0, "on": 0, "window": 0, "cg": {}}


def _scale_params(layer):
    n, k = layer.weight.shape
    out = []
    for name, p in layer.named_parameters(recurse=False):
        if name == "weight" or p is None or p.dim() < 1:
            continue
        if p.dim() >= 2 and p.shape[0] == n:
            out.append((name, "rows"))
        elif p.dim() == 1 and p.numel() == n * (k // 32):
            out.append((name, "flat"))           # 128x4-swizzled: 128-row tiles are contiguous
    return out


def _ranges(n, world):
    tiles = n // TILE
    base, extra = divmod(tiles, world)
    counts = [base + (1 if r < extra else 0) for r in range(world)]
    starts = [sum(counts[:r]) * TILE for r in range(world)]
    return [(s, s + c * TILE) for s, c in zip(starts, counts)], max(counts) * TILE


def window(n, lo, width):
    """Start of the width-row weight window a rank computes: its own range [lo, ...) when that
    fits, else the last width rows (still tile-aligned, still containing the range)."""
    return min(lo, n - width)


def _proxy(layer, n0, n1):
    proxy = copy.copy(layer)
    proxy._parameters = dict(layer._parameters)
    proxy._buffers = dict(layer._buffers)
    proxy._modules = dict(layer._modules)
    k = layer.weight.shape[1]
    proxy._parameters["weight"] = torch.nn.Parameter(layer.weight.data[n0:n1].contiguous(), requires_grad=False)
    for name, kind in _scale_params(layer):
        p = layer._parameters[name]
        sl = p.data[n0:n1] if kind == "rows" else p.data[n0 * (k // 32):n1 * (k // 32)]
        proxy._parameters[name] = torch.nn.Parameter(sl.contiguous(), requires_grad=False)
    for attr in ("output_size", "output_size_per_partition"):
        if hasattr(layer, attr):
            setattr(proxy, attr, n1 - n0)
    proxy._dsv41_split = None
    return proxy


def padded(local, width):
    """This rank's GEMM output as the gather's [M, width] part (a window GEMM already is)."""
    if local.shape[1] < width:
        local = torch.nn.functional.pad(local, (0, width - local.shape[1]))
    return local.contiguous()


def assemble(gathered, ranges, width, offs):
    """[M, world * width] gathered parts -> the stock [M, n] column order."""
    if all(b - a == width for a, b in ranges):
        return gathered
    return torch.cat([gathered[:, r * width + o:r * width + o + (b - a)]
                      for r, ((a, b), o) in enumerate(zip(ranges, offs))], dim=-1)


def compact_layout(ranges):
    """Per-rank own widths and destination column offsets of the stock [M, n] output."""
    return [b - a for a, b in ranges], [a for a, _ in ranges]


def pack_layout(ranges, element_size=2):
    """compact_layout in 16-byte packs (wqkv_a bf16: [64,64,48,48], [0,64,128,176], row 224),
    or None when a width or offset is not whole packs."""
    widths, dst = compact_layout(ranges)
    if any((v * element_size) % PACK for v in widths + dst):
        return None
    return ([w * element_size // PACK for w in widths], [d * element_size // PACK for d in dst],
            sum(widths) * element_size // PACK)


def assemble_stacked(stacked, ranges, offs):
    """[world, M, width] dim-0 gathered parts (views) -> the stock [M, n] column order, one cat."""
    return torch.cat([stacked[r, :, o:o + (b - a)] for r, ((a, b), o) in enumerate(zip(ranges, offs))], dim=-1)


def _roce_columns(group, local, widths):
    """(pynccl_comm, roce runtime) when this call would go to RoCEnante and the columns gather
    admits it; decided from state every TP rank shares (pynccl on/off, sizes, env), never pointers."""
    if not COMPACT_ROCE:
        return None
    pc = getattr(group, "pynccl_comm", None)
    if pc is None or getattr(pc, "disabled", True):
        return None
    roce = getattr(pc, "roce", None)
    if (roce is None or getattr(roce, "max_gather_bytes", 0) <= 0
            or not hasattr(roce, "should_all_gather_columns")):
        return None
    try:
        return (pc, roce) if roce.should_all_gather_columns(local, widths) else None
    except RuntimeError:
        return None


def _local_ok(local, widths, off, rank):
    """This rank's shard is a 16-byte aligned, unit-stride view the columns kernel can read."""
    es = local.element_size()
    return (local.dim() == 2 and local.stride(1) == 1 and (local.stride(0) * es) % PACK == 0
            and local.data_ptr() % PACK == 0 and (off * es) % PACK == 0 and off >= 0
            and off + widths[rank] <= local.shape[1])


def _run_compact(kind, group, rt, local, ranges, width, offs):
    widths, _ = compact_layout(ranges)
    if kind == "roce":
        pc, roce = rt
        rank = group.rank_in_group
        if not _local_ok(local, widths, offs[rank], rank):
            local = local.contiguous().clone()            # a fresh block is aligned; same bytes
        out = torch.empty((local.shape[0], sum(widths)), dtype=local.dtype, device=local.device)
        stream = pc._resolve_stream() if hasattr(pc, "_resolve_stream") else torch.cuda.current_stream()
        return roce.all_gather(local, dim=-1, out=out, stream=stream, columns=widths,
                               column_offset=offs[rank])
    lp = padded(local, width)
    g = group.all_gather(lp, dim=0)                               # [world * M, width], views after
    return assemble_stacked(g.view(len(ranges), lp.shape[0], width), ranges, offs)


def same_bits(a, b):
    """Every byte equal (NaN payloads included; torch.equal alone would call NaN != NaN)."""
    return (a.shape == b.shape and a.dtype == b.dtype
            and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))))


def _agree(flag, group, dev):
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=dev)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
    return bool(t.item())


def _check_compact(layer, kind, group, rt, local, ranges, width, offs):
    """First eager call of `kind` on this layer: run it and the old path on the same shard, compare
    every bit, agree over TP. Two agreement rounds for roce so that no rank launches the RoCE kernel
    unless every rank can (compiled, aligned): a lone launch would wait out the spin limit and poison
    the runtime."""
    rank = group.rank_in_group
    dev = local.device
    widths, _ = compact_layout(ranges)
    if kind == "roce":
        ok = False
        try:
            rt[1].prepare_columns(widths, local.dtype)
            ok = _local_ok(local, widths, offs[rank], rank)
        except Exception as exc:  # noqa: BLE001 - reported, then agreed below
            print(f"[replicated_split] {layer._dsv41_prefix}: RoCE columns gather unavailable on rank "
                  f"{rank}: {exc!r}", flush=True)
        if not _agree(ok, group, dev):
            return False
    new = _run_compact(kind, group, rt, local, ranges, width, offs)
    old = assemble(group.all_gather(padded(local, width), dim=-1), ranges, width, offs)
    same = same_bits(new, old)
    agreed = _agree(same, group, dev)
    c = _LOG["cg"].setdefault(kind, [0, 0])
    c[0] += 1
    c[1] += int(agreed)
    if rank == 0 and (c[0] <= 1 or not agreed):
        print(f"[replicated_split] {layer._dsv41_prefix}: compact gather ({kind}) "
              f"{'bit-identical, ON' if agreed else 'DIFFERS on some rank, OFF for this layer'} "
              f"(rank 0 equal {same}; {c[1]}/{c[0]} layers on so far)", flush=True)
    return agreed


def gather(layer, group, local, st):
    """The split's all-gather: the compact variant once it passed its first-call check, else the
    old padded gather + reorder."""
    proxy, ranges, width, offs = st
    if COMPACT:
        rt = _roce_columns(group, local, compact_layout(ranges)[0])
        kind = "roce" if rt is not None else "minimal"
        checked = getattr(layer, "_dsv41_cg", None)
        if checked is None:
            checked = layer._dsv41_cg = {}
        key = (kind, int(local.shape[0]))                # per row count: every captured M is checked
        ok = checked.get(key)
        if ok is None and not (local.is_cuda and torch.cuda.is_current_stream_capturing()):
            ok = checked[key] = _check_compact(layer, kind, group, rt, local, ranges, width, offs)
        if ok:
            return _run_compact(kind, group, rt, local, ranges, width, offs)
    return assemble(group.all_gather(padded(local, width), dim=-1), ranges, width, offs)


def _rows(x):
    """M of a plain activation or of a pre-quantized Mxfp8SwizzledInput (data, scales)."""
    if isinstance(x, torch.Tensor):
        return x.shape[0] if x.dim() == 2 else -1
    data = getattr(x, "data", None)
    return data.shape[0] if isinstance(data, torch.Tensor) and data.dim() == 2 else -1


def _slice_exact(layer, orig_forward, proxy, w0, w1, inputs):
    """The proxy's columns equal the stock layer's columns [w0, w1) bit for bit on every input."""
    ok = True
    for x in inputs:
        full = orig_forward(layer, x)[0]
        part = orig_forward(proxy, x)[0]
        ok = ok and part.shape[1] == w1 - w0 and bool(torch.equal(full[:, w0:w1], part))
    return ok


def decide(layer, orig_forward, rank, world, x_real):
    """Rank-local candidates: {"window": state or None, "slice": state or None}, a state being
    (proxy, ranges, width, offsets); offsets[r] is where rank r's own columns start inside its
    gathered part. The caller agrees over TP: windows only if every rank's window is exact."""
    n, k = layer.weight.shape
    out = {"window": None, "slice": None}
    if not (n % TILE == 0 and n // TILE >= world and layer.weight.dtype == torch.float8_e4m3fn):
        return out
    ranges, width = _ranges(n, world)
    n0, n1 = ranges[rank]
    dev = layer.weight.device
    g = torch.Generator(device=dev).manual_seed(4321)
    inputs = [x_real] + [torch.randn((rows, k), generator=g, device=dev, dtype=torch.bfloat16)
                         for rows in (1, 6, 16)]
    proxy = _proxy(layer, n0, n1)
    if _slice_exact(layer, orig_forward, proxy, n0, n1, inputs):
        out["slice"] = (proxy, ranges, width, [0] * world)
    if WINDOW:
        wins = [window(n, a, width) for a, _ in ranges]
        w0 = wins[rank]
        offs = [a - w for (a, _), w in zip(ranges, wins)]
        if w0 == n0 and n1 - n0 == width:
            out["window"] = (out["slice"][0], ranges, width, offs) if out["slice"] is not None else None
        else:
            wproxy = _proxy(layer, w0, w0 + width)
            if _slice_exact(layer, orig_forward, wproxy, w0, w0 + width, inputs):
                out["window"] = (wproxy, ranges, width, offs)
    return out


def _build(layer, orig_forward, group, x_real):
    world, rank = group.world_size, group.rank_in_group
    cand = {"window": None, "slice": None}
    try:
        cand = decide(layer, orig_forward, rank, world, x_real)
    except Exception as exc:
        print(f"[replicated_split] {layer._dsv41_prefix}: slice failed on rank {rank}: {exc!r}", flush=True)
    n, k = layer.weight.shape
    flag = torch.tensor([1 if cand["window"] is not None else 0, 1 if cand["slice"] is not None else 0],
                        dtype=torch.int32, device=layer.weight.device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
    win, sl = (bool(v) for v in flag.tolist())
    state = cand["window"] if win else (cand["slice"] if sl else None)
    agreed = state is not None
    _LOG["built"] += 1
    _LOG["on"] += int(agreed)
    _LOG["window"] += int(win)
    if rank == 0 and (_LOG["built"] <= 2 or not agreed or (WINDOW and not win)):
        how = "window GEMMs (no pad)" if win else "exact slices + pad"
        print(f"[replicated_split] {layer._dsv41_prefix} {n}x{k}: split {'ON, ' + how if agreed else 'OFF'} "
              f"(rank 0 window {cand['window'] is not None}, slice {cand['slice'] is not None}; "
              f"{_LOG['on']}/{_LOG['built']} layers on so far)", flush=True)
    return state if agreed else False


def install(linear_module):
    """sglang.srt.layers.linear: ReplicatedLinear, per-instance by prefix."""
    if not SPEC:
        return
    cls = linear_module.ReplicatedLinear
    orig_init, orig_forward = cls.__init__, cls.forward

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        prefix = kw.get("prefix", "") or ""
        self._dsv41_prefix = prefix
        self._dsv41_split = None if any(prefix.endswith(s) for s in SPEC) else False

    def forward(self, x, *a, **kw):
        st = getattr(self, "_dsv41_split", False)
        m = _rows(x) if st is not False else -1
        if st is False or a or kw or m <= 0 or m > MAX_M:
            return orig_forward(self, x, *a, **kw)
        from sglang.srt.distributed import get_tp_group
        group = get_tp_group()
        if group.world_size == 1:
            return orig_forward(self, x)
        if st is None:
            if torch.cuda.is_current_stream_capturing():
                return orig_forward(self, x)
            st = self._dsv41_split = _build(self, orig_forward, group, x)
            if st is False:
                return orig_forward(self, x)
        return gather(self, group, orig_forward(st[0], x)[0], st), None   # no pad when windowed

    cls.__init__ = __init__
    cls.forward = forward
    print(f"[replicated_split] armed for {SPEC} (rows <= {MAX_M}; compact gather "
          f"{'off' if not COMPACT else ('roce + minimal' if COMPACT_ROCE else 'minimal')})", flush=True)
