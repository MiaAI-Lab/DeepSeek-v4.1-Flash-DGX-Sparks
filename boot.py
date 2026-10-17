"""Pinned checkpoint download, integrity verification, serving and API acceptance.

Spark remap of 0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000:
  4x RTX PRO 6000 TP4/EP4  ->  3x DGX Spark (GB10) TP3/EP3 over NCCL.
Engram tables stay on NVMe (or SSHFS) via the row-store adapter. Do not use
RAM mode on Spark: host RAM is the GPU's unified memory.
"""
import concurrent.futures
import base64
import io
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request

REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
MODEL = Path(os.environ.get('MODEL_PATH', '/models/DeepSeek-V4.1-Flash'))
STATE = Path(os.environ.get('STATE_PATH', '/state'))
PORT = int(os.environ.get('SERVER_PORT', '8888'))
NNODES = int(os.environ.get('NNODES', '1'))
NODE_RANK = int(os.environ.get('NODE_RANK', '0'))
TP_SIZE = int(os.environ.get('TP_SIZE', os.environ.get('TP', '3')))
EP_SIZE = int(os.environ.get('EP_SIZE', str(TP_SIZE)))
DIST_INIT_ADDR = os.environ.get('DIST_INIT_ADDR', '')
SERVED_MODEL_NAME = os.environ.get('SERVED_MODEL_NAME', 'deepseek-v4.1-flash')


def save(name, value):
    STATE.mkdir(parents=True, exist_ok=True)
    temporary = STATE / (name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(STATE / name)


def prepare():
    if os.environ.get('SKIP_PREPARE', '0') == '1':
        assert (MODEL / 'config.json').is_file(), f'missing {MODEL}/config.json'
        print('SKIP_PREPARE=1 — using existing checkpoint', flush=True)
        return
    if NODE_RANK != 0:
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if (MODEL / 'config.json').is_file():
                print(f'Rank {NODE_RANK}: checkpoint visible at {MODEL}', flush=True)
                return
            time.sleep(5)
        raise RuntimeError(f'Rank {NODE_RANK}: {MODEL} never appeared')

    from huggingface_hub import snapshot_download
    MODEL.mkdir(parents=True, exist_ok=True)
    metadata = json.load(urllib.request.urlopen(
        f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=60))
    assert metadata['sha'] == REVISION
    snapshot_download(REPO, revision=REVISION, local_dir=str(MODEL), max_workers=4)
    if os.environ.get('SKIP_VERIFY', '0') == '1':
        print('SKIP_VERIFY=1 — not re-hashing the checkpoint', flush=True)
        return
    receipt = STATE / 'verification.json'
    old = json.loads(receipt.read_text()) if receipt.exists() else {}
    cached = {f['name']: f for f in old.get('files', [])} if old.get('revision') == REVISION else {}

    def verify(entry):
        name = entry['rfilename']
        path = MODEL / name
        assert path.resolve().is_relative_to(MODEL.resolve())
        before = path.stat()
        identity = {'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns, 'inode': before.st_ino}
        assert before.st_size == entry['size'], f'Wrong file size: {name}'
        expected = entry['lfs']['sha256'] if entry.get('lfs') else entry['blobId']
        previous = cached.get(name, {})
        if previous.get('digest') == expected and all(previous.get(k) == v for k, v in identity.items()):
            return previous
        digest = hashlib.sha256() if entry.get('lfs') else hashlib.sha1()
        if not entry.get('lfs'):
            digest.update(f'blob {before.st_size}\0'.encode())
        with path.open('rb') as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        assert digest.hexdigest() == expected, f'Hash mismatch: {name}'
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
        print('Verified', name, flush=True)
        return {'name': name, 'digest': expected, **identity}

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        files = list(pool.map(verify, metadata['siblings']))
    verified = {f['name'] for f in files}
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    assert set(index.values()) <= verified
    save('verification.json', {'revision': REVISION, 'status': 'verified', 'files': files})


def key():
    value = os.environ.get('API_KEY', '').strip()
    if value.lower() in ('', 'none', 'off', 'dummy', '0'):
        return ''
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / 'api-key'
    path.write_text(value)
    path.chmod(0o600)
    return value


def request(path, payload=None, timeout=10):
    headers = {'Content-Type': 'application/json'}
    secret = key()
    if secret:
        headers['Authorization'] = 'Bearer ' + secret
    req = urllib.request.Request(
        f'http://127.0.0.1:{PORT}' + path,
        headers=headers,
        data=None if payload is None else json.dumps(payload).encode())
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else {'status': response.status}


def smoke():
    result = request('/v1/chat/completions', dict(
        model=SERVED_MODEL_NAME, temperature=0, max_tokens=32,
        chat_template_kwargs={'thinking': False},
        messages=[dict(role='user', content='What is 19 + 23? Reply only with the number.')]),
        timeout=300)
    choice = result['choices'][0]
    assert choice['message']['content'].strip() == '42' and choice['finish_reason'] == 'stop', result
    save('smoke.json', result)
    print('Fresh inference passed: 19 + 23 = 42', flush=True)
    if os.environ.get('SMOKE_QUICK', '0') == '1':
        return
    common = dict(model=SERVED_MODEL_NAME, temperature=0, max_tokens=256,
                  chat_template_kwargs={'thinking': False})
    structured = request('/v1/chat/completions', dict(common,
        messages=[dict(role='user', content='Return an object whose answer is the integer 42.')],
        response_format={'type': 'json_schema', 'json_schema': {'name': 'answer', 'strict': True,
            'schema': {'type': 'object', 'properties': {'answer': {'type': 'integer'}},
                      'required': ['answer'], 'additionalProperties': False}}}), timeout=300)
    assert json.loads(structured['choices'][0]['message']['content']) == {'answer': 42}
    save('smoke-structured.json', structured)
    tool = {'type': 'function', 'function': {'name': 'lookup_fixture',
        'description': 'Retrieve a stored test value.',
        'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}},
                      'required': ['key'], 'additionalProperties': False}}}
    messages = [dict(role='user', content='Use lookup_fixture to retrieve the value for key alpha. Do not guess.')]
    called = request('/v1/chat/completions', dict(common, messages=messages, tools=[tool]), timeout=300)
    assistant = called['choices'][0]['message']
    calls = assistant.get('tool_calls') or []
    assert len(calls) == 1 and calls[0]['function']['name'] == 'lookup_fixture', called
    assert json.loads(calls[0]['function']['arguments']) == {'key': 'alpha'}
    messages += [assistant, dict(role='tool', tool_call_id=calls[0]['id'], content='{"value":42}')]
    continued = request('/v1/chat/completions', dict(common, messages=messages, tools=[tool]), timeout=300)
    assert '42' in continued['choices'][0]['message']['content'], continued
    save('smoke-tools.json', {'call': called, 'continuation': continued})
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (3024, 588), 'white')
    draw = ImageDraw.Draw(image)
    draw.ellipse((200, 100, 588, 488), fill='red')
    draw.rectangle((2400, 100, 2788, 488), fill='blue')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    visual = request('/v1/chat/completions', dict(common, messages=[dict(role='user', content=[
        {'type': 'text', 'text': 'Describe the two colored shapes and their left-to-right order. Be concise.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}}
    ])]), timeout=300)
    text = visual['choices'][0]['message']['content'].lower()
    assert all(word in text for word in ('red', 'circle', 'blue', 'square')), visual
    details = (visual.get('usage') or {}).get('prompt_tokens_details') or {}
    if 'image_tokens' in details:
        assert details['image_tokens'] == 1024, visual
    save('smoke-vision.json', visual)
    print('Structured output, tool round trip and full-budget native vision passed.', flush=True)


def warmup():
    """Exercise the kernels a fresh engine has not seen yet, so the first real
    requests do not pay for JIT compiles, autotune buckets or first-touch page
    allocation: prompts of a few sizes (prefill tiers), a decode run, and one
    concurrent batch of MAX_RUNNING_REQUESTS. Failures are logged, not fatal."""
    if os.environ.get('WARMUP', '1') in ('0', 'off', 'false'):
        return
    started = time.monotonic()
    common = dict(model=SERVED_MODEL_NAME, temperature=0,
                  chat_template_kwargs={'thinking': False})
    filler = ('The quick brown fox jumps over the lazy dog near the riverbank '
              'while the sun sets slowly behind the distant hills. ')
    sizes = [int(x) for x in os.environ.get('WARMUP_PROMPT_WORDS', '12,200,900,3600').split(',')]
    for words in sizes:
        prompt = (filler * (words // 20 + 1)) + 'Summarise the text above in one sentence.'
        try:
            request('/v1/chat/completions', dict(common, max_tokens=24,
                    messages=[dict(role='user', content=prompt)]), timeout=600)
        except Exception as exc:  # warm-up must never take the engine down
            print(f'warm-up prompt (~{words} words) failed: {exc}', flush=True)
    batch = max(1, int(os.environ.get('MAX_RUNNING_REQUESTS', '4')))
    prompts = ['Explain how a hash table handles collisions in two sentences.',
               'List the steps of the TCP three-way handshake.',
               'Describe a quiet morning in a small harbour town in two sentences.',
               'Explain in plain words why the sky is blue.'] * ((batch + 3) // 4)
    # Sanity check on the batch: every request of a MAX_RUNNING_REQUESTS batch must
    # come back as Latin prose. On 2026-09-11 a boot (chunk 1024 / pool 750k / ctx 256k)
    # returned symbols and CJK for every request but the last of any greedy batch of
    # 3-4 (NaN target logits at bs >= 3) while single requests were fine, so the
    # arithmetic smoke alone cannot catch it. Loud warning, not fatal.
    with concurrent.futures.ThreadPoolExecutor(batch) as pool:
        outcomes = list(pool.map(lambda p: _try_request(dict(common, max_tokens=48,
                                 messages=[dict(role='user', content=p)])), prompts[:batch]))
    garbled = []
    for prompt, (error, text) in zip(prompts[:batch], outcomes):
        if error:
            print(f'warm-up batch request failed: {error}', flush=True)
        elif not _looks_like_latin_prose(text):
            garbled.append((prompt, text))
    if garbled:
        print(f'WARNING: BATCH OUTPUT SANITY CHECK FAILED: {len(garbled)}/{batch} requests of a '
              f'batch of {batch} (MAX_RUNNING_REQUESTS) did not return Latin prose. '
              'Concurrent requests at this batch size are returning garbage '
              '(see logs/profile-2026-09-10/REPORT.md section 16).', flush=True)
        for prompt, text in garbled:
            print(f'  batch of {batch}: {prompt[:40]!r} -> {text[:80]!r}', flush=True)
    else:
        print(f'Batch output sanity check passed: batch of {batch} returned Latin prose.',
              flush=True)
    print(f'Warm-up done in {time.monotonic() - started:.0f}s '
          f'({len(sizes)} prompt sizes + one batch of {batch})', flush=True)


def _looks_like_latin_prose(text):
    """True when the text reads as Latin-script prose: no CJK / Hangul, and at
    least 70% of the non-space characters are ASCII letters, digits or
    common punctuation. The garbage seen at bs >= 3 was symbol runs and CJK."""
    body = [ch for ch in (text or '') if not ch.isspace()]
    if not body:
        return False
    if any('\u3000' <= ch <= '\u9fff' or '\uac00' <= ch <= '\ud7af' for ch in body):
        return False
    latin = sum(1 for ch in body if ch.isascii() and (ch.isalnum() or ch in '.,;:\'"!?-()'))
    return latin / len(body) >= 0.7


def _try_request(payload):
    """(error, text): error is None on success, text is the assistant content."""
    try:
        result = request('/v1/chat/completions', payload, timeout=600)
        return None, result['choices'][0]['message']['content']
    except Exception as exc:
        return str(exc), ''


def _gpu_lines():
    return subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
        text=True).strip().splitlines()


def serve():
    mode = os.environ.get('OFFLOAD_MODE', 'nvme')
    assert mode in ('nvme', 'ram'), 'OFFLOAD_MODE must be nvme or ram'
    context = int(os.environ.get('CONTEXT_LENGTH', '409600'))
    assert 4096 <= context <= 1048576, 'Context must be 4096 through the model limit 1048576'
    gpu = _gpu_lines()
    assert gpu, gpu
    if mode == 'ram':
        if os.environ.get('ALLOW_RAM_OFFLOAD', '0') != '1':
            raise RuntimeError(
                'RAM Engram offload pins ~189 GiB into host RAM, which on DGX Spark '
                'is the same pool as the GPU. Use OFFLOAD_MODE=nvme (default). '
                'Set ALLOW_RAM_OFFLOAD=1 to override.')
        index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
        shards = {index[f'layers.{layer}.engram.embed.weight'] for layer in (1, 14)}
        needed = sum((MODEL / name).stat().st_size for name in shards) + 24 * 2**30
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(mem['MemAvailable'].split()[0]) * 1024
        assert available >= needed, (
            f'RAM offload needs at least {needed / 2**30:.1f} GiB available; '
            f'found {available / 2**30:.1f}. Use nvme mode.')

    mem_frac = os.environ.get('MEM_FRACTION_STATIC', '0.95')
    chunk = os.environ.get('CHUNKED_PREFILL_SIZE', '2048')
    max_req = os.environ.get('MAX_RUNNING_REQUESTS', '4')
    graph_bs = os.environ.get('CUDA_GRAPH_MAX_BS_DECODE', max_req)
    spec = os.environ.get('SPEC_ALGO', 'DSPARK')
    args = [
        '--model-path', str(MODEL),
        '--served-model-name', SERVED_MODEL_NAME,
        '--trust-remote-code', '--load-format', 'safetensors',
        '--tp', str(TP_SIZE), '--ep-size', str(EP_SIZE),
        '--attention-backend', os.environ.get('ATTENTION_BACKEND', 'dsv4'),
        '--moe-runner-backend', os.environ.get('MOE_RUNNER_BACKEND', 'flashinfer_mxfp4'),
        '--mem-fraction-static', mem_frac,
        '--chunked-prefill-size', chunk,
        '--context-length', str(context),
        '--max-running-requests', max_req,
        '--cuda-graph-max-bs-decode', graph_bs,
        '--random-seed', '0',
        '--enable-decoder-swa-bounded-replay',
        '--enable-cache-report',
        '--tool-call-parser', os.environ.get('TOOL_CALL_PARSER', 'deepseekv41'),
        '--reasoning-parser', os.environ.get('REASONING_PARSER', 'deepseek-v41'),
        '--host', os.environ.get('HOST', '0.0.0.0'),
        '--port', str(PORT),
    ]
    if spec and spec.lower() not in ('off', 'none', '0'):
        args += [
            '--speculative-algorithm', spec,
            '--speculative-dspark-block-size', os.environ.get('DSPARK_BLOCK_SIZE', '5'),
        ]
        # The ragged-verify scheduler is where DSpark's throughput actually comes
        # from, and it needs a profiled SPS cost table: without one the planner
        # falls back to verify-all and the whole thing is a no-op. Build one with
        #   python -m sglang.benchmark.dspark_sps_profiler
        # against a running server, then point DSPARK_SPS_TABLE at it.
        table = os.environ.get('DSPARK_SPS_TABLE', '').strip()
        if table and Path(table).is_file():
            args += ['--speculative-dspark-sps-table-path', table]
            # Fills each step's verify window up to the cuda-graph tier the
            # forward is padded to anyway: free verification at the same cost.
            args += ['--speculative-dspark-align-verify-tokens-to-graph-tier']
        elif table:
            print(f'DSPARK_SPS_TABLE={table} not found; '
                  'staying on the verify-all schedule', flush=True)
        sts = os.environ.get('DSPARK_STS_TABLE', '').strip()
        if sts and Path(sts).is_file():
            args += ['--speculative-dspark-confidence-sts-path', sts]
    if NNODES > 1:
        assert DIST_INIT_ADDR, 'DIST_INIT_ADDR is required when NNODES>1'
        args += [
            '--nnodes', str(NNODES),
            '--node-rank', str(NODE_RANK),
            '--dist-init-addr', DIST_INIT_ADDR,
        ]
    # Pin the KV pool. Left to itself SGLang sizes it from whatever happens to be
    # free at load time, and it varies by ~2x between boots -- on unified memory
    # the bigger pool comes straight out of host RAM and the head starts stalling
    # on reclaim, which costs more throughput than the extra KV ever buys.
    total_tokens = os.environ.get('MAX_TOTAL_TOKENS', '').strip()
    if total_tokens and total_tokens != '0':
        args += ['--max-total-tokens', total_tokens]

    extra = os.environ.get('EXTRA_SGLANG_ARGS', '').strip()
    if extra:
        args += extra.split()

    save('launch.json', {
        'revision': REVISION, 'offload_mode': mode, 'args': args, 'gpus': gpu,
        'nnodes': NNODES, 'node_rank': NODE_RANK, 'tp': TP_SIZE, 'ep': EP_SIZE,
    })
    secret = key()
    env = dict(os.environ, DSV41_SOURCE=str(MODEL))
    # compact ragged-verify only pays off with a profiled cost table; without one
    # it degenerates to the same verify-all schedule as static, for more work.
    if '--speculative-dspark-sps-table-path' in args:
        env.setdefault('SGLANG_RAGGED_VERIFY_MODE', 'compact')
    cmd = [sys.executable, '-m', 'sglang.launch_server', *args]
    if secret:
        cmd += ['--api-key', secret]
    process = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)

    def stop(*_):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def logs():
        for line in process.stdout:
            print(line.replace(secret, '[REDACTED]') if secret else line, end='', flush=True)

    threading.Thread(target=logs, daemon=True).start()
    try:
        if NODE_RANK != 0:
            return process.wait()
        deadline = time.monotonic() + int(os.environ.get('READY_TIMEOUT_S', '3600'))
        while process.poll() is None and time.monotonic() < deadline:
            try:
                request('/health', timeout=3)
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError('Server failed to become healthy; inspect container logs')
        if os.environ.get('SKIP_SMOKE', '0') != '1':
            smoke()
            warmup()
        if secret:
            print(f'Ready: API on port {PORT} (rank 0). Key is in {STATE}/api-key.', flush=True)
        else:
            print(f'Ready: API on port {PORT} (rank 0), no API key.', flush=True)
        return process.wait()
    finally:
        stop()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if command == 'health':
        request('/health')
    elif command == 'smoke':
        smoke()
    elif command == 'prepare':
        prepare()
    elif command == 'run':
        prepare()
        sys.exit(serve())
    else:
        raise SystemExit('Commands: run, prepare, health, smoke')
