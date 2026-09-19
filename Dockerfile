# Arm64 sibling of the 0xSero pin (lmsysorg/sglang:dev-dsv41, linux/amd64
# digest sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5).
FROM lmsysorg/sglang:dev-dsv41
WORKDIR /opt/dsv41
COPY adapter /opt/dsv41/adapter
RUN g++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC -pthread \
    adapter/row_store.cpp -o adapter/librow_store.so
COPY runtime/flash_mla_sm120.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
# Tolerate the DeepSeek image placeholder token in message text. Upstream raises there,
# which turns any transcript carrying the token into a self-sustaining HTTP 500 loop on
# the Anthropic endpoint. Idempotent, and it fails the build if the base image's
# encoding_dsv41.py no longer matches the anchor.
COPY runtime/patch_encoding_dsv41.py /opt/dsv41/runtime/patch_encoding_dsv41.py
RUN python3 /opt/dsv41/runtime/patch_encoding_dsv41.py
COPY boot.py /opt/dsv41/boot.py
COPY scripts /opt/dsv41/scripts
COPY tests /opt/dsv41/tests
COPY benchmarks /opt/dsv41/benchmarks
RUN PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_thinking_alias.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_max_new_tokens.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_loop_abort.py
ENV PYTHONPATH=/opt/dsv41/adapter \
    MODEL_PATH=/models/DeepSeek-V4.1-Flash \
    STATE_PATH=/state OFFLOAD_MODE=nvme DSV41_CACHE_GIB=16
EXPOSE 8888
# -S: skip site (the adapter's sitecustomize imports the engine, which takes >10 s on a
# busy head and marked the container unhealthy during long prefills); health needs stdlib only.
HEALTHCHECK --interval=30s --timeout=30s --start-period=30m --retries=3 \
    CMD ["python3", "-S", "/opt/dsv41/boot.py", "health"]
ENTRYPOINT ["python3", "-u", "/opt/dsv41/boot.py"]
CMD ["run"]
