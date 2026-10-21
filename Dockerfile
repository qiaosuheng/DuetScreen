ARG DEVEL_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ARG RUNTIME_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

FROM ${DEVEL_IMAGE} AS extensions
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 MAX_JOBS=2 NVCC_THREADS=2
WORKDIR /build
COPY requirements.txt /build/requirements.txt
RUN python -m pip install -r /build/requirements.txt
RUN python -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheelhouse \
    causal-conv1d==1.5.3.post1 mamba-ssm==2.2.6.post3

FROM ${RUNTIME_IMAGE} AS runtime
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/duetscreen-hf TRITON_CACHE_DIR=/tmp/duetscreen-triton \
    XDG_CACHE_HOME=/tmp/duetscreen-cache
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/duetscreen
COPY requirements.txt /opt/duetscreen/requirements.txt
COPY --from=extensions /wheelhouse /tmp/wheelhouse
RUN python -m pip install -r /opt/duetscreen/requirements.txt \
 && python -m pip install --no-deps /tmp/wheelhouse/*.whl \
 && rm -rf /tmp/wheelhouse
COPY pyproject.toml README.md LICENSE THIRD_PARTY.md ./
COPY duetscreen ./duetscreen
COPY configs ./configs
COPY docs ./docs
RUN python -m pip install --no-deps --no-build-isolation . \
 && python -c "import torch, transformers, mamba_ssm, causal_conv1d, duetscreen.model; assert torch.__version__.startswith('2.8.0'); assert torch.version.cuda == '12.8'" \
 && useradd --create-home --uid 10001 duetscreen \
 && mkdir -p /opt/duetscreen/runs /opt/duetscreen/out \
 && chown -R duetscreen:duetscreen /opt/duetscreen
USER 10001:10001
ENTRYPOINT ["python", "-m", "duetscreen.train"]
