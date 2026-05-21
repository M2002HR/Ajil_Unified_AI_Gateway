FROM python:3.11-slim

WORKDIR /app
COPY unified_gateway/requirements.txt /app/unified_gateway/requirements.txt

# Optional build-time networking/proxy/index settings (passed via docker-compose build args).
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG ALL_PROXY=""
ARG NO_PROXY=""
ARG PIP_INDEX_URL=""
ARG PIP_EXTRA_INDEX_URL=""

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

RUN set -eux; \
    if [ -n "${PIP_INDEX_URL}" ]; then pip config set global.index-url "${PIP_INDEX_URL}"; fi; \
    if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then pip config set global.extra-index-url "${PIP_EXTRA_INDEX_URL}"; fi; \
    HTTP_PROXY="${HTTP_PROXY}" HTTPS_PROXY="${HTTPS_PROXY}" ALL_PROXY="${ALL_PROXY}" NO_PROXY="${NO_PROXY}" \
    pip install --upgrade --no-cache-dir pip setuptools wheel; \
    HTTP_PROXY="${HTTP_PROXY}" HTTPS_PROXY="${HTTPS_PROXY}" ALL_PROXY="${ALL_PROXY}" NO_PROXY="${NO_PROXY}" \
    pip install --no-cache-dir -r /app/unified_gateway/requirements.txt

COPY . /app

EXPOSE 8080
CMD ["uvicorn", "unified_gateway.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
