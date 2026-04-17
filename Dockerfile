# --- Flink Builder (Python 3.10) ---
FROM python:3.10-slim-bookworm AS flink-builder

# Use cache mounts to speed up apt and pip installations
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-17-jdk-headless \
    librdkafka-dev

RUN ln -s /usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture) /usr/lib/jvm/java-17-active
ENV JAVA_HOME=/usr/lib/jvm/java-17-active
WORKDIR /app

COPY build-constraints.txt Makefile ./

# Cache pip downloads even across cold builds
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /app/flink_venv && \
    /app/flink_venv/bin/pip install --upgrade "pip<24.1" "setuptools<70" wheel && \
    /app/flink_venv/bin/pip install apache-flink>=1.19.0 confluent-kafka -c build-constraints.txt


# --- Main Builder (Python 3.11) ---
FROM python:3.11-slim-bookworm AS main-builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential

WORKDIR /app
COPY requirements.txt build-constraints.txt Makefile ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /app/main_venv && \
    /app/main_venv/bin/pip install --upgrade pip && \
    /app/main_venv/bin/pip install -r requirements.txt -c build-constraints.txt


# --- Final Runtime Stage ---
FROM python:3.11-slim-bookworm

# Environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    JAVA_HOME=/usr/lib/jvm/java-17-active

# Install runtime dependencies (OpenJRE, librdkafka)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    librdkafka1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture) /usr/lib/jvm/java-17-active
WORKDIR /app

# 1. Copy application code first for development mount support
COPY . .

# 2. Copy environments and Runtimes LAST (Safe Overwrite Strategy)
COPY --from=flink-builder /usr/local/bin/python3.10 /usr/local/bin/python3.10
COPY --from=flink-builder /usr/local/lib/libpython3.10.so.1.0 /usr/local/lib/
COPY --from=flink-builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=flink-builder /app/flink_venv /app/flink_venv
COPY --from=main-builder /app/main_venv /app/main_venv

RUN ldconfig && chmod +x entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
CMD ["dashboard"]
