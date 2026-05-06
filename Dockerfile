FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY main.py /app/main.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh \
    && uv sync --frozen --no-dev

CMD ["/app/docker-entrypoint.sh"]
