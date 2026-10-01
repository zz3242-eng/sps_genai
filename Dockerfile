FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv>=0.5.0

# Copy dependency files and install runtime dependencies (spaCy model is a pinned dependency)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application code
COPY app ./app

EXPOSE 8000

CMD [".venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
