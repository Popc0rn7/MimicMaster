FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY mimic_master/ ./mimic_master/
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "mimic_master.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
