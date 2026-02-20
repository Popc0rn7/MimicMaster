#!/bin/bash
# Test runner script for Mimic Master

set -e

echo "Running Mimic Master test suite..."

# Install test dependencies
echo "Installing test dependencies..."
uv add --dev pytest pytest-asyncio pytest-cov

# Run tests with coverage
echo "Running tests..."
uv run pytest tests/ \
  --cov=mimic_master \
  --cov-report=term-missing \
  --cov-report=html \
  -v

# Exit with test status
echo "Test complete!"
