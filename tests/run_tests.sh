#!/bin/bash
# Test runner script for Mimic Master

set -e

echo "Running Mimic Master test suite..."

# Run tests with coverage
echo "Running tests..."
uv run --with pytest-cov pytest tests/ \
  --cov=mimic_master \
  --cov-report=term-missing \
  --cov-report=html \
  -v

# Exit with test status
echo "Test complete!"
