#!/bin/bash

set -e

echo "Env vars:"
env | grep -v -E "SECRET|KEY"

if [ "${1}" = "train" ]; then
    echo "Running training..."
    uv run python -m skincancer_vit.train
elif [ "${1}" = "test" ]; then
    echo "Running testing..."
    uv run python -m skincancer_vit.test
elif [ "${1}" = "app" ]; then
    echo "Running Gradio App..."
    uv run python -m skincancer_vit.gradio_app
elif [ "${1}" = "develop" ]; then
    echo "Running Gradio App (development mode)..."
    uv run gradio skincancer_vit/gradio_app.py
else
    exec "$@"
fi
