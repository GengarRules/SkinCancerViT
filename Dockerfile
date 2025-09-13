# This stage installs build tools, uv, and builds the Python virtual environment.
FROM python:3.10.17-slim AS builder

# Install system dependencies required for building Python packages (like git)
RUN apt-get update && apt-get install -y git

# Install uv - a fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

# Set the working directory
WORKDIR /app

# Create a virtual environment to keep dependencies isolated
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install dependencies into the virtual environment using uv
# This is faster than pip and uses the lock file for reproducible builds.
RUN uv sync --locked --no-cache

# This stage creates the final, lean image for production.
FROM python:3.10.17-slim

# Install runtime-only system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from the builder stage so it's available at runtime
COPY --from=builder /usr/local/bin/uv /usr/local/bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    GRADIO_SERVER_NAME="0.0.0.0"

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
# This contains all the installed dependencies but not the build tools.
COPY --from=builder /opt/venv /opt/venv

# Copy your application code into the image
# Using .dockerignore prevents unnecessary files from being copied.
COPY . .

# Expose the port the application runs on
EXPOSE 7860

# Define the entrypoint and default command for the container
ENTRYPOINT ["/app/docker_entrypoint.sh"]
CMD ["app"]