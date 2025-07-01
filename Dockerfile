# Let's start with a base image supplied by Astral.sh (creators of uv)
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
# Some optimizations for uv
# https://docs.astral.sh/uv/guides/integration/docker/#optimizations
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

# set our working directory to /app
WORKDIR /app

# copy our local src directory to /app/src in the container
ADD src src

# uv requires a README.md file to present
# so we just create an empty one
RUN touch README.md

# Install dependencies
# we are using our local cache to speed up the build process
# and bind mount the uv.lock and pyproject.toml files
# this means that those files are not copied into the container
# but are used directly from the host machine
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --frozen

# Add the newly created virtual environment to the PATH
# so that we can use
ENV PATH="/app/.venv/bin:$PATH"

# Let the container know that port 8000 is exposed
# this is the default port for FastAPI
EXPOSE 8000

# Set the entrypoint to run the FastAPI application
# on container startup
ENTRYPOINT ["fastapi", "run", "/app/src/bestmlops/api.py"]
