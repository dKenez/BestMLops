# MLOPS: ML is a dish best served containerized

Welcome to my MLOps introductory course! This guide will walk you through setting up a basic MLOps workflow using modern Python tooling and best practices. As can be deciphered from the course title, the goal of this short excursion into MLops-land is to serve our model through an API which is deployed in a Docker container. To achieve this, we will use:

- [uv](https://github.com/astral-sh/uv) for fast Python dependency management
- [ruff](https://github.com/astral-sh/ruff) for linting
- [mypy](https://mypy.readthedocs.io/en/stable/getting_started.html) for type checking
- [Hugging Face](https://huggingface.co/) for datasets, models, and training
- [FastAPI](https://fastapi.tiangolo.com/) for serving your ML model
- [Docker](https://www.docker.com/) for containerization
- GitHub Actions for CI/CD

This guide is designed to be beginner-friendly, so you can follow along even if you're new to MLOps. By the end of this course, you'll have a solid foundation to build upon for your own projects.

While you are expected to follow along this guide, you can also find the complete code in this GitHub repository (using the the Github Tags, you can find the code for each step of the course):

---

## 1. Project Initialization

### 1.1. Install Prerequisites

To get started, ensure you have the following tools installed. Platform specific installation instructions can be found by following the respective links.

- [uv](https://docs.astral.sh/uv/getting-started/installation/):
    - There are a few ways to install `uv`, but the recommended way is to use the `pip` tool if you already have that installed:
    ```sh
    pip install uv
    ```
- [Docker](https://docs.docker.com/get-docker/)
- [Git](https://git-scm.com/)

### 1.2. Set Up Project Structure

We will start off by creating a new directory for our MLOps project and initializing it with `uv`. This will set up a virtual environment and the basic project structure.

```sh
mkdir BestMLops
cd BestMLops

uv init --name BestMLops --package --python 3.11
```

You now have a new directory called `BestMLops` with a virtual environment set up. `uv` also created the initial project structure, including a `pyproject.toml` file for managing dependencies and configurations.

Going forward will write your code in the `src/bestmlops/` directory, you should also see a basic `__init__.py` file there. Try running it with `uv run bestmlops`!

Take a moment to explore the generated files:
- `pyproject.toml`: This file is used to manage your project dependencies and configurations.
- `src/bestmlops/__init__.py`: This is the main entry point for
your MLOps project. You can add your code here or create additional modules as needed.
- '.venv/': This directory contains the virtual environment created by `uv`. It isolates your project dependencies from the global Python environment.
- `.python-verision`: This file specifies the Python version used in your project. It is automatically created by `uv` based on the version you specified during initialization.

---

## 2. Dependency Management

### 2.1. Adding our Dev Dependencies

To start off, lets make sure we write clean code and have the necessary libraries for our MLOps workflow. We will install the following dev packages:

```sh
uv add ruff mypy --group dev
```

Take a moment to explore the `pyproject.toml` file. You should see the newly added dependencies appear under the `dev` dependency group.

- `ruff`: A fast linter for Python code. (for making your code beautiful and clean)
- `mypy`: A static type checker for Python. (for making sure your code is type-safe)

You can now use these tool to ensure your code is clean and type-safe. Run the following commands semi regularly during development to check your code quality:

```sh
uv run ruff check --fix
uv run ruff format
uv run mypy .
```

### 2.2. Pre-commit Hooks

It is annoying to remember to run the above commands every time you make a change to your code. To automate this, we use pre-commit hooks. Pre-commit hooks are scripts that run automatically before you commit your code, allowing you to enforce code quality checks.

```sh
uv add pre-commit --group dev
uv run pre-commit sample-config >> .pre-commit-config.yaml
```

In the root of your project, you should now see a `.pre-commit-config.yaml` file. This file defines the pre-commit hooks that will run automatically when you commit your code. At this point, it should look like this:

```yaml
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v3.2.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
```
Now, let's add the `ruff` and `mypy` as pre-commit hooks to this file. Open `.pre-commit-config.yaml` and add the following lines under the `repos` section:

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  # Ruff version.
  rev: v0.12.1
  hooks:
    # Run the linter.
    - id: ruff-check
      args: [ --fix ]
    # Run the formatter.
    - id: ruff-format

- repo: local
  hooks:
    - id: mypy
      name: mypy
      entry: bash -c 'uv run mypy "$@"' --
      language: system
      types: [python]
      verbose: true
```

Check our the [pre-commit website](https://pre-commit.com/hooks.html) for more hooks you can add to your project. You can add hooks for security checks, secret scanning, commit linting, and more. The above configuration is just a starting point.

Now that you have defined your pre-commit hooks, you need to install them. Run the following command:

```sh
uv run pre-commit install
```

This will ensure that the pre-commit hooks are installed and will run automatically when you commit your code. In case you want to run the pre-commit hooks manually, you can do so by running (remember to `git add .` first to stage your changes):


```sh
uv run pre-commit run --all-files
```


---

## 3. Code Quality: Ruff and Ty

### 3.1. Ruff (Linting)

- Add to your `pyproject.toml`:
    ```toml
    [tool.ruff]
    line-length = 88
    target-version = "py39"
    ```
- Run:
    ```sh
    ruff check .
    ```

### 3.2. Ty (Type Checking)

- Run:
    ```sh
    ty
    ```

---

## 4. Model Training with Hugging Face

### 4.1. Use Datasets

- Example:
    ```python
    from datasets import load_dataset

    dataset = load_dataset("imdb")
    print(dataset)
    ```

### 4.2. Train a Model

- Use Hugging Face Transformers or your preferred ML library.
- Save your trained model to the `models/` directory.

---

## 5. Serving with FastAPI

- Create `main.py`:
    ```python
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    # You can extend this with endpoints for inference
    ```
- Run locally:
    ```sh
    uvicorn main:app --reload
    ```

---

## 6. Dockerize the Application

- Create a `Dockerfile`:
    ```dockerfile
    FROM python:3.9-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
    ```
- Build and run:
    ```sh
    docker build -t mlops-app .
    docker run -p 80:80 mlops-app
    ```

---

## 7. CI/CD with GitHub Actions

- Create `.github/workflows/ci.yml`:
    ```yaml
    name: CI

    on: [push, pull_request]

    jobs:
      build:
        runs-on: ubuntu-latest
        steps:
        - uses: actions/checkout@v3
        - uses: actions/setup-python@v5
          with:
            python-version: '3.9'
        - name: Install UV
          run: pip install uv
        - name: Install dependencies
          run: uv pip install -r requirements.txt
        - name: Lint
          run: ruff check .
        - name: Type Check
          run: ty
        - name: Test
          run: pytest
    ```

---

## 8. Next Steps & Resources

- Extend FastAPI with prediction endpoints.
- Automate model retraining and deployment.
- Explore Hugging Face Spaces for deployment.
- Integrate monitoring/logging (e.g., Prometheus, Grafana).

**References:**
- [UV Docs](https://github.com/astral-sh/uv)
- [Ruff Docs](https://docs.astral.sh/ruff/)
- [Ty Docs](https://github.com/tiangolo/ty)
- [Hugging Face](https://huggingface.co/docs)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Docker Docs](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)

Happy MLOps-ing!
