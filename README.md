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

###

This course would not have been possible without the Course that kickstarted my MLOps journey, do check it out here: [DTU-MLOps](https://skaftenicki.github.io/dtu_mlops/). It covers a wide range of topics and provides a solid foundation for building MLOps pipelines. A lot more is covered there than was possible in this short course.

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
- `.python-version`: This file specifies the Python version used in your project. It is automatically created by `uv` based on the version you specified during initialization.

### 1.3. Define your `.gitignore`

During development we create and generate a lot of files that we do not want to commit to our Git repository. Git makes it easy to ignore these files by using a `.gitignore` file. This file tells Git which files and directories to ignore when committing changes.

You can write your own `.gitignore` file, but this is a task that many programmers have already come across, and as such there is a more or less standard `.gitignore` file that you can use for Python projects. You can find it [here](https://github.com/github/gitignore/blob/main/Python.gitignore). This repository contains a collection of `.gitignore` files for various programming languages and frameworks, which you might find useful in the future.

For now, create a `.gitignore` file in the root of your project and copy the contents of the Python `.gitignore` file from the link above into it.


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

#### Exercise
Write some code in the `src/bestmlops/__init__.py` file and try to run the `ruff` and `mypy` manually. Also try committing your code using `git` and see the pre-commit hooks in action.

*As you progress through this course, don't forget to regularly commit your code!*

#### TIP
If you are getting tired of haveing to type `uv run` before every command, you can activate the virtual environment created by `uv` by running the following command:

```sh
source .venv/bin/activate
```
This makes all the scripts you have added available in your terminal. Try running `ruff`, `mypy`, and `pre-commit` commands now.

I will henceforth assume you have the virtual environment activated, so the commands I will provide will not include the `uv run` prefix where not needed.

---

## 3. Let's write some code!

### 3.1. Our First Model

Naturally to do MLOps, we need a model to work with. Due to time constraints, we will not go deep into training models, instead we will use a pre-trained model from Hugging Face, a popular platform for sharing all things machine learning. If you don't need a bespoke model, chances are you can find a pre-trained model on Hugging Face that suits your future ML needs.

The model we will use is based on the SigLIP architecture and is trained on the MNIST dataset for handwritten digit classification. You can find the model [here](https://huggingface.co/prithivMLmods/Mnist-Digits-SigLIP2).


- Create a new file `src/bestmlops/model.py` and add the following code:

```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

# Load model and processor
model_name = "prithivMLmods/Mnist-Digits-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def classify_digit(image):
    """Predicts the digit in the given handwritten digit image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    labels = {
        "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
        "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}

    return predictions
```

This code downloads the pre-trained model and processor from Hugging Face, and defines a function `classify_digit` that takes an image as input and returns the predicted digit probabilities.

To test this locally, we can use the `gradio` library to create a simple web interface for our model.

- Create a new file `src/bestmlops/local_deploy.py` and add the following code:

```python
import gradio as gr

from bestmlops.model import classify_digit

# Create Gradio interface
iface = gr.Interface(
    fn=classify_digit,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="MNIST Digit Classification 🔢",
    description="Upload a handwritten digit image (0-9) to recognize it using MNIST-Digits-SigLIP2.",
)

def deploy_model():
    """
    Function to deploy the model.
    This function is called when the Gradio app is launched.
    """
    iface.launch()


# Launch the app
if __name__ == "__main__":
    deploy_model()
```

### 3.2. Install the required libraries

Now that we have the model code, lets install the required libraries to run it. We will use `uv` to add the necessary dependencies:

```sh
uv add transformers torch pillow gradio
```

You might notice, that this time we are not using `--group dev` flag. This is because these libraries are required to run the model, and we want them to be available in the production environment as well. Later on when we install the dependencies for the project during deployment, we have the option of only the production dependencies (those that are not in the `dev` group). This is a really powerful feature of `uv`.


### 3.3. Run the model locally

We can now run the model locally using Gradio. Of course, we can use `uv` to run the `local_deploy.py` file directly, but since we might want to make it obvious to anyone looking at our python project that this is one of the scripts we intend for them to use, lets define it as an entry point in our `pyproject.toml` file. Open the `pyproject.toml` file and add the following lines under the `[project.scripts]` section:

```toml
local_deploy = "bestmlops.local_deploy:deploy_model"
```

On the left is the script name that we will use to run the model, and on the right is the function that will be called when we run the script. You can see that we are qualifying the function with the module name(`bestmlops.local_deploy`) , and the function name ( `deploy_model`), separated by a colon.

You can now run the model locally using the following command:

```sh
uv run local_deploy
```

Once `gradio` starts, it will provide you with a local URL where you can upload images of handwritten digits (0-9) and see the model's predictions. Look for the link in the terminal output, it should look something like this:

```
* Running on local URL:  http://127.0.0.1:7860
```

Try experimenting with different handwritten digit images to see how well the model performs. You could even try to see if it works with photos of handwritten digits you take with your phone!



---

## 4. API Crafting

We finally have a working model! While we can interact with it locally using Gradio, it is a rigid interface that lacks some functionality we expect from an API. To make our model more accessible, we will create a REST API for our model using [FastAPI](https://fastapi.tiangolo.com/#installation), a modern web framework for building APIs with Python.

### 4.1. Getting started with FastAPI

First of all, as you might have guessed, we need to install FastAPI.Let's add these dependencies to our project:

```sh
uv add "fastapi[standard]"
```

- Create a new file `src/bestmlops/api.py` and add the following code:

```python
from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
```

This is a basic FastAPI application with two endpoints. The first endpoint (`/`) returns a simple JSON response, and the second endpoint (`/items/{item_id}`) takes an `item_id` as a path parameter and an optional query parameter `q`.

Run the FastAPI application locally with:

```sh
fastapi dev src/bestmlops/api.py --port 8070
```
Experiment a bit with the API by visiting it on the port you defined. Try removing the `--port` flag, which port does FastAPI use by default?


#### TIP
navigate to the `/docs` page on the fastapi app.

As you can see here, the functions defined in `api.py` are automatically created as endpoints and documented by FastAPI. This is one of the many features that make FastAPI a great choice for building APIs. Right now you have two `GET` endpoints, the next step is creating an endpoint where you can upload images to and get the model's predictions back. This will require a `POST` endpoint.

### 4.2. Creating the model inference endpoint

#### Exercise
Now that you have a feel for how FastAPI works, create an endpoint for the model. For inspiration, take a look at the `local_deploy.py` file we created earlier. You can use the `classify_digit` function from the `model.py` file to process the image input and return the predictions.

#### TIP
To be able to upload images to the API, add these lines to your `api.py` file to get started:

```python
# import necessary libraries
from io import BytesIO
import numpy as np
from fastapi import UploadFile
from PIL import Image

# You existing FastAPI code...
# ...
# ...

@app.post("/infer/")
async def infer(file: UploadFile):
    # Read the file contents
    contents = await file.read()
    # Convert it to a numpy array
    image = np.array(Image.open(BytesIO(contents)))

    # Add your own logic here to classify the digit!
    # Don't forget to return the predictions!


# you will probably want to be able to debug
# your FastAPI app locally, so add this at the end of your file:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8071)
```

<details>
  <summary>SPOILER: solution here</summary>

```python
from io import BytesIO
from typing import Union

import numpy as np
from fastapi import FastAPI, UploadFile
from PIL import Image

from bestmlops.model import classify_digit

app = FastAPI()

# you can get rid of the two original endpoints if you want
@app.get("/")
def read_root():
    return {"Hello": "World"}

# you can get rid of the two original endpoints if you want
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/infer/")
async def infer(file: UploadFile):
    contents = await file.read()
    image = np.array(Image.open(BytesIO(contents)))

    predictions = classify_digit(image)
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8071)

```
</details>

## 5. Dockerizing the API

We have reached a point where most of our business logic has been implmented and working. This next step will talk about how to deploy our API and use it in production.

As is abundantly obvious from the title of this course, containerization was bound to make an appearance. We will use Docker to containerize our FastAPI application, making it easy to deploy and run in any environment.

You should already have `Docker` installed on your machine. If not, please follow the [installation instructions](https://docs.docker.com/get-docker/) for your platform.

### 5.1. The Dockerfile

Docker, being a containerization technology, revolves around the concept of containers. A container is a lightweight, standalone, machine that  that includes everything needed to run a piece of software, including the code, runtime, libraries, and dependencies.

To create our own Docker container, we need to define a `Dockerfile`. This file contains declarative instructions for Docker to build our container image.

To get started, create a new file named `Dockerfile` in the root of your project directory. This file will contain the instructions for building our FastAPI application container.

`uv` was built with Docker in mind, and provides a guide for containerizing your `uv` application. You can find that documentation [here](https://docs.astral.sh/uv/guides/integration/docker/).


```dockerfile
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
```

- Build the image:
```sh
docker build -f Dockerfile -t bestmlops:latest .

# You can see all your images by running:
docker images
```
- Run the container:
```sh
docker run -p 8070:8000 --name bestmlops bestmlops:latest
```

There are numerous options for optimizing your Dockerfile, and subsequently running the resulting container. It is worth perusing the [Docker documentation](https://docs.docker.com/) to learn more about best practices for building and running Docker containers.

You are now running your FastAPI application in a Docker container! This lets you quickly deploy on any machine that has docker installed, or even in the cloud. Should you want to, you can also deploy 10 instances of this container on a single machine, and have them all run independently of each other. This is one of the main benefits of using Docker for deploying applications.

Try it out:

- Build and run:
```sh
docker run -d -p 8071:8000 --name bestmlops_copy1 bestmlops:latest
docker run -d -p 8072:8000 --name bestmlops_copy2 bestmlops:latest
docker run -d -p 8073:8000 --name bestmlops_copy3 bestmlops:latest
```

The `-d` flag runs the container in detached mode, so it doesn't block your terminal. You can see all the running docker instances by running:

```sh
docker ps
```

Or run the following to see stopped containers as well:
```sh
docker ps -a
```


---

## 6. CI/CD with GitHub Actions

GitHub Actions is a powerful tools that allows you to run automated tasks on predefined triggers. In our case we want to automatically build our Docker image whenever we push changes to the main branch of our repository. This way, we can ensure that our application is always up-to-date and ready for deployment.

Workflows are a great way to perform linting, tests, or any number of tasks on your codebase. In our case, we will create a workflow that builds and pushes our Docker image to the GitHub Container Registry (GHCR) whenever we create a new release.

- Create `.github/workflows/release.yml`:
```yaml
name: Build Docker Image

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to GHCR
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push
      run: |
            REPO=$(echo "${GITHUB_REPOSITORY}" | tr '[:upper:]' '[:lower:]')
            IMAGE=ghcr.io/$REPO:${{ github.event.release.tag_name }}
            docker build -t $IMAGE .
            docker push $IMAGE

```

Once you push this commit. You should see the newly defined workflow on the project repository under the "Actions" tab.

In order for this to work properly, you navigate to your repositories settings page, and under the Actions > General find the "Workflow permissions" section. Make sure that the "Read and write permissions" option is selected. This will allow the workflow to push the Docker image to the GitHub Container Registry.

Test the new workflow by creating a new release on GitHub. This will trigger the workflow and build your Docker image, which will then be pushed to the GitHub Container Registry.

Once the workflow finished running, you will find the container image under packages.

---

## 7. Creating a frontend

While this course is focused on MLOps, it is worth creating a simple frontend to interact with our API. This will give you an overview of how these different component work together in a real-world application.

There is a small update that needs to be made to the `api.py` file to allow CORS (Cross-Origin Resource Sharing) requests from the frontend. This is necessary because the frontend will be running on a different domain (or port) than the API.

To allow CORS requests, we will use the `fastapi.middleware.cors` module. Add the following lines to your `api.py` file:

```python
from fastapi.middleware.cors import CORSMiddleware
# the rest of your imports...

app = FastAPI() # <-- this already exists in your file

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (in real life you should specify the frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Create a file `web/index.html` and paste in the content of [this file](https://github.com/dKenez/BestMLops/blob/master/web/index.html). This is a simple HTML page that allows you to draw a digit on a canvas and submit it to the API for classification. The results are displayed as a bar chart.

You can then run a simple HTTP server to serve the static files. You can use Python's built-in HTTP server for this:

```sh
cd web
python3 -m http.server 8060
```


## 8. Next Steps & Resources

Congratulations! You have completed the MLOps introductory course. You have learned how to set up a basic MLOps workflow using modern Python tooling and best practices. You have also learned how to containerize your FastAPI application and deploy it using GitHub Actions.

To continue your MLOps journey, here are some next steps you can take:

- Explore some of the more advance functionality of `uv`, `ruff`, `mypy`, and `pre-commit`.
- Search for a different model on Hugging Face and try to deploy it using the same workflow.
- Extend FastAPI with prediction endpoints, add authentication, and more.
- Automate model retraining and deployment in GitHub Actions.
- Integrate monitoring/logging (e.g., Prometheus, Grafana).

**References:**
- [UV Docs](https://github.com/astral-sh/uv)
- [Ruff Docs](https://docs.astral.sh/ruff/)
- [Hugging Face](https://huggingface.co/docs)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Docker Docs](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)

**Inspiration**

This course was greatly inspired by the awesome DTU-MLOps course by SkafteNicki on Github: [DTU-MLOps](https://skaftenicki.github.io/dtu_mlops/). If you are looking for a more comprehensive course on MLOps, I highly recommend checking it out. It covers a wide range of topics and provides a solid foundation for building MLOps pipelines.

A lot more is covered there than was possible in this short course.

**Thank you for taking this course!** I hope you found it helpful and informative. If you have any questions or feedback, feel free to reach out to me on GitHub or via email.

Happy MLOps-ing!
