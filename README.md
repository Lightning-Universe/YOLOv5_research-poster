# ⚡️ Object Detection with YoloV5 🔬

[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai)
![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

This app is a research poster demo of YoloV5 Object Detection model by Ultralytics. It showcases a notebook, a blog, and
a model demo where you can upload photos to get a bounding box visualized output image.

To create a research poster for your work please
use [Lightning Research Template app](https://github.com/Lightning-AI/lightning-template-research-app).

## Getting started

To create a Research Poster you can install this app via the [Lightning CLI](https://lightning.ai/lightning-docs/) or
[use the template](https://docs.github.com/en/articles/creating-a-repository-from-a-template) from GitHub and
manually install the app as mentioned below.

### Installation

#### With Lightning CLI

`lightning install app lightning/object-detector`

### Example

```python
# update app.py at the root of the repo
import lightning as L

app = L.LightningApp(
    ResearchApp(
        poster_dir="resources",
        paper="https://arxiv.org/pdf/2103.00020.pdf",
        blog="https://openai.com/blog/clip/",
        training_log_url="https://wandb.ai/aniketmaurya/herbarium-2022/runs/2dvwrme5",
        github="https://github.com/mlfoundations/open_clip",
        notebook_path="resources/Interacting_with_CLIP.ipynb",
        launch_jupyter_lab=False,
        launch_gradio=True,
        tab_order=[
            "Poster",
            "Blog",
            "Paper",
            "Notebook",
            "Training Logs",
            "Model Demo",
        ],
    )
)
```

## FAQs

1. How to pull from the latest template
   code? [Answer](https://stackoverflow.com/questions/56577184/github-pull-changes-from-a-template-repository)
