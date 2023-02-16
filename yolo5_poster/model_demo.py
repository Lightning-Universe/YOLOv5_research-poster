import logging
from typing import List

import gradio as gr
import torch
from lightning import BuildConfig
from PIL import Image
from lightning.app.components.serve import ServeGradio

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]")

logger = logging.getLogger(__name__)


class ModelBuildConfig(BuildConfig):
    def build_commands(self) -> List[str]:
        return [
            "pip uninstall -y opencv-python",
            "pip uninstall -y opencv-python-headless",
            "pip install opencv-python-headless==4.5.5.64",
        ]


# This code is adapted from Ultralytics docs - https://docs.ultralytics.com/tutorials/pytorch-hub/
class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    inputs = gr.inputs.Image(type="pil", label="Upload Image for Object Detection")
    outputs = gr.outputs.Image(type="pil", label="Output Image")
    # examples = [["zidane.jpg"], [["bus.jpg"]]]
    enable_queue = True

    def __init__(self):
        super().__init__(parallel=True, cloud_build_config=ModelBuildConfig())

    def build_model(self):
        for f in ["zidane.jpg", "bus.jpg"]:
            torch.hub.download_url_to_file("https://ultralytics.com/images/" + f, f)
        model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        return model

    def yolo(self, image):
        results = self.model(image, size=640)  # includes NMS
        return Image.fromarray(results.render()[0])

    def predict(self, image) -> str:
        return self.yolo(image)


if __name__ == "__main__":
    demo = ModelDemo()
    demo.run()
