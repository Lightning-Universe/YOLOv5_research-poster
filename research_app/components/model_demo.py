import logging

import gradio as gr
import torch
from PIL import Image
from lightning.components.serve import ServeGradio
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)


# credits to Ahsen Khaliq (@akhaliq) for building amazing demo with YoloV5
# here is the original work - https://huggingface.co/spaces/pytorch/YOLOv5/blob/main/app.py
class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    inputs = gr.inputs.Image(type='pil', label="Original Image")
    outputs = gr.outputs.Image(type="pil", label="Output Image")

    def __init__(self):
        super(ServeGradio, self).__init__(parallel=True)
        self._model = None
        self.enable_queue = True

    def build_model(self):
        # Images
        torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg',
            'zidane.jpg')
        torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg',
            'bus.jpg')

        # Model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # force_reload=True to update
        return model

    def yolo(self, im, size=640):
        g = (size / max(im.size))  # gain
        im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize

        results = self.model(im)  # inference
        results.render()  # updates results.imgs with boxes and labels
        return Image.fromarray(results.imgs[0])

    def predict(self, image) -> str:
        return self.yolo(image)


if __name__ == "__main__":
    demo = ModelDemo()
    demo.run()
