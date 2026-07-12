from ultralytics import YOLO
import numpy as np
import tempfile
import os

class DentalYOLODetector:

    def __init__(self):
        print("Loading Dental YOLO...")

        self.model = YOLO.from_pretrained(
            "liodon-ai/dental-panoramic-detector"
        )

    def detect(self, pil_image):

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as tmp:

            pil_image.save(tmp.name)

            results = self.model.predict(
                source=tmp.name,
                verbose=False
            )

        os.unlink(tmp.name)

        output=[]

        r=results[0]

        if r.boxes is None:
            return output

        names=self.model.names

        for box in r.boxes:

            cls=int(box.cls)

            conf=float(box.conf)

            xyxy=box.xyxy.cpu().numpy()[0].tolist()

            output.append({

                "class_id":cls,

                "class_name":names[cls],

                "confidence":conf,

                "bbox":xyxy

            })

        return output
