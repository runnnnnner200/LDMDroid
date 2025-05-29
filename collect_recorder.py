import json
from enum import Enum
from os import path

from PIL.Image import Image
from loguru import logger


class CollectRecorderType(Enum):
    FindListItem = 1
    SpecificCRUD = 2
    SelectAction = 3
    EvaluateAction = 4
    DetermineCrudType = 5
    DetermineCrudStage = 6
    DetermineReturnToDataList = 7
    CheckForDME = 8


class CollectRecorder:
    output_dir: str | None = None
    counter = 0

    @classmethod
    def init(cls, output_dir: str):
        cls.output_dir = output_dir

        logger.add(
            sink=path.join(output_dir, "log.txt"),
            format="{time:HH:mm:ss} {level} {message}",
        )

    def __init__(self, recorder_type: CollectRecorderType):
        CollectRecorder.counter += 1
        self.index = CollectRecorder.counter
        self.output_dir = CollectRecorder.output_dir
        self.img_counter = 0
        self.recorder_type = recorder_type
        self.data = {"type": recorder_type.name}

    def record_image(self, image: Image) -> str:
        self.img_counter += 1
        file_name = f"{self.index:04}-{self.img_counter}.jpg"
        img_path = path.abspath(path.join(self.output_dir, file_name))
        image.save(img_path)
        return file_name

    def save(self):
        with open(path.join(self.output_dir, f"{self.index:04}.json"), "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
