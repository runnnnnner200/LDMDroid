import base64
import json
from json_repair import repair_json
from enum import Enum
from io import BytesIO

from PIL.Image import Image
from loguru import logger
from openai import OpenAI as Client


class LLM:
    name: str
    client: Client
    format_model: 'Model'

    class Model(Enum):
        pass

    @staticmethod
    def _image_to_base64(image: Image, max_height: int = 640) -> str:
        old_width, old_height = image.size
        if old_height > max_height:
            new_width = max_height / old_height * old_width
            image = image.resize((int(new_width), int(max_height)))

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"

    @classmethod
    def init(cls, api_key: str, base_url: str, format_model: Model) -> None:
        cls.client = Client(api_key=api_key, base_url=base_url)
        cls.format_model = format_model

    @classmethod
    def chat(cls, model: Model, prompt: str, temperature: float = 0.5, **kwargs) -> str:
        logger.debug(f"Requesting {cls.name}'s {model.value}, T:{temperature}, Prompt:\n{prompt}")
        completion = cls.client.chat.completions.create(
            model=model.value,
            messages=[{"role": "user", "content": prompt}],
            top_p=0.7,
            temperature=temperature,
            **kwargs
        )
        res = completion.choices[0].message.content
        logger.debug(f"Response from {cls.name}'s {model.value}:\n{res}")
        return res

    @classmethod
    def chat_with_image(cls, model: Model, prompt: str, image: Image, temperature: float = 0.5, **kwargs) -> str:
        image_base64 = LLM._image_to_base64(image)
        logger.debug(f"Requesting {cls.name}'s {model.value} with image, T:{temperature}, Prompt:\n{prompt}")
        completion = cls.client.chat.completions.create(
            model=model.value,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]}, ],
            top_p=0.7,
            temperature=temperature,
            **kwargs
        )
        res = completion.choices[0].message.content
        logger.debug(f"Response from {cls.name}'s {model.value}:\n{res}")
        return res

    @classmethod
    def chat_with_image_list(cls, model: Model, prompt: str, image_list: list[Image], temperature: float = 0.5,
                             **kwargs) -> str:
        image_base64_list = [LLM._image_to_base64(image) for image in image_list]
        logger.debug(
            f"Requesting {cls.name}'s {model.value} with {len(image_list)} image, T:{temperature}, Prompt:\n{prompt}")
        completion = cls.client.chat.completions.create(
            model=model.value,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "image_url": {"url": image_base64}} for image_base64 in image_base64_list]
            ]}, ],
            top_p=0.7,
            temperature=temperature,
            **kwargs
        )
        res = completion.choices[0].message.content
        logger.debug(f"Response from {cls.name}'s {model.value}:\n{res}")
        return res

    @classmethod
    def format_to_json(cls, res: str, type_json: dict) -> dict:
        prompt = (
            "Raw Content:\n{}\n---\n"
            "Given the raw content above, please extract the relevant information and present it in the following "
            "JSON format:\n{}\n"
            "Please review the raw content thoroughly and provide a comprehensive answer. "
            "Only output the JSON object that exactly and strictly matches the specified 'JSON format' description. "
            "If multiple JSON objects are found, output the first one. "
        )
        prompt = prompt.format(res, json.dumps(type_json))
        formatted_res = cls.chat(cls.format_model, prompt, temperature=0.0)
        try:
            out = repair_json(formatted_res, return_objects=True, ensure_ascii=False)
            for key in type_json:
                assert key in out  # all keys should be in the output
            return out
        except:
            raise Exception("Formatting llm response failed")


class Zhipu(LLM):
    name: str = "Zhipu"

    class Model(Enum):
        GLM_4_PLUS = "glm-4-plus"
        GLM_4_0520 = "glm-4-0520"
        GLM_4V = "glm-4v"
        GLM_4V_PLUS = "glm-4v-plus"


class OpenAI(LLM):
    name: str = "OpenAI"

    class Model(Enum):
        GPT_3_5_TURBO = "gpt-3.5-turbo"
        GPT_4O = "gpt-4o"
        GPT_4O_MINI = "gpt-4o-mini"