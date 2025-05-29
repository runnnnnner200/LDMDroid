import base64
import json
from json_repair import repair_json
from io import BytesIO

from PIL.Image import Image
from loguru import logger
from openai import OpenAI as Client


class LLM:
    client: Client
    model_format: str
    model_text: str
    model_visual: str


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
    def init(cls, api_key: str, base_url: str, model_format: str, model_text: str, model_visual: str) -> None:
        cls.client = Client(api_key=api_key, base_url=base_url)
        cls.model_format = model_format
        cls.model_text = model_text
        cls.model_visual = model_visual

    @classmethod
    def chat(cls, prompt: str, temperature: float = 0.5, **kwargs) -> str:
        logger.debug(f"Requesting {cls.model_text}, T:{temperature}, Prompt:\n{prompt}")
        completion = cls.client.chat.completions.create(
            model=cls.model_text,
            messages=[{"role": "user", "content": prompt}],
            top_p=0.7,
            temperature=temperature,
            **kwargs
        )
        res = completion.choices[0].message.content
        logger.debug(f"Response from {cls.model_text}:\n{res}")
        return res

    @classmethod
    def chat_with_image(cls, prompt: str, image: Image, temperature: float = 0.5, **kwargs) -> str:
        image_base64 = LLM._image_to_base64(image)
        logger.debug(f"Requesting {cls.model_visual} with image, T:{temperature}, Prompt:\n{prompt}")
        completion = cls.client.chat.completions.create(
            model=cls.model_visual,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]}, ],
            top_p=0.7,
            temperature=temperature,
            **kwargs
        )
        res = completion.choices[0].message.content
        logger.debug(f"Response from {cls.model_visual}:\n{res}")
        return res

    @classmethod
    def chat_with_image_list(cls, prompt: str, image_list: list[Image], temperature: float = 0.5,
                             **kwargs) -> str:
        image_base64_list = [LLM._image_to_base64(image) for image in image_list]
        logger.debug(
            f"Requesting {cls.model_visual} with {len(image_list)} image, T:{temperature}, Prompt:\n{prompt}")
        completion = cls.client.chat.completions.create(
            model=cls.model_visual,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "image_url": {"url": image_base64}} for image_base64 in image_base64_list]
            ]}, ],
            top_p=0.7,
            temperature=temperature,
            **kwargs
        )
        res = completion.choices[0].message.content
        logger.debug(f"Response from {cls.model_visual}:\n{res}")
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
        formatted_res = cls.chat(prompt, temperature=0.0)
        try:
            out = repair_json(formatted_res, return_objects=True, ensure_ascii=False)
            assert isinstance(out, dict)
            for key in type_json:
                assert key in out  # all keys should be in the output
            return out
        except:
            raise Exception("Formatting llm response failed")