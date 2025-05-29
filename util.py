import json
import os
import random
import string
import sys
import time

import yaml
from loguru import logger

from llm import Zhipu, OpenAI


def prompt_in_yaml(data, extra_intend=2) -> str:
    return "\n".join(" " * extra_intend + line for line in
                     yaml.safe_dump(data, allow_unicode=True, default_style='|').splitlines())


def node_list_to_ui(node_list):
    ui = []
    for node in node_list:
        for name in ("text", "content-desc", "hint", "resource-id"):
            v = node.attributes.get(name)
            if name == "resource-id":
                v = v.split("/")[-1]
            if v:
                ui.append(f"{node.tag}({v})")
                break
    return ui


def edit_text_widget_content(widget) -> str:
    txt = widget.attributes["content-desc"].strip()
    if not txt:
        txt = widget.attributes["text"].strip()
    if not txt:
        txt = widget.attributes.get("hint")
        if txt is not None:
            txt = txt.strip()
        else:
            txt = ""
    return f"{widget.tag}({txt})"


def action_related_data_node(data_list, action):
    cur_node = action.node
    while cur_node and not cur_node.virtual:
        # if cur node matches any data in data_group
        for data_node in data_list:
            if cur_node == data_node:
                return cur_node

        # go up
        cur_node = cur_node.parent

    return None


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def unique_filepath(directory: str, base_name: str):
    base, ext = os.path.splitext(base_name)
    counter = 1
    while True:
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1
        if not os.path.exists(os.path.join(directory, unique_filename)):
            break

    return os.path.join(directory, unique_filename)


def generate_random_char_list(length: int) -> list[str]:
    characters = string.ascii_lowercase
    random_char_list = [random.choice(characters) for _ in range(length)]
    return random_char_list


def basic_init():
    logger.remove()
    logger.add(
        sink=sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
    )

    with open("env.json", "r", encoding="utf-8") as f:
        env = json.load(f)

    if "zhipu" in env:
        Zhipu.init(env["zhipu"]["api_key"], env["zhipu"]["base_url"], format_model=Zhipu.Model.GLM_4_PLUS)
    if "openai" in env:
        OpenAI.init(env["openai"]["api_key"], env["openai"]["base_url"], format_model=OpenAI.Model.GPT_4O_MINI)

    ensure_dir("working/apks")
    ensure_dir("working/cache")
    ensure_dir("working/output")


class EventCounter:
    max_event_num: int
    cur_event_num: int
    inited = False

    @classmethod
    def init(cls, max_event_num: int):
        cls.max_event_num = max_event_num
        cls.cur_event_num = 0
        cls.inited = True

    @classmethod
    def add(cls, num: int = 1):
        if cls.inited:
            cls.cur_event_num += num

    @classmethod
    def is_full(cls):
        if not cls.inited:
            return False

        return cls.cur_event_num >= cls.max_event_num


def timestamp():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
