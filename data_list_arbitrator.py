import re
from functools import cmp_to_key
from typing import Type

from PIL import ImageDraw
from PIL.Image import Image
from loguru import logger

from device import Device
from llm import LLM
from node import Node

from collect_recorder import CollectRecorder
from util import prompt_in_yaml


def find_list_items_groups(static_string_set: set[str]) -> list[list[Node]]:
    root = Device.get_node_list()[0].parent

    res: list[set[Node]] = []
    queue: list[Node] = [root]
    while queue:
        node = queue.pop()
        data_groups = node.find_similar_node_list_in_direct_children(0.2, 0)

        if not data_groups:
            queue.extend(node.children)
        else:
            res.extend(data_groups)

    if not res:
        logger.warning("No similar node list found")
        return []

    return filter_list_items_groups(res, static_string_set)


def filter_list_items_groups(groups: list[set[Node]], static_string_set: set[str]) -> list[list[Node]]:
    # filter group without any text or content-desc
    res = [[node for node in group if node.contain_txt()] for group in groups]
    res = [group for group in res if len(group) > 1]
    res = list(filter(None, res))
    if not res:
        logger.warning("No group with text or content-desc found")
        return []

    # filter group and node only containing static strings
    res = [
        [node for node in group if not node.only_contain_static_text(static_string_set)]
        for group in res
    ]

    res = [group for group in res if len(group) > 1]
    res = list(filter(None, res))
    if not res:
        logger.warning("No group with more than static text found")
        return []

    def sort_by_xpath(node_a: Node, node_b: Node):
        match_a = re.search(r'\[(\d+)](?!.*\[\d+])', node_a.xpath)
        match_b = re.search(r'\[(\d+)](?!.*\[\d+])', node_b.xpath)
        if match_a and not match_b:
            return -1
        elif match_b and not match_a:
            return 1
        elif match_a and match_b:
            index_a = int(match_a.group(1))
            index_b = int(match_b.group(1))
            return index_a - index_b
        return 0

    return [sorted(group, key=cmp_to_key(sort_by_xpath)) for group in res]


def draw_list_items_groups(groups: list[list[Node]]) -> Image:
    img = Device.screenshot()
    draw = ImageDraw.Draw(img)
    color_list = ["red", "blue", "green", "yellow", "purple", "orange"]
    for index, group in enumerate(groups):
        color = color_list[index % len(color_list)]
        for n in group:
            draw.rectangle(n.bounds, outline=color, width=5)
    return img


def select_max_ave_length_group(groups: list[list[Node]]) -> list[Node]:
    return max(groups, key=lambda group: sum(len(node.to_txt()) for node in group) / len(group))
