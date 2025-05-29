import subprocess
from collections import deque

import uiautomator2 as u2
from PIL import ImageDraw
from PIL.Image import Image
from loguru import logger
from lxml import etree
from uiautomator2.xpath import XMLElement, XPathSelector
from PIL import ImageFont

from node import Node


class ActiveAppInfo:
    package: str
    activity: str

    def __init__(self, package: str, activity: str):
        self.package = package
        self.activity = activity


class Device:
    device: u2.Device
    device_serial: str
    font: ImageFont

    @classmethod
    def connect(cls, device_serial: str | None = None):
        cls.font = ImageFont.truetype("working/font.ttf", 50)
        cls.device = u2.connect(device_serial)
        cls.device_serial = cls.device.serial

    @classmethod
    def ime_set(cls, enable: bool):
        if not cls.device_serial.startswith("emulator"):
            logger.warning("IME set only work on emulator")
            return

        if enable:
            cls.device.shell("pm enable com.android.inputmethod.latin")
        else:
            cls.device.shell("pm disable-user com.android.inputmethod.latin")

    @classmethod
    def install_app(cls, apk_path: str, uninstall: bool = True):
        cls.device.adb_device.install(apk_path, uninstall=uninstall)

    @classmethod
    def logcat_error(cls, path: str):
        with open(path, "w") as log_file:
            return subprocess.Popen(["adb", "-s", cls.device_serial, "logcat", "*:E", "-v", "process"], stdout=log_file)

    @classmethod
    def logcat_coverage(cls, path: str):
        # adb logcat COVERAGE *:S
        with open(path, "w") as log_file:
            return subprocess.Popen(["adb", "-s", cls.device_serial, "logcat", "COVERAGE", "*:S"], stdout=log_file)

    @classmethod
    def logcat_clear(cls):
        subprocess.run(["adb", "-s", cls.device_serial, "logcat", "-c"])

    @classmethod
    def start_app(cls, package_name: str):
        cls.device.app_start(package_name, wait=True, use_monkey=True)

    @classmethod
    def start_app_session(cls, package_name: str, attach: bool = False):
        return cls.device.session(package_name, attach)

    @classmethod
    def active_app_info(cls) -> ActiveAppInfo:
        info = cls.device.app_current()
        package: str = info["package"]
        activity: str = info["activity"]
        if activity.startswith("."):
            activity = package + activity
        return ActiveAppInfo(package, activity)

    @classmethod
    def get_node_list(cls) -> list[Node]:

        """
        get node list in preorder
        """
        node_list = []

        def _traverse_node(node: etree.Element, parent: Node, level: int):
            cur_node = Node(node, parent, level)
            parent.children.append(cur_node)
            node_list.append(cur_node)

            for i, child in enumerate(node, 1):
                _traverse_node(child, cur_node, level + 1)

        xml_hierarchy = cls.device.dump_hierarchy(max_depth=100)
        root = etree.fromstring(xml_hierarchy.encode())
        virtual_root = Node(root, None, -1, True)
        for c in root:
            _traverse_node(c, virtual_root, 0)

        # xpath
        virtual_root.xpath = "/hierarchy"
        queue: deque[Node | None] = deque(virtual_root.children)
        children_tag_counter = {}
        while queue:
            the_node = queue.popleft()
            if the_node is None:
                children_tag_counter = {}  # next children
            else:
                full_tag = the_node.attributes["class"]
                if children_tag_counter.get(full_tag) is None:
                    children_tag_counter[full_tag] = 1
                else:
                    children_tag_counter[full_tag] += 1
                parent_xpath = the_node.parent.xpath
                cur_xpath = f"/{full_tag}[{children_tag_counter[full_tag]}]"

                the_node.xpath = parent_xpath + cur_xpath

                queue.extend(the_node.children)
                queue.append(None)  # sentinel

        return node_list

    @classmethod
    def find_device_node(cls, node: Node, return_sl=False) -> XMLElement | XPathSelector:
        """
        find the node in device
        """

        def _find_helper(xpath: str):
            sl = Device.device.xpath(xpath)
            if sl.exists:
                elements = sl.all()
                if len(elements) == 1:
                    return True, sl if return_sl else elements[0]
                else:
                    return False, sl if return_sl else elements[0]
            else:
                return False, None

        # resource-id
        resource_id_element = None
        if node.attributes["resource-id"]:
            resource_id_xpath = f'//*[@resource-id="{node.attributes["resource-id"]}"]'
            flag, resource_id_element = _find_helper(resource_id_xpath)
            if flag:
                return resource_id_element

        # full xpath
        flag, full_xpath_element = _find_helper(node.xpath)
        if flag:
            return full_xpath_element

        # relative xpath
        flag, relative_xpath_element = _find_helper(node.xpath_relative)
        if flag:
            return relative_xpath_element

        # attributes xpath
        flag, attributes_xpath_element = _find_helper(node.attributes_xpath)
        if flag:
            return attributes_xpath_element

        for e in [resource_id_element, full_xpath_element, relative_xpath_element, attributes_xpath_element]:
            if e is not None:
                return e

        raise Exception(f"Node not found")

    @classmethod
    def screenshot(cls) -> Image:
        return cls.device.screenshot()

    @classmethod
    def draw_node_with_bound(cls, node: Node, title: str = None) -> Image:
        img = cls.screenshot()
        draw = ImageDraw.Draw(img)
        draw.rectangle(node.bounds, outline="red", width=5)
        if title is not None:
            draw.text((0, 0), title, fill="red", font=cls.font)

        return img

    @classmethod
    def screenshot_with_title(cls, title: str) -> Image:
        img = cls.screenshot()
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), title, fill="red", font=cls.font)
        return img

    @classmethod
    def is_ui_changed(cls, old_node_list: list[Node]) -> bool:
        new_node_list = cls.get_node_list()
        if len(old_node_list) != len(new_node_list):
            return True
        for old_node, new_node in zip(old_node_list, new_node_list):
            if old_node != new_node:
                return True
        return False
