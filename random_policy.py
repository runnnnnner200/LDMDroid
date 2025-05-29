import random
import string

from action import Action, ActionType
from node import Node


class RandomPolicy:
    range_click: int
    range_long_click: int
    range_scroll: int
    range_edit: int
    range_back: int
    range_all: int

    package_name: str

    important_click_class_set = {"android.widget.CheckBox", "android.widget.Button", "android.widget.Switch",
                                 "android.widget.ImageButton"}
    click_package_set: set

    edit_class_set = {"android.widget.EditText", "android.widget.AutoCompleteTextView"}

    string_symbols = ",.!?"
    places_candidates = ["Beijing", "London", "Paris", "New York", "Tokyo"]
    special_candidates = ["www.baidu.com", "www.google.com"]

    def __init__(self, weight_click: int, weight_long_click: int, weight_scroll: int, weight_edit: int,
                 weight_back: int, package_name: str):

        self.range_click = weight_click
        self.range_long_click = self.range_click + weight_long_click
        self.range_scroll = self.range_long_click + weight_scroll
        self.range_edit = self.range_scroll + weight_edit
        self.range_back = self.range_edit + weight_back
        self.range_all = self.range_back

        self.click_package_set = {package_name, "android", "com.android.settings", "com.google.android",
                                  "com.google.android.inputmethod.latin", "com.google.android.permissioncontroller",
                                  "com.android.packageinstaller", "com.android.permissioncontroller",
                                  "com.google.android.packageinstaller"}

    def gen_random_text(self) -> str:
        text_style = random.randint(0, 8)
        text_length = random.randint(1, 5)

        if text_style == 0:
            random_string = ''.join(random.choice(string.ascii_letters) for _ in range(text_length))
        elif text_style == 1:
            random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(text_length))
        elif text_style == 2:
            random_string = ''.join(
                random.choice(string.ascii_letters + string.digits + self.string_symbols) for _ in range(text_length))
        elif text_style == 3:
            random_string = random.choice(self.places_candidates)
        elif text_style == 4:
            random_string = random.choice(string.ascii_letters)
        elif text_style == 5:
            random_string = random.choice(string.digits)
        elif text_style == 6:
            random_string = random.choice(self.special_candidates)
        elif text_style == 7:
            random_string = "10086"
        else:
            random_string = ""

        return random_string

    def gen_random_action(self, all_node_list: list[Node]) -> Action:
        while True:
            event_type = random.randint(0, self.range_all)
            if event_type < self.range_click:
                # Click
                filtered_nodes = []
                important_filtered_nodes = []
                for node in all_node_list:
                    if (node.attributes["clickable"] != "true"
                            or node.attributes["package"] not in self.click_package_set):
                        continue
                    if node.attributes["class"] in self.important_click_class_set:
                        important_filtered_nodes.append(node)
                    else:
                        filtered_nodes.append(node)

                if len(important_filtered_nodes) == 0 and len(filtered_nodes) == 0:
                    continue

                if len(important_filtered_nodes) and len(filtered_nodes) == 0:
                    node = random.choice(important_filtered_nodes)
                elif len(filtered_nodes) and len(important_filtered_nodes) == 0:
                    node = random.choice(filtered_nodes)
                else:
                    # give some chance to click normal nodes
                    if random.random() < 0.67:
                        node = random.choice(important_filtered_nodes)
                    else:
                        node = random.choice(filtered_nodes)

                return Action(ActionType.Click, node)

            elif event_type < self.range_long_click:
                # LongClick
                filtered_nodes = []
                important_filtered_nodes = []
                for node in all_node_list:
                    if (node.attributes["long-clickable"] != "true"
                            or node.attributes["package"] not in self.click_package_set):
                        continue
                    if node.attributes["class"] in self.important_click_class_set:
                        important_filtered_nodes.append(node)
                    else:
                        filtered_nodes.append(node)

                if len(important_filtered_nodes) == 0 and len(filtered_nodes) == 0:
                    continue

                if len(important_filtered_nodes) and len(filtered_nodes) == 0:
                    node = random.choice(important_filtered_nodes)
                elif len(filtered_nodes) and len(important_filtered_nodes) == 0:
                    node = random.choice(filtered_nodes)
                else:
                    # give some chance to click normal nodes
                    if random.random() < 0.67:
                        node = random.choice(important_filtered_nodes)
                    else:
                        node = random.choice(filtered_nodes)

                return Action(ActionType.LongClick, node)

            elif event_type < self.range_scroll:
                # Scroll
                filtered_nodes = []
                for node in all_node_list:
                    if node.attributes["scrollable"] == "true":
                        filtered_nodes.append(node)

                if len(filtered_nodes) > 0:
                    node = random.choice(filtered_nodes)
                    direction = random.choice(["up", "down"])
                    return Action(ActionType.Scroll, node, None, direction)
                else:
                    continue
            elif event_type < self.range_edit:
                # Edit
                filtered_nodes = []
                for node in all_node_list:
                    if node.attributes["class"] in self.edit_class_set:
                        filtered_nodes.append(node)

                if len(filtered_nodes) > 0:
                    node = random.choice(filtered_nodes)
                    random_text = self.gen_random_text()
                    return Action(ActionType.InputTexts, node, random_text)
                else:
                    continue
            else:
                # Back
                return Action(ActionType.Back, None)
