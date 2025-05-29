from typing import Callable
from loguru import logger
from lxml.etree import Element


class NodeMathRule:
    xpath: str
    attributes: dict

    xpath_relative: str
    attributes_xpath: str

    def __init__(self, xpath: str, attributes: dict):
        self.xpath = xpath
        self.attributes = attributes

        path_list = xpath[1:].split("/")
        self.xpath_relative = f'//{"/".join(path_list[-5:])}'

        new_attrs = self.attributes.copy()
        new_attrs.pop("class")
        new_attrs.pop("package")
        # new_attrs.pop("text")
        attr_str = " and ".join(
            [f'@{key}="{value}"' for key, value in new_attrs.items()]
        )
        self.attributes_xpath = f"//{self.attributes['class']}[{attr_str}]"

    @staticmethod
    def from_dict(obj: dict) -> 'NodeMathRule':
        return NodeMathRule(obj["xpath"], obj["attributes"])

    def to_dict(self):
        return {"xpath": self.xpath, "attributes": self.attributes}

    @staticmethod
    def _match_only_one(all_node_list: list['Node'], match_func: Callable[['Node'], bool]):
        """
        :return: (flag, node) flag: whether match only one node; node: the first node match the rule
        """
        out = None
        for node in all_node_list:
            if match_func(node):
                if out is not None:
                    return False, out
                out = node
        if out is not None:
            return True, out
        return False, None

    def match_node_in_ui(self, all_node_list: list['Node']) -> 'Node':
        # resource-id
        node_by_id = None
        if self.attributes["resource-id"]:
            flag, node_by_id = self._match_only_one(
                all_node_list, lambda n: n.attributes["resource-id"] == self.attributes["resource-id"])
            if flag:
                return node_by_id
        # full_xpath
        flag, node_by_full_xpath = self._match_only_one(all_node_list, lambda n: n.xpath == self.xpath)
        if flag:
            return node_by_full_xpath
        # relative_xpath
        flag, node_by_relative_xpath = self._match_only_one(all_node_list, lambda n: n.xpath_relative == self.xpath)
        if flag:
            return node_by_relative_xpath
        # attributes_xpath
        flag, node_by_attributes_xpath = self._match_only_one(all_node_list,
                                                              lambda n: n.attributes_xpath == self.attributes_xpath)
        if flag:
            return node_by_attributes_xpath

        # match more than one node
        match_dependency = [self.attributes_xpath, self.xpath, self.xpath_relative, self.attributes["resource-id"]]
        match_node_list = [node_by_attributes_xpath, node_by_full_xpath, node_by_relative_xpath, node_by_id]
        for dep, node in zip(match_dependency, match_node_list):
            if node is not None:
                logger.warning("match more than one node, dependency:\n{}\nchoose:\n{}", dep, node.to_visualized_txt())
                return node
        # no node found
        raise ValueError("no node match")


class Node:
    parent: 'Node|None'
    children: list['Node']

    level: int
    attributes: dict[str, str]
    tag: str
    bounds: tuple[tuple[int, int], tuple[int, int]]  # for drawing rect
    _xpath: str
    xpath_relative: str
    attributes_xpath: str
    virtual: bool

    def __init__(self, node: Element, parent: 'Node|None', level: int, virtual: bool = False):
        self.parent = parent
        self.level = level
        self.attributes = dict(node.attrib)

        if not virtual:
            bounds = self.attributes['bounds']
            parts = bounds.strip('[]').split('][')
            [x1, y1] = parts[0].split(',')
            [x2, y2] = parts[1].split(',')
            self.bounds = ((int(x1), int(y1)), (int(x2), int(y2)))
            self.tag = self.attributes["class"].split(".")[-1]

            new_attrs = self.attributes.copy()
            new_attrs.pop("class")
            new_attrs.pop("package")
            # new_attrs.pop("text")
            attr_str = " and ".join(
                [f'@{key}="{value}"' for key, value in new_attrs.items()]
            )
            self.attributes_xpath = f"//{self.attributes['class']}[{attr_str}]"

        self.virtual = virtual
        self.children = []

    @property
    def xpath(self):
        return self._xpath

    @xpath.setter
    def xpath(self, xpath: str):
        self._xpath = xpath
        path_list = xpath[1:].split("/")
        xpath_relative_list = path_list[-5:]
        if xpath_relative_list[0] == "hierarchy" and len(xpath_relative_list) >= 2:
            xpath_relative_list.pop(0)
            xpath_relative_list.pop(0)
        self.xpath_relative = f'//{"/".join(xpath_relative_list)}'

    def __str__(self) -> str:
        return f"Node(tag: {self.tag}, level: {self.level}, text: {self.attributes.get('text')})"

    def __eq__(self, other):
        if isinstance(other, Node):
            # cur level
            if self.level != other.level:
                return False
            if any([self.attributes.get(key) != other.attributes.get(key) for key in
                    ("package", "class", "resource-id", "text", "content-desc", "hint")]):
                return False

            if len(self.children) != len(other.children):
                return False
            # children level
            for i in range(len(self.children)):
                # any not same -> False
                if self.children[i] != other.children[i]:
                    return False
            return True
        return False

    def __hash__(self):
        return hash(self.xpath)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "attributes": self.attributes
        }

    def to_txt(self) -> str:
        out = []
        txt = self.attributes["text"].strip()
        if not txt:
            txt = self.attributes["content-desc"].strip()
        if not txt:
            txt = self.attributes.get("hint")
            if txt is not None:
                txt = txt.strip()
        if txt:
            out.append(f"{self.tag}({txt})")

        for child in self.children:
            t = child.to_txt()
            if t:
                out.append(t)

        return "\n".join(out)

    def to_id_text(self):
        # find the first non-empty id
        rid = self.attributes["resource-id"].split("/")[-1]
        if rid:
            return f"{self.tag}({rid})"

        for child in self.children:
            rid = child.to_id_text()
            if rid is not None:
                return rid
        return None

    def to_visualized_txt(self, level: int = 0) -> str:
        out = []
        txt = self.attributes["text"].strip()
        if not txt:
            txt = self.attributes["content-desc"].strip()
        if txt:
            out.append(f'| {" " * level}{txt}' if level > 0 else f'├─{" " * level}{txt}')
            new_level = level + 1
        else:
            new_level = level

        for child in self.children:
            t = child.to_visualized_txt(new_level)
            if t:
                out.append(t)

        return "\n".join(out)

    def contain_txt(self) -> bool:
        if self.attributes.get("text"):
            return True
        elif self.attributes.get("content-desc"):
            return True
        elif self.attributes.get("hint"):
            return True

        for child in self.children:
            if child.contain_txt():
                return True

        return False

    def contain_id(self) -> bool:
        if self.attributes.get("resource-id"):
            return True

        for child in self.children:
            if child.contain_id():
                return True

    def only_contain_static_text(self, static_string_set: set[str]) -> bool:
        txt = self.attributes.get("text")
        if not txt:
            txt = self.attributes.get("content-desc")
        if not txt:
            txt = self.attributes.get("hint")

        if txt:
            if txt not in static_string_set:
                return False

        for child in self.children:
            if not child.only_contain_static_text(static_string_set):
                return False

        # no text -> true
        return True

    def similar_to(self, node: 'Node', tolerance_rate: float, tolerance_count: int) -> bool:
        """
        Check if the nodes are similar in structure
        """

        # class, package must be the same
        for name in ("class", "package"):
            if self.attributes.get(name) != node.attributes.get(name):
                return False

        len1 = len(self.children)
        len2 = len(node.children)

        # handle empty node cases
        if len1 == 0 and len2 == 0:
            return True
        elif len1 == 0 or len2 == 0:
            return False

        # child length difference tolerance
        count_diff = abs(len1 - len2)
        if count_diff > max(len1, len2) * tolerance_rate and count_diff > tolerance_count:
            return False

        # child node similarity
        similar_count = 0
        matched = [False] * len2
        for i in range(len1):
            for j in range(len2):
                if not matched[j] and self.children[i].similar_to(node.children[j], tolerance_rate,
                                                                  tolerance_count):
                    matched[j] = True
                    similar_count += 1

        if similar_count / len2 >= 1 - tolerance_rate or len2 - similar_count <= tolerance_count:
            return True

    def similar_to_list_item(self, node: 'Node', tolerance_rate: float, tolerance_count: int) -> bool:
        """
        Check if the nodes are similar, possibly located in the same list structure
        """

        # class, package, resource-id must be the same
        for name in ("class", "package", "resource-id"):
            if self.attributes.get(name) != node.attributes.get(name):
                return False

        # x and y must be different
        if self.bounds[0][0] == node.bounds[0][0] and self.bounds[0][1] == node.bounds[0][1]:
            return False

        # width difference tolerance
        max_width = max(self.bounds[1][0] - self.bounds[0][0], node.bounds[1][0] - node.bounds[0][0])
        width_diff = abs(self.bounds[1][0] - self.bounds[0][0] - node.bounds[1][0] + node.bounds[0][0])
        if width_diff > max_width * tolerance_rate:
            return False

        len1 = len(self.children)
        len2 = len(node.children)

        # handle empty node cases
        if len1 == 0 and len2 == 0:
            return True
        elif len1 == 0 or len2 == 0:
            return False

        # child length difference tolerance
        count_diff = abs(len1 - len2)
        if count_diff > max(len1, len2) * tolerance_rate and count_diff > tolerance_count:
            return False

        # child node similarity
        similar_count = 0
        matched = [False] * len2
        for i in range(len1):
            for j in range(len2):
                if not matched[j] and self.children[i].similar_to_list_item(node.children[j], tolerance_rate,
                                                                            tolerance_count):
                    matched[j] = True
                    similar_count += 1

        if similar_count / len2 >= 1 - tolerance_rate or len2 - similar_count <= tolerance_count:
            return True

    def find_similar_node_list_in_direct_children(self, tolerance_rate: float,
                                                  tolerance_count: int) -> list[set['Node']]:
        """
        Find similar node set list representing data groups in direct children
        """
        similar_node_pairs = []
        length = len(self.children)
        for i in range(length):
            for j in range(i + 1, length):
                if self.children[i].similar_to_list_item(self.children[j], tolerance_rate, tolerance_count):
                    similar_node_pairs.append((self.children[i], self.children[j]))

        if not similar_node_pairs:
            return []

        # there could be multiple similar node groups
        similar_node_set_list: list[set['Node']] = [set(similar_node_pairs[0])]
        cur_set = similar_node_set_list[0]
        for pair in similar_node_pairs:
            if pair[0] in cur_set or pair[1] in cur_set:
                cur_set.add(pair[0])
                cur_set.add(pair[1])
            else:
                similar_node_set_list.append(set(pair))
                cur_set = similar_node_set_list[-1]

        return similar_node_set_list
