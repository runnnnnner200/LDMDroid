import json
import time
from enum import Enum
from functools import partial
from typing import Type, Any

from PIL import ImageDraw
from PIL.Image import Image
from loguru import logger

from avd_controller import AvdController
from device import Device
from llm import LLM
from node import Node, NodeMathRule
from collect_recorder import CollectRecorder
from util import prompt_in_yaml, node_list_to_ui, action_related_data_node, generate_random_char_list, EventCounter


class ActionType(Enum):
    Click = 1
    LongClick = 2
    InputTexts = 3
    Scroll = 4
    Back = 5


class Action:
    action_type: ActionType
    node: Node | None
    node_match_rule: NodeMathRule | None
    input_content: str | None
    related_data: Node | None
    snapshot: bool
    extra_data: Any

    def __init__(self, action_type: ActionType, node: Node | None, input_content: str = None, extra_data: Any = None):
        self.action_type = action_type
        self.node = node
        self.node_match_rule = None
        self.input_content = input_content
        self.extra_data = extra_data
        self.related_data = None
        self.snapshot = False

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.action_type == other.action_type and self.node == other.node
        return False

    def execute(self, snapshot: bool = False, direct: bool = False):
        # save snapshot
        if snapshot:
            AvdController.snapshot_save("before_action")
            self.snapshot = True

        if self.node is not None:
            target = Device.find_device_node(self.node)

            if not direct:
                self.node_match_rule = NodeMathRule(self.node.xpath, self.node.attributes)

            if self.action_type == ActionType.Click:
                target.click()
            elif self.action_type == ActionType.LongClick:
                x, y = target.center()
                Device.device.long_click(x, y, 0.6)
            elif self.action_type == ActionType.InputTexts:
                if self.input_content:
                    Device.find_device_node(self.node).click()
                    Device.device.clear_text()
                    # for _ in range(5):
                    #     Device.device.shell("input keyevent 20")
                    #     Device.device.clear_text()
                    time.sleep(0.1)
                    Device.device.send_keys(self.input_content)
                else:
                    logger.error("Not supported action: {}", self.to_dict())
        else:
            if self.action_type == ActionType.Scroll:
                if self.node is not None and self.extra_data is not None:
                    # up or down
                    Device.find_device_node(self.node).scroll(self.extra_data)
                else:
                    Device.device(scrollable=True).scroll.toEnd()
            elif self.action_type == ActionType.Back:
                Device.device.press("back")
            else:
                logger.error("Not supported action: {}", self.to_dict())

    def cancel(self):
        if self.snapshot:
            AvdController.snapshot_load("before_action")
            logger.success("Action canceled successfully!")
        else:
            logger.error("No snapshot recorded for this action")

    def to_dict(self) -> dict:
        out = {"action_type": self.action_type.name}
        if self.node:
            out["target_widget"] = self.node.to_txt() if self.node.contain_txt() else (
                self.node.to_id_text() if self.node.contain_id()
                else f"{self.node.tag}()"
            )

            if self.action_type == ActionType.InputTexts and self.input_content:
                out["input_content"] = self.input_content

            if self.related_data is not None:
                out["related_data"] = self.related_data.to_txt()

        return out


class ActionAble:
    data_group: list[Node]
    data_group_text_list: list[str] = []
    data_prompt: str
    action_history: list[tuple[Action, str]]
    block_actions: list[Action]
    task_content: str

    def evaluate_action_by_virtual_llm(self, previous_ui: Image, current_ui: Image, action: Action, llm: Type[LLM],
                                       model_visual: LLM.Model,
                                       temperature: float,
                                       recorder: CollectRecorder) -> str:
        prompt = (
            "You are an expert in UI testing.\n"
            "Test task: {}\n"
            f"{self.data_prompt}"
            "Previous UI: the first image{}.\n"
            "Current UI: the second image.\n"
            "Action:\n{}\n"
            "\n\nBased on the information above, please answer my question.\n"
            "What are the changes in the UI as a result of this action?"
        )

        # draw red box if target widget is given
        if action.node is not None:
            draw = ImageDraw.Draw(previous_ui)
            draw.rectangle(action.node.bounds, outline="red", width=5)

        prompt = prompt.format(
            self.task_content,
            " with the target widget highlighted in a red box" if action.node is not None else "",
            prompt_in_yaml(action.to_dict())
        )
        res = llm.chat_with_image_list(model=model_visual, prompt=prompt, temperature=temperature,
                                       image_list=[previous_ui, current_ui])
        formatted_res = llm.format_to_json(res, {
            "ui_changes": "<string: answer of 'What are the changes in the UI as a result of this action?'>",
        })

        recorder.data["llm"] = {
            "model": model_visual.name,
            "query": prompt,
            "answer": res,
            "formatted_answer": formatted_res,
            "previous_ui": recorder.record_image(previous_ui),
            "current_ui": recorder.record_image(current_ui)
        }

        ui_changes = formatted_res["ui_changes"]
        recorder.data["result"] = {
            "ui_changes": ui_changes,
        }
        return ui_changes
    @staticmethod
    def gen_inputs_by_visual_llm(crud_type: str, package_name: str, activity_name: str, edit_node_list: list[Node],
                                 llm: Type[LLM], model_visual: LLM.Model, temperature: float,
                                 recorder: CollectRecorder | None) -> list[str]:
        random_char_list = json.dumps(generate_random_char_list(5))
        prompt = (
            "In the '{}' app, on the '{}' page, there are {} EditText fields. "
            "The provided image shows the screen of this page, with the EditText fields highlighted in red boxes.\n\n"
            "Details about the EditText fields: \n{}\n\n"
            "During the '{}' operation, please provide one example input for each EditText field."
            " (Each EditText should have exactly one example input.)\n\n"
            f"In order to improve the randomness of input, use characters {random_char_list} as a starting point, and "
            "then expand or transform it into a meaningful input that fits the context.\n"
            "What's the “EditText Field Input” for each EditText field?"
        )

        details = []
        current_ui = Device.screenshot()
        draw = ImageDraw.Draw(current_ui)
        for index, node in enumerate(edit_node_list, 1):
            draw.rectangle(node.bounds, outline="red", width=5)
            draw.text((node.bounds[0][0] + 5, node.bounds[0][1] + 5), str(index), fill="red", font=Device.font)
            txt = node.attributes["hint"].strip()
            if not txt:
                txt = node.attributes["content-desc"].strip()
            if not txt:
                txt = node.attributes["text"]
            details.append(f"{index}. the EditText field is about '{txt}'")

        prompt = prompt.format(
            package_name, activity_name, len(edit_node_list),
            "\n".join(details), crud_type
        )

        res = llm.chat_with_image(model=model_visual, prompt=prompt, temperature=temperature, image=current_ui)
        format_string = "<string[]: all the example input for these EditText fields>" if len(edit_node_list) > 1 \
            else "<string: the example input for this EditText field>"
        formatted_res = llm.format_to_json(
            res, {"input": format_string}
        )

        if len(edit_node_list) == 1:
            if isinstance(formatted_res["input"], list):
                formatted_res["input"] = ["".join(formatted_res["input"])]
            elif isinstance(formatted_res["input"], str):
                formatted_res["input"] = [formatted_res["input"]]

        if recorder is not None:
            recorder.data["llm"] = {
                "model": model_visual.name,
                "query": prompt,
                "answer": res,
                "formatted_answer": formatted_res,
                "current_ui": recorder.record_image(current_ui)
            }

        return formatted_res["input"]

    @staticmethod
    def gen_search_inputs_by_visual_llm(target_data: Node, crud_type: str, package_name: str, activity_name: str,
                                        edit_node_list: list[Node],
                                        llm: Type[LLM], model_visual: LLM.Model, temperature: float,
                                        recorder: CollectRecorder | None) -> list[str]:
        prompt = (
            "In the '{}' app, on the '{}' page, there are {} EditText fields. "
            "The provided image shows the screen of this page, with the EditText fields highlighted in red boxes.\n\n"
            "Details about the EditText fields: \n{}\n\n"
            "Target data:\n{}\n"
            "During the '{}' operation, we are attempting to search for the target data. "
            "Please provide one input for each EditText field. "
            "(Each EditText should have exactly one input.)\n\n"
            "What's the “EditText Field Input” for each EditText field?"
        )

        details = []
        current_ui = Device.screenshot()
        draw = ImageDraw.Draw(current_ui)
        for index, node in enumerate(edit_node_list, 1):
            draw.rectangle(node.bounds, outline="red", width=5)
            draw.text((node.bounds[0][0] + 5, node.bounds[0][1] + 5), str(index), fill="red", font=Device.font)
            txt = node.attributes["hint"].strip()
            if not txt:
                txt = node.attributes["content-desc"].strip()
            if not txt:
                txt = node.attributes["text"]
            details.append(f"{index}. the EditText field is about '{txt}'")

        prompt = prompt.format(
            package_name, activity_name, len(edit_node_list),
            "\n".join(details),
            prompt_in_yaml(target_data.to_txt()),
            crud_type,
        )

        res = llm.chat_with_image(model=model_visual, prompt=prompt, temperature=temperature, image=current_ui)
        format_string = "<string[]: all the input for these EditText fields>" if len(edit_node_list) > 1 \
            else "<string: the input for this EditText field>"
        formatted_res = llm.format_to_json(
            res, {"input": format_string}
        )

        if len(edit_node_list) == 1:
            if isinstance(formatted_res["input"], list):
                formatted_res["input"] = ["".join(formatted_res["input"])]
            elif isinstance(formatted_res["input"], str):
                formatted_res["input"] = [formatted_res["input"]]

        if recorder is not None:
            recorder.data["llm"] = {
                "model": model_visual.name,
                "query": prompt,
                "answer": res,
                "formatted_answer": formatted_res,
                "current_ui": recorder.record_image(current_ui)
            }

        return formatted_res["input"]

    def _fill_in_the_blanks(self, gen_func, crud_type: str, all_node_list: list[Node], package_name: str,
                            activity_name: str,
                            llm: Type[LLM], model: LLM.Model,
                            temperature: float,
                            recorder: CollectRecorder) -> bool:
        """
        :return: if exists any input widget in the UI
        """

        ui_changes = []
        text_changes = []
        node_list = [node for node in all_node_list if
                     node.tag.endswith("EditText") or node.tag.endswith("AutoCompleteTextView")]

        if not node_list:
            return False

        input_list = gen_func(crud_type, package_name, activity_name, node_list, llm, model, temperature, recorder)
        for node, new_text in zip(node_list, input_list):
            old_text = node.attributes["text"]
            action = Action(ActionType.InputTexts, node, new_text)
            ui_change = f"Change the text in the {node.tag} widget from '{old_text}' to '{new_text}'" \
                if old_text else f"Insert '{new_text}' into the EditText widget."
            try:
                action.execute(snapshot=False)  # no need to snapshot for this input action
                EventCounter.add()
                logger.info("Wait for 2s to ensure the action is executed")
                time.sleep(2)
                ui_changes.append(ui_change)
                text_changes.append((old_text, new_text))
            except Exception as e:
                logger.warning("Failed to fill in the blank: {}", e)
                if package_name == "lying.fengfeng.foodrecords":
                    Device.device.press("back")
                time.sleep(2)

        if ui_changes:
            action = Action(ActionType.InputTexts, None)
            action.extra_data = {
                "text_changes": text_changes,
                "ui_changes": ui_changes,
            }
            self.action_history.append((action, "\n".join(ui_changes)))
            return True
        return False

    def fill_in_the_blanks_by_visual_llm(self, crud_type: str, all_node_list: list[Node], package_name: str,
                                         activity_name: str,
                                         llm: Type[LLM], model_visual: LLM.Model,
                                         temperature: float,
                                         recorder: CollectRecorder) -> bool:
        return self._fill_in_the_blanks(self.gen_inputs_by_visual_llm, crud_type, all_node_list, package_name,
                                        activity_name, llm, model_visual, temperature, recorder)

    def fill_in_the_search_blanks_by_visual_llm(self, target_data: Node, all_node_list: list[Node], package_name: str,
                                                activity_name: str,
                                                llm: Type[LLM], model_visual: LLM.Model,
                                                temperature: float,
                                                recorder: CollectRecorder) -> bool:
        return self._fill_in_the_blanks(partial(self.gen_search_inputs_by_visual_llm, target_data),
                                        "Search",
                                        all_node_list,
                                        package_name,
                                        activity_name, llm, model_visual, temperature, recorder)

    def _select_action_by_visual_llm(self, all_node_list: list[Node], package_name: str, activity_name: str,
                                     additional_prompt: str,
                                     llm: Type[LLM], model_visual: LLM.Model,
                                     temperature: float,
                                     recorder: CollectRecorder) -> Action:
        prompt = (
            "In the {} app's {} activity, there is a 'Widget list' with 'Action ID'.\n"
            "The provided image shows "
            "the 'Current UI', where each widget in 'Widget list' is highlighted with a red box, and the index of"
            " it is displayed in the top-left corner. Please fully make use of the meaning of 'Current UI' image.\n"
            "Widget list:\n{}\n"
            "Additional actions:\n{}\n"
            f"{self.data_prompt}"
            "Action history:\n{}\n"
            f"We aims to finish the operation: {self.task_content}\n"
            f"{additional_prompt}\n"
            "\n\nBased on the information above, "
            "please identify the most likely one action for the next step from the 'Widget list' or 'Additional "
            "actions' to "
            f"{self.task_content}"
            ".\nWhat's the most likely Action ID like x-y?"
        )
        prompt, action_dict, node_list = self._prompt_select_action(prompt, all_node_list, package_name, activity_name,
                                                                    loose=True)
        image = Device.screenshot()
        draw = ImageDraw.Draw(image)
        for index, node in enumerate(node_list, 1):
            draw.rectangle(node.bounds, outline="red", width=5)
            draw.text((node.bounds[0][0] + 5, node.bounds[0][1] + 5), str(index), fill="red", font=Device.font)

        res = llm.chat_with_image(model=model_visual, prompt=prompt, temperature=temperature, image=image)
        formatted_res = llm.format_to_json(res, {"action_id": "<string: action id string like i-j>"})

        record = {
            "model": model_visual.name,
            "query": prompt,
            "answer": res,
            "formatted_answer": formatted_res,
            "current_ui": recorder.record_image(image)
        }
        if "llm" in recorder.data:
            recorder.data["llm"].append(record)
        else:
            recorder.data["llm"] = record

        action = action_dict[formatted_res["action_id"]]
        action.related_data = action_related_data_node(self.data_group, action)

        return action

    def _prompt_select_action(self, prompt: str, all_node_list: list[Node], package_name: str, activity_name: str,
                              loose=False) -> tuple[str, dict[str, Action], list[Node]]:
        action_dict, widget_list, node_list = self._widget_list(all_node_list, loose)
        action_dict, additional_actions = self._additional_actions(all_node_list, action_dict,
                                                                   len(widget_list) + 1)

        action_history = self._action_history()
        prompt = prompt.format(
            package_name, activity_name,
            prompt_in_yaml(widget_list),
            prompt_in_yaml(additional_actions),
            prompt_in_yaml(action_history)
        )
        return prompt, action_dict, node_list

    def _node_available_actions(self, node: Node, node_index: int) -> tuple[dict[str, Action], dict]:
        available_text_actions = {}
        available_actions = {}
        action_index = 1

        def add_action(action: Action):
            nonlocal action_index
            if action not in self.block_actions:
                action_id = f"{node_index}-{action_index}"
                available_actions[action_id] = action
                available_text_actions[action_id] = action.action_type.name
                action_index += 1

        for attrib, action_type in (("clickable", ActionType.Click), ("long-clickable", ActionType.LongClick)):
            if node.attributes.get(attrib) == "true":
                add_action(Action(action_type, node))

        if self._equal_to_data(node):
            add_action(Action(ActionType.Click, node))  # consider it as clickable

        return available_actions, available_text_actions

    def _equal_to_data(self, node: Node) -> bool:
        if not self.data_group_text_list:
            self.data_group_text_list = [data.to_txt() for data in self.data_group]
        if self.data_group_text_list[0].find("\n") >= 0:
            return False

        for text in self.data_group_text_list:
            if f"{node.tag}({node.attributes["text"].strip()})" == text:
                return True
        return False

    def _widget_list(self, all_node_list: list[Node], loose=False) -> tuple[dict[str, Action], list[dict], list[Node]]:
        root = all_node_list[0].parent
        widget_list = []
        node_list = []
        action_dict = {}
        widget_index = 1

        def _post_order(node: Node):
            nonlocal widget_index  # make widget_index editable

            res_list = [_post_order(child) for child in node.children]
            if any(res_list):
                return True  # stop

            if node.level == -1:  # virtual root node
                return False

            # system control
            if node.attributes["package"] == "com.android.systemui":
                return False

            if node.attributes["clickable"] == "true" or node.attributes["long-clickable"] == "true" \
                    or self._equal_to_data(node):
                if node.contain_txt():
                    available_actions, available_text_actions = self._node_available_actions(node, widget_index)
                    if len(available_actions):
                        action_dict.update(available_actions)
                        widget_list.append({
                            "index": widget_index,
                            "widget content": node.to_txt(),
                            "available action": available_text_actions
                        })
                        node_list.append(node)
                        widget_index += 1
                    return True  # stop anyway
                elif node.contain_id():
                    available_actions, available_text_actions = self._node_available_actions(node, widget_index)
                    if len(available_actions):
                        action_dict.update(available_actions)
                        widget_list.append({
                            "index": widget_index,
                            "widget content": node.to_id_text(),
                            "available action": available_text_actions
                        })
                        node_list.append(node)
                        widget_index += 1
                    return True  # stop anyway
                elif loose:  # for visual llm (allow node without text)
                    available_actions, available_text_actions = self._node_available_actions(node, widget_index)
                    if len(available_actions):
                        action_dict.update(available_actions)
                        widget_list.append({
                            "index": widget_index,
                            "widget content": f"{node.tag}()",
                            "available action": available_text_actions
                        })
                        node_list.append(node)
                        widget_index += 1
                    # don't stop in this case
            return False

        _post_order(root)

        return action_dict, widget_list, node_list

    def _additional_actions(self, all_node_list: list[Node], action_dict: dict[str, Action],
                            additional_action_index: int) -> tuple[dict[str, Action], dict[str, str]]:
        additional_actions = {}
        index = 1
        back_action = Action(ActionType.Back, None)
        if back_action not in self.block_actions:
            action_id = f"{additional_action_index}-{index}"
            additional_actions[action_id] = "Back"
            action_dict[action_id] = back_action
            index += 1
        # if exist scrollable node
        for n in all_node_list:
            if n.attributes.get("scrollable") == "true":
                scroll_action = Action(ActionType.Scroll, None)
                if scroll_action not in self.block_actions:
                    action_id = f"{additional_action_index}-{index}"
                    additional_actions[action_id] = "Scroll"
                    action_dict[action_id] = scroll_action
                    index += 1
                    break

        return action_dict, additional_actions

    def _action_history(self) -> list[dict]:
        output = []
        for action, ui_change in self.action_history:
            obj = {"action type": action.action_type.name, "ui change": ui_change}
            if action.node is not None:
                obj["target widget"] = action.node.to_txt() if action.node.contain_txt() else (
                    action.node.to_id_text() if action.node.contain_id()
                    else f"{action.node.tag}()"
                )
            output.append(obj)
        return output
