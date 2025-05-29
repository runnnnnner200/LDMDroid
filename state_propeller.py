from collections import Counter
from enum import Enum
from typing import Type

from loguru import logger

from action import ActionAble, Action
from llm import LLM
from node import Node, NodeMathRule
from collect_recorder import CollectRecorder
from util import prompt_in_yaml


class CRUD(Enum):
    Create = "C"
    Read = "R"
    Update = "U"
    Delete = "D"
    Search = "S"


class StatePropeller(ActionAble):
    crud_type: CRUD
    data_group_match_rule: NodeMathRule
    target_data: Node | None
    operation_workflows: list[str]
    operation_guidance: list[str]
    determine_stage_guidance: list[str]
    op_stage_index: int = 0

    def __init__(self, crud_type: str, specific_crud: str,
                 operation_workflows: list[str], operation_guidance: list[str], determine_stage_guidance: list[str],
                 data_group: list[Node]):
        self.data_group = data_group
        self.operation_workflows = operation_workflows
        self.operation_guidance = operation_guidance
        self.determine_stage_guidance = determine_stage_guidance

        self.action_history = []
        self.block_actions = []
        self.crud_type = CRUD(crud_type)

        self.target_data = data_group[-1] if self.crud_type == CRUD.Search else None

        self.task_content = specific_crud
        # if target data is determined
        if self.target_data is not None:
            self.data_prompt = "Target data of '{}' operation: {}\n".format(
                self.crud_type.name, prompt_in_yaml(self.target_data.to_txt())
            )
        else:
            self.data_prompt = "Data list before '{}' operation: \n{}\n".format(
                self.crud_type.name, prompt_in_yaml([n.to_txt() for n in self.data_group])
            )

        # data group match rule
        data_list_parent = self.data_group[0].parent
        self.data_group_match_rule = NodeMathRule(data_list_parent.xpath, data_list_parent.attributes)

    def is_back_to_data_list_ui(self, all_node_list: list[Node]) -> list[Node] | None:
        # find same parent node
        try:
            node = self.data_group_match_rule.match_node_in_ui(all_node_list)
            for child in node.children:
                # any child node is similar to data list item
                for data in self.data_group:
                    if child.similar_to(data, 0.2, 2):
                        return node.children
            return None
        except ValueError:
            logger.warning("Data list node not found")
            return None
        except Exception as e:
            logger.exception(e)
            return None

    def determine_crud_op_stage(self, package_name: str, llm: Type[LLM], model_text: LLM.Model,
                                temperature: float,
                                recorder: CollectRecorder) -> bool:
        """
        update self.op_stage_index
        :return: op_stage_index == -1 or op_stage_index == len(operation_workflows)-1
        """

        workflow_prompt = "\n".join([f"{index}. {item}" for index, item in enumerate(self.operation_workflows, 1)])

        if self.determine_stage_guidance:
            guidance_prompt = "Additional guidance:\n" + "\n".join(
                [f"{index}. {item}" for index, item in enumerate(self.determine_stage_guidance, 1)]) + "\n\n"
        else:
            guidance_prompt = ""

        prompt = (
            "You are tracking the execution of CRUD operations within the app. Based on the current operation "
            "history, the provided app context, and the last inferred step, infer which specific step of the "
            "operation you are about to perform next.\n\n"
            f"Action history: \n{prompt_in_yaml(self._action_history())}\n\n"
            f"App package name: {package_name}\n\n"
            f"Operation task: {self.task_content}\n\n"
            f"All steps for operation task:\n{workflow_prompt}\n\n"
            f"{guidance_prompt}"
            "Based on the provided operation history, app context, and the last inferred step, infer which specific "
            "step you are about to perform next. "
            f"Please choose from the “All steps for {self.crud_type.name} operation”.\n"
            f"If all steps for the {self.crud_type.name} operation have been completed, "
            f"output: 0. All steps completed."
        )

        least_times = 2
        test_times = len(self.operation_workflows) * (least_times - 1) + 1
        results = []
        for dt in range(test_times):
            temp = round(temperature + 0.1 * dt, 1)
            if temp > 1:
                temp = 1.0

            res = llm.chat(model=model_text, prompt=prompt, temperature=temp)
            formatted_res = llm.format_to_json(res, {
                "step_number": "<int: specific step number of the operation you are about to perform next>"
            })

            op_stage_index = int(formatted_res["step_number"]) - 1
            if op_stage_index >= len(self.operation_workflows) or op_stage_index < -1:
                logger.error(f"op_stage_index {op_stage_index} is out of range")
                return False

            results.append(op_stage_index)
            times_counter = Counter(results)
            if times_counter.most_common()[0][1] >= least_times:
                self.op_stage_index = op_stage_index
                recorder.data["llm"] = {
                    "model": model_text.name,
                    "query": prompt,
                    "answer": res,
                    "formatted_answer": formatted_res
                }
                return op_stage_index == -1 or op_stage_index == len(self.operation_workflows) - 1

        return False

    def select_action_by_visual_llm(self, all_node_list: list[Node], package_name: str, activity_name: str,
                                    llm: Type[LLM], model_visual: LLM.Model,
                                    temperature: float,
                                    recorder: CollectRecorder) -> Action:
        workflows_prompt = "\n".join([f"{index}. {step}" for index, step in enumerate(self.operation_workflows, 1)])
        additional_prompt = (
            f"\nThe operation general workflow:\n{workflows_prompt}\n\n"
            f"The specific workflows of the {self.crud_type.name} operation may vary depending on the app. "
            f"You don’t need to follow these steps strictly—they are provided as a reference only.\n"
        )

        if self.op_stage_index >= 0:
            additional_prompt += (
                "The next step you need to take is "
                f"{self.op_stage_index + 1}. {self.operation_workflows[self.op_stage_index]}"
            )

        if self.operation_guidance:
            guidance_prompt = "\n".join([f"{index}. {item}" for index, item in enumerate(self.operation_guidance, 1)])
            additional_prompt += f"\n\nAdditional guidance:\n{guidance_prompt}"

        action = self._select_action_by_visual_llm(all_node_list, package_name, activity_name, additional_prompt,
                                                   llm, model_visual, temperature, recorder)
        # set target data
        if self.crud_type != CRUD.Create and self.crud_type != CRUD.Search \
                and self.target_data is None and action.related_data is not None:
            self.target_data = action.related_data
            self.data_prompt = "Target data of '{}' operation: {}\n".format(
                self.crud_type.name, prompt_in_yaml(self.target_data.to_txt())
            )

        return action
