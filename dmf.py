import json
import os
import sys
import time
from typing import Type, Callable

from loguru import logger

import data_list_arbitrator
import guidance
from action import ActionType, Action
from apk import Apk
from avd_controller import AvdController
from device import Device
from llm import LLM
from node import Node, NodeMathRule
from collect_recorder import CollectRecorder, CollectRecorderType
from state_propeller import StatePropeller, CRUD
from util import prompt_in_yaml, action_related_data_node, unique_filepath, EventCounter, timestamp


def find_list_item(apk: Apk) -> list[Node] | None:
    recorder = CollectRecorder(CollectRecorderType.FindListItem)

    groups = data_list_arbitrator.find_list_items_groups(apk.static_string_set)
    if groups:
        chosen_group = data_list_arbitrator.select_max_ave_length_group(groups)
        data_list_parent = chosen_group[0].parent
        data_list = data_list_parent.children

        recorder.data["result"] = {
            "data_list": recorder.record_image(data_list_arbitrator.draw_list_items_groups([data_list])),
        }
        recorder.save()

        return data_list

    recorder.data["result"] = None
    recorder.save()
    return None


def specific_crud_meanings(package_name: str, temperature: float, llm: Type[LLM], model_text: LLM.Model):
    recorder = CollectRecorder(CollectRecorderType.SpecificCRUD)

    prompt = f'''You are an Android automation testing tool. Based on the provided App context information, please describe the actual meaning of the following CRUD operations, without needing to specify the exact content of each operation.

App Context Information:
- App Package Name: {package_name}
- UI Screenshot: the given image.

Based on this context, please describe the actual meaning of the following CRUD operations:
- C (Create): Add a new item to the list in the app.
- R (Read): View the details or properties of an item in the list.
- U (Update): Modify an item in the list.
- D (Delete): Remove an item from the list.
- S (Search): Search for an item in the list.

Example:
Assume the app is a note-taking app, and the current screen is a list of notes. Based on the context, we can describe the CRUD operations as follows:
- C (Create): Add a new note to the list.
- R (Read): View the detailed content of a note within the list.
- U (Update): Modify the title or content of a note in the list.
- D (Delete): Remove a note from the list.
- S (Search): Search for a note in the list.

Assume the app is a file-management app, and the current screen is a list of files. Based on the context, we can describe the CRUD operations as follows:
- C (Create): Add a new file or folder to the list.
- R (Read): Retrieve the properties of a specific file within the list.
- U (Update): Modify the name of a file or folder present in the list.
- D (Delete): Remove a file or folder from the list.
- S (Search): Search for a file or folder in the list.'''

    image = Device.screenshot()
    res = llm.chat_with_image(model=model_text, prompt=prompt, image=image, temperature=temperature)
    formatted_res = llm.format_to_json(res, {
        "C": "<the description of Create operation>",
        "R": "<the description of Read operation>",
        "U": "<the description of Update operation>",
        "D": "<the description of Delete operation>",
        "S": "<the description of Search operation>",
    })

    for t in ["C", "R", "U", "D", "S"]:
        if t not in formatted_res:
            raise ValueError(f"fail to specific crud meanings, {t} not found in {formatted_res}")

    recorder.data["llm"] = {
        "model": model_text.name,
        "query": prompt,
        "answer": res,
        "formatted_answer": formatted_res
    }
    recorder.data["result"] = formatted_res
    recorder.save()
    return formatted_res


def _state_propel_action(sp: StatePropeller, all_node_list: list[Node], max_action_exec: int, package_name: str,
                         activity_name: str, temperature: float, llm: Type[LLM], model_visual: LLM.Model,
                         ) -> -1 | 0 | 1:
    """
    :return: -1:  Max action exec reached, 0:  ui not changed, 1:  ui changed
    """

    # check max action exec
    select_recorder = CollectRecorder(CollectRecorderType.SelectAction)
    length = len(sp.action_history)
    if length >= max_action_exec:
        select_recorder.data["result"] = f"Max action exec reached: {length}"
        select_recorder.save()

        logger.info("Max action exec reached: {}", length)
        return -1

    if EventCounter.is_full():
        select_recorder.data["result"] = f"Max event num reached: {EventCounter.cur_event_num}"
        select_recorder.save()

        logger.info("Max event num reached: {}", EventCounter.cur_event_num)
        return -1

    if (sp.crud_type == CRUD.Create or sp.crud_type == CRUD.Update) and \
            sp.operation_workflows[sp.op_stage_index].startswith("Enter") and \
            sp.action_history[-1][0].action_type != ActionType.InputTexts:
        # input text
        if sp.fill_in_the_blanks_by_visual_llm(sp.crud_type.name, all_node_list, package_name, activity_name, llm,
                                               model_visual,
                                               temperature, select_recorder):
            action = sp.action_history[-1][0]
            select_recorder.data["result"] = action.to_dict()
            select_recorder.data["result"]["image"] = select_recorder.record_image(Device.screenshot())

            select_recorder.save()
            logger.info("Propeller force input text: {}", action.to_dict())
            return 1

    # Search the last data of list
    if sp.crud_type == CRUD.Search and \
            sp.operation_workflows[sp.op_stage_index].startswith("Enter") and \
            sp.action_history[-1][0].action_type != ActionType.InputTexts:
        if sp.fill_in_the_search_blanks_by_visual_llm(sp.target_data, all_node_list, package_name,
                                                      activity_name, llm, model_visual,
                                                      temperature, select_recorder):
            action = sp.action_history[-1][0]
            select_recorder.data["result"] = action.to_dict()
            select_recorder.data["result"]["image"] = select_recorder.record_image(Device.screenshot())
            select_recorder.save()
            logger.info("Propeller force input text: {}", action.to_dict())
            return 1

    # select action
    action = sp.select_action_by_visual_llm(
        all_node_list=all_node_list,
        package_name=package_name,
        activity_name=activity_name,
        llm=llm,
        model_visual=model_visual,
        temperature=temperature,
        recorder=select_recorder
    )

    select_recorder.data["result"] = action.to_dict()
    if action.node is not None:
        select_recorder.data["result"]["image"] = select_recorder.record_image(
            Device.draw_node_with_bound(action.node))

    select_recorder.save()
    logger.info("Propeller selected action: {}", action.to_dict())

    # evaluate action
    evaluate_recorder = CollectRecorder(CollectRecorderType.EvaluateAction)
    previous_ui_image = Device.screenshot_with_title("Previous UI")
    action.execute()
    EventCounter.add()
    logger.info("Wait for 2s to ensure the action is executed")
    time.sleep(2)
    if Device.is_ui_changed(all_node_list):
        if action.node is not None:
            sp.block_actions.append(action)  # block it

        if Device.active_app_info().package == "com.android.launcher3":
            raise ValueError("Back to desktop, restart this round")

        ui_changes = sp.evaluate_action_by_virtual_llm(
            previous_ui=previous_ui_image,
            current_ui=Device.screenshot_with_title("Current UI"),
            action=action,
            llm=llm,
            model_visual=model_visual,
            temperature=temperature,
            recorder=evaluate_recorder
        )
        evaluate_recorder.save()
        sp.action_history.append((action, ui_changes))

        return 1
    else:
        logger.warning("UI not changed after action")
        if action.node is not None:
            sp.block_actions.append(action)  # block it
        evaluate_recorder.data["result"] = "UI not changed"
        evaluate_recorder.save()
        sp.action_history.append((action, "UI not changed"))
        return 0


def complete_crud_operation(max_action_exec: int, sp: StatePropeller, package_name: str,
                            activity_name: str, temperature: float,
                            llm: Type[LLM], model_visual: LLM.Model, model_text: LLM.Model,
                            ) -> -1 | 0 | 1:
    """
    :return: -1:  Max action exec reached, 0:  ui not changed, 1:  ui changed
    """

    while True:
        action_res = _state_propel_action(sp, Device.get_node_list(), max_action_exec, package_name, activity_name,
                                          temperature, llm, model_visual,
                                          )
        if action_res == -1:  # max action exec reached
            return action_res
        elif action_res == 1:
            # determine if operation finished
            finish_recorder = CollectRecorder(CollectRecorderType.DetermineCrudStage)
            if sp.determine_crud_op_stage(
                    package_name=package_name,
                    llm=llm,
                    model_text=model_text,
                    temperature=temperature,
                    recorder=finish_recorder,
            ):
                res = f"Operation finished! Current stage: {sp.operation_workflows[sp.op_stage_index]}"
                logger.success(res)
                finish_recorder.data["result"] = res
                finish_recorder.save()
                return action_res
            else:
                res = f"Operation not finished! Current stage: {sp.operation_workflows[sp.op_stage_index]}"
                logger.info(res)
                finish_recorder.data["result"] = res
                finish_recorder.save()
        else:  # UI not changed
            pass


def back_to_data_list(max_action_exec: int, sp: StatePropeller, package_name: str, activity_name: str,
                      temperature: float, llm: Type[LLM], model_visual: LLM.Model) -> list[Node] | None:
    action_res = 1
    while True:
        all_node_list = Device.get_node_list()
        if action_res == 1:
            # determine if back to data list
            res = sp.is_back_to_data_list_ui(all_node_list)
            return_recorder = CollectRecorder(CollectRecorderType.DetermineReturnToDataList)
            return_recorder.data["result"] = {"msg": "Successfully return to the page with the 'Data list'!"} \
                if res is not None else {"msg": "Not return to the page with the 'Data list'"}
            return_recorder.data["result"]["image"] = return_recorder.record_image(Device.screenshot())
            return_recorder.save()
            if res is not None:
                return res

        action_res = _state_propel_action(sp, all_node_list, max_action_exec, package_name, activity_name, temperature,
                                          llm, model_visual)
        if action_res == -1:  # max action exec reached
            return None


def check_for_dme_by_llm(crud_type: CRUD, target_data: Node | None, old_data_list: list[Node] | None,
                         new_data_list: list[Node] | None, input_data: dict | None,
                         llm: Type[LLM], model_visual: LLM.Model, model_text: LLM.Model,
                         temperature: float,
                         ) -> tuple[bool, str, str, str]:
    least_times = 2
    test_times = 2 * (least_times - 1) + 1
    out = []
    for dt in range(test_times):
        temp = round(temperature + 0.1 * dt, 1)
        if temp > 1:
            temp = 1.0

        recorder = None
        if CollectRecorder.output_dir is not None:
            recorder = CollectRecorder(CollectRecorderType.CheckForDME)
            recorder.data["result"] = {
                "type": crud_type.name,
            }

        task_background = (
            "We are performing UI testing for an Android app and need to evaluate whether a logical error occurs during"
            f"the {crud_type.name} operation. "
            "We have recorded the data list displayed in the app interface, along with relevant text input "
            "information. We will use this information to determine if the operation was successful or if a logical "
            "error exists."
        )

        additional_guidance = "\n".join(
            [f"{index}. {item}" for index, item in enumerate(guidance.all_test_oracle_guidance["Additional"], 1)]
        )

        if crud_type != CRUD.Create:
            if target_data is None:
                logger.error("Data target not determined")
                if recorder is not None:
                    recorder.data["result"]["msg"] = "Data target not determined"
                    recorder.save()
                return False, "Data target not determined", "", ""

        if crud_type == CRUD.Read:
            prompt = (
                f"{task_background}\n\n"
                "Target Data of Read operation:\n{}\n\n"
                "Read result: the image displayed the result of Read operation.\n\n"
                f"{guidance.all_test_oracle_guidance[crud_type.value]}\n\n"
                f"Additional guidance:\n{additional_guidance}\n\n"
                "Based on the information above, please answer my question.\n"
                "Is there any logical error in the Read operation? Why?"
            ).format(target_data.to_txt())
            read_result = Device.screenshot()
            res = llm.chat_with_image(model=model_visual, prompt=prompt, image=read_result, temperature=temp)
            formatted_res = llm.format_to_json(
                res,
                {
                    "op_invalid": f"<boolean: answer of 'Is there any logical error in the {crud_type.name} operation?'>",
                    "reason": "<string>"
                }
            )

            if recorder is not None:
                recorder.data["llm"] = {
                    "model": model_visual.name,
                    "query": prompt,
                    "answer": res,
                    "formatted_answer": formatted_res,
                    "read_result": recorder.record_image(read_result)
                }

        elif crud_type == CRUD.Search:
            input_prompt = "Search input related information:\n{}".format('\n'.join([
                f"{i}. {v}." for i, v in enumerate(input_data['ui_changes'], 1)
            ]))
            target_data_prompt = f"Target Data of {crud_type.name} operation:\n{target_data.to_txt()}\n\n"

            prompt = (
                "We are performing UI testing for an Android app and need to evaluate whether a logical error occurs during"
                "the Search operation. "
                "We have recorded the target data being searched for, along with the corresponding search keyword inputs. "
                "This information will be used to determine whether the operation was successful or if a logical error occurred.\n\n"
                f"Target data for the Search operation:\n{target_data_prompt}\n\n"
                f"{input_prompt}\n\n"
                "Search result: the provided image shows the outcome of the Search operation.\n\n"
                f"{guidance.all_test_oracle_guidance[crud_type.value]}\n\n"
                f"Additional guidance:\n{additional_guidance}\n\n"
                "Based on the information above, please answer my question.\n"
                "Is there any logical error in the Read operation? Why?"
            )

            search_result = Device.screenshot()
            res = llm.chat_with_image(model=model_visual, prompt=prompt, image=search_result, temperature=temp)
            formatted_res = llm.format_to_json(
                res,
                {
                    "op_invalid": f"<boolean: answer of 'Is there any logical error in the {crud_type.name} operation?'>",
                    "reason": "<string>"
                }
            )

            if recorder is not None:
                recorder.data["llm"] = {
                    "model": model_visual.name,
                    "query": prompt,
                    "answer": res,
                    "formatted_answer": formatted_res,
                    "read_result": recorder.record_image(search_result)
                }

        else:
            prompt = (
                f"{task_background}\n\n"
                "Data list before {0} operation:\n{1}\n\n"
                "Data list after {0} operation:\n{2}\n\n"
                "{3}\n\n"  # related info
                f"{guidance.all_test_oracle_guidance[crud_type.value]}\n\n"
                f"Additional guidance:\n{additional_guidance}\n\n"
                "Based on the information above, please answer my question.\n"
                "Is there any logical error in the {0} operation? Why?"
            )

            input_prompt = ""
            if input_data is not None:
                input_prompt = "Input related information:\n{}".format('\n'.join([
                    f"{i}. {v}." for i, v in enumerate(input_data['ui_changes'], 1)
                ]))

            if crud_type == CRUD.Create:
                prompt = prompt.format(
                    crud_type.name,
                    prompt_in_yaml([n.to_txt() for n in old_data_list]),
                    prompt_in_yaml([n.to_txt() for n in new_data_list]),
                    input_prompt
                )
            else:
                target_data_prompt = f"Target Data of {crud_type.name} operation:\n{target_data.to_txt()}\n\n"
                if crud_type == CRUD.Update:
                    prompt = prompt.format(
                        crud_type.name,
                        prompt_in_yaml([n.to_txt() for n in old_data_list]),
                        prompt_in_yaml([n.to_txt() for n in new_data_list]),
                        f"{target_data_prompt}{input_prompt}"
                    )
                else:  # crud_type == CRUD.Delete:
                    prompt = prompt.format(
                        crud_type.name,
                        prompt_in_yaml([n.to_txt() for n in old_data_list]),
                        prompt_in_yaml([n.to_txt() for n in new_data_list]),
                        target_data_prompt
                    )

            res = llm.chat(model=model_text, prompt=prompt, temperature=temp)
            formatted_res = llm.format_to_json(
                res,
                {
                    "op_invalid": f"<boolean: answer of 'Is there any logical error in the {crud_type.name} operation?'>",
                    "reason": "<string>"
                }
            )

            if recorder is not None:
                recorder.data["llm"] = {
                    "model": model_text.name,
                    "query": prompt,
                    "answer": res,
                    "formatted_answer": formatted_res
                }

        if recorder is not None:
            recorder.data["result"]["answer"] = formatted_res
            recorder.save()

        out.append((formatted_res["op_invalid"], formatted_res["reason"], res, prompt))
        op_invalid_times = sum(1 for x in out if x[0] is True)
        op_valid_times = len(out) - op_invalid_times
        if op_invalid_times >= least_times:
            return next((x for x in out if x[0] is True), None)
        if op_valid_times >= least_times:
            return next((x for x in out if x[0] is False), None)


def dme_report(prompt: str, reason: str, full_res: str, sp: StatePropeller, recorder_start_index: int):
    crud = sp.crud_type.name
    recorder_end_index = CollectRecorder.counter
    doc = [
        "# DME Report",
        "## Data manipulation type", crud,
        "## Prompt", prompt,
        "## Reason", f">{reason}", full_res,
        "## Recorder", f"{recorder_start_index}.json ~ {recorder_end_index}.json"
    ]
    output_path = os.path.join(CollectRecorder.output_dir, f"DME-{crud}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(doc))


def record_dmf(sp: StatePropeller, apk_path: str, out_dir: str, ):
    out = {
        "apk_path": apk_path,
        "dmf_type": sp.crud_type.name,
        "data_group_match": sp.data_group_match_rule.to_dict(),
        "actions": []
    }

    for action, _ in sp.action_history:
        obj = {
            "type": action.action_type.name,
            "target": action.node_match_rule.to_dict() if action.node_match_rule is not None else None
        }
        if action.action_type == ActionType.InputTexts:
            obj["input"] = action.extra_data["text_changes"]
        out["actions"].append(obj)

    file_path = os.path.join(out_dir, f"{sp.crud_type.name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    logger.success("DMF saved to {}", file_path)


def collect_dmf(apk: Apk, max_action_exec: int, crud_type: str, specific_crud: str,
                data_group: list[Node],
                operation_workflows: list[str], operation_guidance: list[str], determine_stage_guidance: list[str],
                temperature: float, out_dir: str, llm: Type[LLM], model_visual: LLM.Model,
                model_text: LLM.Model,
                ) -> -1 | 0 | 1:
    """
    :return: -1: exist dme, 0: failed to complete dmf, 1: success to complete dmf
    """
    recorder_start_index = CollectRecorder.counter + 1

    activity_name = Device.active_app_info().activity.split(".")[-1]

    sp = StatePropeller(crud_type, specific_crud, operation_workflows, operation_guidance, determine_stage_guidance,
                        data_group)

    complete_res = complete_crud_operation(max_action_exec, sp, apk.package_name, activity_name, temperature,
                                           llm, model_visual, model_text)
    if complete_res == -1:
        logger.error("Failed to complete dmf")
        return 0

    # fetch latest input data form history
    input_data = None
    for history in reversed(sp.action_history):
        if history[0].action_type == ActionType.InputTexts:
            input_data = history[0].extra_data
            break

    new_data_list = None
    if sp.crud_type != CRUD.Read and sp.crud_type != CRUD.Search:
        new_data_list = back_to_data_list(max_action_exec, sp, apk.package_name, activity_name, temperature, llm,
                                          model_visual)
        if new_data_list is None:
            logger.error("Failed to return to data list, max action exec reached")
            return 0

    op_invalid, reason, llm_res, prompt = check_for_dme_by_llm(sp.crud_type, sp.target_data, sp.data_group,
                                                               new_data_list,
                                                               input_data, llm, model_visual, model_text, temperature)
    if op_invalid:
        dme_report(prompt, reason, llm_res, sp, recorder_start_index)
        logger.success("Detected DME during collecting dmf")
        # return -1
        return 0  # continue

    # record dmf
    record_dmf(sp, apk.apk_path, out_dir)
    return 1


def collect_all_dmf(apk: Apk, max_action_exec: int, out_dir: str, temperature: float, llm: Type[LLM],
                    model_visual: LLM.Model,
                    model_text: LLM.Model,
                    ):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    candidate_list = []
    for crud_type in ["Create", "Update", "Delete", "Read", "Search"]:
        # if dmf record exist, skip
        if not os.path.exists(os.path.join(out_dir, f"{crud_type}.json")):
            candidate_list.append(crud_type)
    if not candidate_list:
        logger.success("All dmf collected")
        return
    logger.success("Collecting dmf for {}", candidate_list)

    # find list item groups
    data_group = find_list_item(apk)
    if data_group is None:
        logger.error("No list item groups found, please initialize the app")
        sys.exit(0)

    logger.info("Save initial state...")
    AvdController.snapshot_save("initial_state")
    Device.logcat_clear()

    specific_cruds = specific_crud_meanings(apk.package_name, temperature, llm, model_visual)
    log_process = None
    for crud_type in candidate_list:
        # if dmf record exist, skip
        if os.path.exists(os.path.join(out_dir, f"{crud_type}.json")):
            continue

        # collect the crud type for 5 times
        for dt in range(5):
            temp = round(temperature + 0.1 * dt, 1)
            if temp > 1:
                temp = 1.0

            try:
                if EventCounter.is_full():
                    logger.info("Max event num reached: {}", EventCounter.cur_event_num)
                    return

                app_session = Device.start_app_session(apk.package_name, attach=True)
                time.sleep(3.5)
                
                # logcat_pid_ts.log
                logcat_path = os.path.normpath(
                    os.path.join(CollectRecorder.output_dir, f"logcat_{app_session.pid}_{timestamp()}.log"))
                log_process = Device.logcat_error(logcat_path)

                logger.debug("Collecting {} dmf with temperature={}", crud_type, temp)
                abbreviate_crud_type = crud_type[0]
                res = collect_dmf(apk, max_action_exec, abbreviate_crud_type,
                                  specific_cruds[abbreviate_crud_type],
                                  data_group,
                                  guidance.all_operation_workflows[abbreviate_crud_type],
                                  guidance.all_operation_guidance[abbreviate_crud_type],
                                  guidance.all_determine_stage_guidance[abbreviate_crud_type],
                                  temp, out_dir, llm, model_visual, model_text,
                                  )
                if res == 1:
                    break
            except KeyboardInterrupt:
                try:
                    logger.warning("KeyboardInterrupt")
                    choice = input("Input choice! 1: Exit, 2: Break, Other: Continue. Your choice: ")
                    if choice == "1":
                        sys.exit(0)
                    elif choice == "2":
                        break
                    else:
                        continue
                except KeyboardInterrupt:
                    sys.exit(0)
            except Exception as e:
                logger.exception(e)
            finally:
                AvdController.snapshot_load("initial_state")
                if log_process is not None:
                    log_process.terminate()
                time.sleep(5)


def dme_report2(crud_list: list[str], prompt: str, reason: str, full_res: str, output_dir: str):
    doc = [
        "# DME Report",
        "## Data manipulation type", "->".join(crud_list),
        "## Prompt", prompt,
        "## Reason", f">{reason}", full_res,
    ]
    output_path = unique_filepath(output_dir, "DME.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(doc))
    return output_path


def is_dmf_available(dmf: dict) -> bool:
    try:
        all_node_list = Device.get_node_list()
        crud_type = CRUD[dmf["dmf_type"]]
        data_group_match = NodeMathRule.from_dict(dmf["data_group_match"])
        actions: list[dict] = dmf["actions"]

        old_data_list = data_group_match.match_node_in_ui(all_node_list).children

        if crud_type != CRUD.Create and len(old_data_list) == 0:
            # no data to Search/Update/Delete
            return False

        if crud_type == CRUD.Create and len(old_data_list) >= 8:
            # too many data
            return False

        for action in actions:
            target = action["target"]
            if target is None:
                continue
            target_rule = NodeMathRule.from_dict(target)
            target_rule.match_node_in_ui(all_node_list)
            # first action target available
            return True

        return False
    except:
        return False


def reproduce_and_validate_dmf(dmf: dict, temperature: float, llm: Type[LLM], model_visual: LLM.Model,
                               model_text: LLM.Model, before_action_exec: Callable[[Action], None] | None = None,
                               after_action_exec: Callable[[int], None] | None = None, skip_validate: bool = False):
    crud_type = CRUD[dmf["dmf_type"]]
    data_group_match = NodeMathRule.from_dict(dmf["data_group_match"])
    actions: list[dict] = dmf["actions"]

    all_node_list = Device.get_node_list()

    #  old data list
    old_data_list = data_group_match.match_node_in_ui(all_node_list).children

    # replay actions
    target_data = None
    input_data = None
    for action in actions:
        all_node_list = Device.get_node_list()
        action_type = ActionType[action["type"]]

        if action_type == ActionType.InputTexts:
            # generate new input by llm
            edit_node_list = [node for node in all_node_list
                              if node.tag.endswith("EditText") or node.tag.endswith("AutoCompleteTextView")]

            if not edit_node_list:
                raise ValueError("No editable node found")

            cur_info = Device.active_app_info()
            if crud_type == CRUD.Search:
                target_data = old_data_list[-1]
                input_list = StatePropeller.gen_search_inputs_by_visual_llm(
                    target_data, crud_type.name, cur_info.package, cur_info.activity, edit_node_list,
                    llm, model_visual, temperature, None)
            else:
                input_list = StatePropeller.gen_inputs_by_visual_llm(crud_type.name, cur_info.package,
                                                                     cur_info.activity,
                                                                     edit_node_list, llm, model_visual, temperature,
                                                                     None)
            input_data = []
            for index, node in enumerate(edit_node_list):
                old_text = node.attributes["text"]
                new_text = input_list[index]
                # update history with current text
                input_data.append([old_text, new_text])

                input_action = Action(ActionType.InputTexts, node, new_text)
                if before_action_exec is not None:
                    before_action_exec(input_action)

                try:
                    input_action.execute()
                    EventCounter.add()
                    if after_action_exec is not None:
                        after_action_exec(1)
                except Exception as e:
                    logger.warning("InputTexts failed: {}", e)
                    Device.device.press("back")
                time.sleep(1)
            # update history with current text
            action["input"] = input_data
        else:
            if action["target"] == "None" or action["target"] is None:
                action_node = None
            else:
                action_node_match = NodeMathRule.from_dict(action["target"])
                action_node = action_node_match.match_node_in_ui(all_node_list)

            action = Action(action_type, action_node)
            if target_data is None and crud_type != CRUD.Search:
                target_data = action_related_data_node(old_data_list, action)

            if before_action_exec is not None:
                before_action_exec(action)
            action.execute()
            EventCounter.add()
            if after_action_exec is not None:
                after_action_exec(1)

            time.sleep(1.5)

    # new data list
    new_data_list = None
    if crud_type != CRUD.Read:
        all_node_list = Device.get_node_list()
        new_data_list = data_group_match.match_node_in_ui(all_node_list).children

    # input data (latest)
    if input_data is not None:
        input_data = {
            "ui_changes": [f"Change the text from '{old_text}' to '{new_text}'" for old_text, new_text in input_data],
            "text_changes": input_data
        }

    if skip_validate:
        return
    # validate
    return check_for_dme_by_llm(crud_type, target_data, old_data_list, new_data_list, input_data, llm, model_visual,
                                model_text, temperature)
