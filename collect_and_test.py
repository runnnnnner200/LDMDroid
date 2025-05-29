import json
import os
import random
import time
from typing import Type

from loguru import logger
from uiautomator2 import UiObjectNotFoundError

from action import ActionType, Action
from apk import Apk
from avd_controller import AvdController
from collect_recorder import CollectRecorder
from device import Device
from dmf import collect_all_dmf, is_dmf_available, reproduce_and_validate_dmf, dme_report2
from llm import Zhipu, LLM
from random_policy import RandomPolicy
from state_propeller import StatePropeller
from test_recorder import TestRecorder
from util import ensure_dir, basic_init, EventCounter, timestamp
from uiautomator2 import Device as D
from app_init import run_init


def script(apk: Apk, device: D):
    script_path = os.path.normpath(os.path.join("./working/init", apk.package_name + ".txt"))
    if not os.path.exists(script_path):
        logger.error("Script file not found: {}", script_path)
        return

    time.sleep(3)
    logger.info("Restore to empty snapshot...")
    AvdController.snapshot_load("empty")  # restore to empty snapshot if empty snapshot exists
    time.sleep(3)

    Device.install_app(apk.apk_path)
    run_init(d=device, script_path=script_path)


def collect(apk: Apk, max_action_exec: int, base_out_dir: str, temperature: float,
            llm: Type[LLM], model_visual: LLM.Model, model_text: LLM.Model):
    cur_output_dir = f"{base_out_dir}/collect"
    dmf_output_dir = f"working/dmf/{apk.package_name}"
    ensure_dir(cur_output_dir)
    ensure_dir(dmf_output_dir)

    # clear dmf files
    for file in os.listdir(dmf_output_dir):
        if file.endswith(".json"):
            os.remove(os.path.join(dmf_output_dir, file))

    Device.logcat_clear()
    Device.ime_set(enable=False)
    CollectRecorder.init(cur_output_dir)
    collect_all_dmf(apk, max_action_exec, dmf_output_dir, temperature, llm, model_visual, model_text)

    # inspect and reorganize logcat
    final_log = []
    file_list = []
    for file in os.listdir(cur_output_dir):
        if file.startswith("logcat_"):
            file_list.append(file)
            pid = file.split("_")[1]
            new_logs = []
            with open(os.path.join(cur_output_dir, file), "r") as f:
                for line in f:
                    if pid in line[:8]:
                        new_logs.append(line[9:].strip())
            if new_logs:
                final_log += new_logs
                final_log.append("==============")

    with open(os.path.join(cur_output_dir, "raw_logcat_errors.txt"), "w") as f:
        f.write("\n".join(final_log))

    for file in file_list:
        os.remove(os.path.join(cur_output_dir, file))


def test(base_out_dir: str, apk: Apk, max_minute: int, temperature: float, llm: Type[LLM],
         model_visual: LLM.Model, model_text: LLM.Model, start_time: float, skip_validate: bool = False
         ):
    output_dir = f"{base_out_dir}/test"
    ensure_dir(output_dir)

    logger.debug("Output dir: {}", output_dir)
    logger.add(
        sink=os.path.join(output_dir, "log.txt"),
        format="{time:HH:mm:ss} {level} {message}",
    )

    dmf_dir = os.path.normpath(f"working/dmf/{apk.package_name}")
    dmf_names = [file[0:-5] for file in os.listdir(dmf_dir) if file.endswith(".json")]
    if not dmf_names:
        logger.error("No dmf files found!")
    else:
        logger.info("DMF list: {}", ", ".join(dmf_names))

    dmf_list: list[dict] = []
    for dmf_name in dmf_names:
        with open(os.path.join(dmf_dir, dmf_name + ".json"), "r") as f:
            dmf_list.append(json.load(f))

    Device.ime_set(enable=False)

    Device.logcat_clear()
    logger.success("Logcat cleared!")

    logcat_path = os.path.normpath(os.path.join(output_dir, "logcat.log"))
    logcat_process = Device.logcat_error(logcat_path)
    logger.debug("Crash logcat path: {}", logcat_path)

    recorder = TestRecorder(output_dir, logcat_path)
    random_weight = {
        "weight_click": 40,
        "weight_long_click": 30,
        "weight_scroll": 10,
        "weight_edit": 30,
        "weight_back": 5,
    }
    random_policy = RandomPolicy(**random_weight, package_name=apk.package_name)
    logger.debug("Random policy weight: {}", random_weight)

    app_session = Device.start_app_session(apk.package_name, attach=True)
    recorder.record_app_session(app_session)
    Device.device.app_wait(apk.package_name, 3, front=True)
    logger.success("App started!")
    time.sleep(2)

    running_minute = round((time.time() - start_time) / 60)

    def after_action_exec(the_count: int):
        nonlocal recorder, running_minute
        EventCounter.add(the_count)
        logger.info("Action executed: {}/{}", EventCounter.cur_event_num, EventCounter.max_event_num)
        logger.info("Time: {}/{}", running_minute, max_minute)

    def before_action_exec(the_action: Action):
        recorder.record_action(the_action)
        logger.info("Action executing: {}", the_action.action_type.name)

    while not EventCounter.is_full():
        running_minute = round((time.time() - start_time) / 60)
        if running_minute >= max_minute:
            break

        if recorder.check_and_record_logcat_update(app_session):
            logger.warning("Error Logcat updated!")

        # detecting if the app is running
        if not app_session.running():
            logger.error("App session stop! Restart...")
            recorder.record_app_stop()
            app_session = Device.start_app_session(apk.package_name)
            recorder.record_app_session(app_session)
            time.sleep(2)
            continue

        cur_info = Device.active_app_info()
        # com.android.packageinstaller Grant
        if cur_info.package == "com.android.packageinstaller":
            logger.info("Grant dialog detected!")
            try:
                Device.device.xpath(
                    '//*[@resource-id="com.android.packageinstaller:id/permission_allow_button"]').click(timeout=2)
                logger.success("Granted!")
                time.sleep(1)
            except UiObjectNotFoundError:
                logger.warning("Grant button not found!")
            continue

        # detecting if the app is active
        if cur_info.package != apk.package_name:
            logger.warning("App is not active! Activate... ({}!={})", cur_info.package, apk.package_name)
            recorder.record_app_not_active()
            Device.device.press("back")
            Device.device.app_start(apk.package_name)  # only try to activate the app
            app_session = Device.start_app_session(apk.package_name, attach=True)
            recorder.record_app_session(app_session)
            time.sleep(2)
            # no continue

        # get dmf candidate list
        dmf_candidate_list = [dmf for dmf in dmf_list if is_dmf_available(dmf)]
        if dmf_candidate_list:
            logger.info("DMF candidate list: {}", ", ".join([dmf["dmf_type"] for dmf in dmf_candidate_list]))
            recorder._record_msg("DMF candidate list", ", ".join([dmf["dmf_type"] for dmf in dmf_candidate_list]))
        else:
            logger.info("No DMF candidate found")

        if dmf_candidate_list:
            choice = random.choice(["random"] * 2 + ["dmf"] * 3)
        else:
            choice = "random"
        logger.info("Choice: {}", choice)

        # gen and exec action
        try:
            if choice == "random":
                action = random_policy.gen_random_action(Device.get_node_list())
                if action.action_type == ActionType.InputTexts:
                    action.input_content = StatePropeller.gen_inputs_by_visual_llm(
                        "", cur_info.package, cur_info.activity, [action.node], llm, model_visual, temperature, None
                    )[0]
                before_action_exec(action)
                action.execute(direct=True)
                after_action_exec(1)
                time.sleep(1.5)
            else:
                dmf = random.choice(dmf_candidate_list)

                logger.success("Chosen DMF: {}", dmf["dmf_type"])
                recorder.record_dmf(dmf["dmf_type"], "Start")

                reproduce_res = reproduce_and_validate_dmf(
                    dmf, temperature, llm, model_visual, model_text,
                    before_action_exec=before_action_exec,
                    after_action_exec=after_action_exec,
                    skip_validate=skip_validate
                )

                logger.success("{} DMF executed!", dmf["dmf_type"])
                recorder.record_dmf(dmf["dmf_type"], "Over")

                if not skip_validate:
                    op_invalid, reason, llm_res, prompt = reproduce_res
                    if op_invalid:
                        output_path = dme_report2([dmf["dmf_type"]], prompt, reason, llm_res, output_dir)
                        logger.success("DME found: {}", output_path)
                        recorder.record_dmf(dmf["dmf_type"], "Detect Error", output_path)
                        continue

        except KeyboardInterrupt:
            logger.warning("Stop by KeyboardInterrupt")
            break
        except Exception as e:
            logger.exception(e)
            recorder.record_action_exception(e)

    if EventCounter.is_full():
        logger.success("Max event number reached! {}/{}", EventCounter.cur_event_num, EventCounter.max_event_num)
    elif running_minute >= max_minute:
        logger.warning("Max time reached! {}/{}", running_minute, max_minute)

    recorder.record_app_screen()
    logcat_process.terminate()
    if recorder.check_and_record_logcat_update(app_session):
        logger.warning("Crash updated!")

    app_session.close()
    Device.device.press("home")


def collect_and_test(avd_serial: str, avd_port: int, apk_path: str, max_action_exec: int, temperature: float,
                     event_num, max_minute: int, skip_validate: bool, llm: Type[LLM], model_visual: LLM.Model,
                     model_text: LLM.Model):
    basic_init()

    apk_path = os.path.normpath(apk_path)
    if os.name == "nt":
        aapt_path = "working/aapt.exe"
    else:
        aapt_path = "working/aapt"
    apk = Apk(
        apk_path,
        aapt_path=aapt_path,
        apktool_path="working/apktool_2.10.0.jar",
        output_dir="working/cache/",
        cleanup=False,
    )

    Device.connect(device_serial=avd_serial)
    Device.ime_set(enable=False)
    logger.success("Device connected!")
    AvdController.connect(port=avd_port)

    base_out_dir = f"working/output/collect_and_test/{apk.package_name}/{timestamp()}"
    EventCounter.init(event_num)

    logger.info("Run init script...")
    script(apk, Device.device)

    start_time = time.time()

    logger.info("Collect DMF...")
    collect(apk, max_action_exec, base_out_dir, temperature, llm, model_visual, model_text)

    time.sleep(3)

    logger.info("Test with DMF...")
    test(base_out_dir, apk, max_minute, temperature, llm, model_visual, model_text, start_time, skip_validate)
