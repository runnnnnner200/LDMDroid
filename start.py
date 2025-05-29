import argparse
import os
import time

from loguru import logger

from apk import Apk
from avd_controller import AvdController
from collect_and_test import collect_and_test
from device import Device
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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-avd_serial", type=str, required=False, default="emulator-5554")
    parser.add_argument("-avd_port", type=int, required=False, default=5554)
    parser.add_argument("-apk_path", type=str, required=True)
    parser.add_argument("-temperature", type=float, required=False, default=0.0, help="temperature of llm")
    parser.add_argument(
        "-max_action_times", type=int, required=False, default=10, help="Maximum length of action history"
    )
    parser.add_argument("-event_num", type=int, required=False, default=200, help="How many events are in a test case")
    parser.add_argument("-max_minute", type=int, required=False, default=60, help="Maximum test time in minutes")
    parser.add_argument("-skip_validate", type=bool, required=False, default=False, help="Skip DMF validation")

    args = parser.parse_args()

    collect_and_test(
        avd_serial=args.avd_serial,
        avd_port=args.avd_port,
        apk_path=args.apk_path,
        max_action_exec=args.max_action_times,
        temperature=args.temperature,
        event_num=args.event_num,
        max_minute=args.max_minute,
        skip_validate=args.skip_validate,
    )


if __name__ == "__main__":
    main()
