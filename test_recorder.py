from os import path
from typing import TextIO

from PIL.Image import Image
from uiautomator2 import Session

from action import Action, ActionType
from device import Device
from util import ensure_dir, timestamp


class TestRecorder:
    output_dir: str
    image_output_dir: str
    output_f: TextIO
    index = 0
    last_logcat_index = 0
    logcat_file: str
    error_output_f: TextIO
    error_set: set[str] = set()

    def __init__(self, output_dir: str, logcat_file: str):
        self.output_dir = output_dir
        self.image_output_dir = path.join(output_dir, "images")
        ensure_dir(self.image_output_dir)
        self.output_f = open(path.join(output_dir, "events.txt"), "w", encoding="utf-8")
        self.error_output_f = open(path.join(output_dir, "errors.txt"), "w", encoding="utf-8")
        self.logcat_file = logcat_file

    def __del__(self):
        self.output_f.close()
        self.error_output_f.close()

    def _record_image(self, image: Image):
        image.save(path.join(self.image_output_dir, f"{self.index:04}.jpg"))

    def _record_msg(self, title: str, msg: str = None):
        ts = timestamp()
        if msg is None:
            self.output_f.write(f"[{ts}][{self.index:04}] {title}\n\n")
        else:
            self.output_f.write(f"[{ts}][{self.index:04}] {title}\n{msg}\n\n")
        self.output_f.flush()

    def record_action(self, action: Action):
        self.index += 1

        if action.node is None:
            node_str = "No target"
        else:
            node_str = f"{action.node.tag}({action.node.attributes})"

        if action.action_type == ActionType.InputTexts:
            msg = f"{action.action_type.name} | {action.input_content} | {node_str}"
        else:
            msg = f"{action.action_type.name} | {node_str}"

        self._record_msg("About to exec action", msg)
        if action.node is None:
            self._record_image(Device.screenshot_with_title(action.action_type.name))
        else:
            self._record_image(Device.draw_node_with_bound(action.node, action.action_type.name))

    def record_action_exception(self, error: Exception):
        self.index += 1
        self._record_msg("Action exception", str(error))

    def check_and_record_logcat_update(self, app_session: Session):
        pid = str(app_session.pid)
        updated = []
        with open(self.logcat_file, "r", encoding="utf-8") as f:
            index = self.last_logcat_index
            for index, line in enumerate(f):
                if index > self.last_logcat_index:
                    if pid not in line[:8]:
                        continue

                    continue_flag = False
                    for keyword in (
                            "Invalid ID 0x",
                            "error queuing buffer to SurfaceTexture",
                            "EGL_emulation",
                            "call to OpenGL ES API with no current context",
                            "Destroying unexpected ActionMode instance of TYPE_FLOATING"
                    ):
                        if keyword in line:
                            continue_flag = True
                            break
                    if continue_flag:
                        continue

                    updated.append(line[9:].strip())
            self.last_logcat_index = index

        if updated:
            unique_num = len(updated)
            # check duplicate
            updated_no_pid = [line.replace(pid, "<PID>") for line in updated]  # remove pid
            for i, line_no_pid in enumerate(updated_no_pid):
                if line_no_pid in self.error_set:
                    unique_num -= 1
                    updated[i] = "=\t" + updated[i]
                else:
                    updated[i] = "+\t" + updated[i]

            if unique_num <= 0:
                # duplicate
                self._record_msg("Duplicate Error Logcat")
                return False

            self.error_set.update(updated_no_pid)

            self._record_msg(f"Error Logcat updated(PID: {pid})", "\n".join(updated))
            self.error_output_f.write(f"[{self.index:04}]\nError Logcat updated({pid})\n" + "\n".join(updated) + "\n\n")

        return bool(updated)

    def record_dmf(self, dmf_type: str, state: str, dme_path: str = None):
        self.index += 1
        self._record_msg(f"Executing DMF", f"{dmf_type} | {state}")
        self._record_image(Device.screenshot_with_title(f"{dmf_type}:{state}"))
        if dme_path is not None:
            self.error_output_f.write(f"[{self.index:04}]\nDetect DME: {dme_path}\n\n")

    def record_app_session(self, app_session: Session):
        self._record_msg("App session", f"PID: {app_session.pid}")

    def record_app_stop(self):
        self.index += 1
        self._record_msg("App stopped")
        self._record_image(Device.screenshot_with_title("App stopped"))

    def record_app_not_active(self):
        self.index += 1
        self._record_msg("App was not active")
        self._record_image(Device.screenshot_with_title("App inactive"))

    def record_app_screen(self):
        self.index += 1
        self._record_image(Device.screenshot_with_title("Final screen"))
