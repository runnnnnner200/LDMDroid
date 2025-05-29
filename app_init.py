import time

import uiautomator2 as u2

from device import Device


def run_init(script_path: str, clear_data: bool = False, device_serial: str = None, d: u2.Device = None):
    if d is None:
        d = u2.connect(device_serial)
    d.wait_timeout = 3
    d.shell("pm disable-user com.android.inputmethod.latin")

    with open(script_path, 'r') as f:
        package = next(f).strip()
        print(package)

        if clear_data:
            print('clear app data')
            d.app_clear(package)
            time.sleep(1)

        d.app_start(package, wait=True, use_monkey=True)
        print("wait 4s for app init")
        time.sleep(4)
        for action in f:
            if action.startswith('#') or not action.strip():
                continue
            try:
                tmp = action.strip().split(',', maxsplit=1)
                action_type, param = tmp if len(tmp) == 2 else [tmp[0], '']
                print(action_type, param)
                if action_type == 'Click':
                    d.xpath(param).click()
                elif action_type == 'LongClick':
                    target = d.xpath(param)
                    x, y = target.center()
                    Device.device.long_click(x, y, 0.6)
                elif action_type == 'Input':
                    d.send_keys(param)
                elif action_type == 'Back':
                    d.press('back')
                elif action_type == 'CleanFile':
                    d.shell('mkdir -p /storage/emulated/0/Download')
                    d.shell('rm -rf /storage/emulated/0/Download/*')
                elif action_type == 'Swipe':
                    d.shell(f"input swipe {tmp[1].replace(',', ' ')}")
            except:
                print('unable to exec action:', action.strip())
            time.sleep(2)

    print('init done')

