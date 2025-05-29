# LDMDroid

**LDMDroid** is an automated GUI testing tools for Android applications, designed to detect Data Manipulation Errors (DMEs) by leveraging the generalization capability of LLMs.

## Directory Structure

```
LDMDroid
 ├── start.py       The entry of LDMDroid, which accepts the tool parameters
 ├── evaluation     The evaluation results
 │   ├── bugs           The bugs detected by LDMDroid
 │   └── dmfs           The dmfs discovered by LDMDroid
 └── working        The working directory of LDMDroid
     ├── apks           The subject apps
     └── init           The initialization scripts
```

## Environment
- Android emulators (generic\_x86\_64, Android 9.0, 2GB RAM, 4-core CPU) 
- Python 3.12

## Installation

- Clone this repository:
```bash
git clone https://github.com/runnnnnner200/LDMDroid.git
cd LDMDroid
```

- Install the required Python packages:
```bash
poetry install
```

##  Usage

- Provide LLM API keys in `env.json` file: 
```json lines
{
    "zhipu": {
        "api_key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxx",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/"
    },
    "openai": {
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "base_url": "https://api.openai.com/v1/"
    }
}
```
Note that you need to choose the LLM module in `start.py`.

- Run LDMDroid on a target Android app:

```bash
python start.py -avd_serial emulator-5554 -avd_port 5554 -apk_path ./working/apks/easynotes.apk -max_minute 90
```
Here,
* `-avd_serial`: AVD device serial number.
* `-avd_port`: AVD port number.
* `-apk_path`: Path to the APK file.
* `-max_minute`: Max test duration in minutes.


