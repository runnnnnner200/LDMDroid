import re
import shutil
import subprocess
import os

from loguru import logger
from lxml import etree


class Apk:
    apk_path: str
    package_name: str
    activities: dict[str, bool]
    static_string_set: set[str] | None = None

    def __init__(self, apk_path: str, aapt_path: str
                 , apktool_path: str | None = None, output_dir: str | None = None, cleanup: bool = False):
        result = subprocess.Popen([aapt_path, 'dump', 'xmltree', apk_path, 'AndroidManifest.xml'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, _ = result.communicate()

        if result.returncode != 0:
            raise Exception(f"Error running aapt: {result.stderr}")

        # parse manifest
        package_name = None
        activities = []
        inside_activity = False
        for line in stdout.splitlines():
            line = line.strip()

            if package_name is None and line.startswith('A: package='):
                package_name = re.search(r'A: package.*?="([^"]+)"', line).group(1)

            # into activity tag
            if line.startswith('E: activity'):
                inside_activity = True

            # find activity name
            if inside_activity and 'A: android:name' in line:
                activity_name = re.search(r'A: android:name.*?="([^"]+)"', line)
                if activity_name:
                    activities.append(activity_name.group(1))
                inside_activity = False  # out of activity tag

        if package_name is None:
            raise Exception("Could not find package name")

        self.apk_path = apk_path
        self.package_name = package_name
        self.activities = {name: False for name in activities}

        # extract apk
        if apktool_path is not None and output_dir is not None:
            output_dir = os.path.join(output_dir, package_name)

            if os.path.exists(output_dir):
                if cleanup:
                    logger.info("Cleaning up existing output directory: {}", output_dir)
                    shutil.rmtree(output_dir)
                    logger.info("Extracting apk: {}", output_dir)
                    subprocess.run(["java", "-jar", apktool_path, 'd', apk_path, '-o', output_dir])
                    logger.success("Apk extracted successfully")
                else:
                    logger.info("Cache already exists, skipping apk extraction")
            else:
                logger.info("Extracting apk: {}", output_dir)
                subprocess.run(["java", "-jar", apktool_path, 'd', apk_path, '-o', output_dir])
                logger.success("Apk extracted successfully")

            # extract strings
            res_path = os.path.join(output_dir, 'res')
            string_set = set()
            for root, dirs, files in os.walk(res_path):
                for file_name in files:
                    if file_name == 'strings.xml':
                        file_path = os.path.join(root, file_name)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            matches = re.findall('<string .*?>([^<]+)</string>', content)
                            string_set.update(matches)
                    elif file_name == 'arrays.xml':
                        file_path = os.path.join(root, file_name)
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            items = [item.text for item in etree.fromstring(content).findall('.//string-array/item')]
                            string_set.update(items)

            self.static_string_set = string_set

    def __str__(self) -> str:
        return f"Apk(apk_path={self.apk_path}, package_name={self.package_name}, activity_count={len(self.activities)})"
