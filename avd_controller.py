from telnetlib import Telnet

from loguru import logger


class AvdController:
    host: str
    port: int
    tn: Telnet

    @classmethod
    def connect(cls, host: str = "localhost", port: int = 5554):
        cls.host = host
        cls.port = port
        cls.tn = Telnet(host, port)

        _, welcome_message = cls.read_response()
        token_path = welcome_message.splitlines()[-2].strip("'")
        with open(token_path) as f:
            token = f.read()

        cls.send_command(f"auth {token}")
        success, auth_res = cls.read_response()
        if not success:
            logger.error("AvdController auth failed: {}", auth_res)
            raise RuntimeError(auth_res)
        else:
            logger.success("AvdController auth success")

    @classmethod
    def disconnect(cls):
        cls.tn.close()

    @classmethod
    def send_command(cls, command: str):
        cls.tn.write(f"{command}\n".encode('ascii'))

    @classmethod
    def read_response(cls):
        index, match, _ = cls.tn.expect([b"OK", b"KO"], timeout=15)
        if match is None:
            return False, ""
        return index == 0, match.string.decode('ascii')

    @classmethod
    def snapshot_save(cls, name: str):
        cls.send_command(f"avd snapshot save {name}")
        success, snapshot_res = cls.read_response()
        if not success:
            logger.error("AvdController snapshot save failed: {}", snapshot_res)
        else:
            logger.success("AvdController snapshot save success")
        return success

    @classmethod
    def snapshot_load(cls, name: str):
        cls.send_command(f"avd snapshot load {name}")
        success, snapshot_res = cls.read_response()
        if not success:
            logger.error("AvdController snapshot load failed: {}", snapshot_res)
        else:
            logger.success("AvdController snapshot load success")
        return success

    @classmethod
    def snapshot_delete(cls, name: str):
        cls.send_command(f"avd snapshot delete {name}")
        success, snapshot_res = cls.read_response()
        if not success:
            logger.error("AvdController snapshot delete failed: {}", snapshot_res)
        else:
            logger.success("AvdController snapshot delete success")
        return success
