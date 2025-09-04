# read_joint_quick.py
import time
from piper_sdk import C_PiperInterface, LogLevel

if __name__ == "__main__":
    piper = C_PiperInterface(
        can_name="can0",          # 또는 "can_piper"
        judge_flag=False,         # 비공식 CAN이면 False 유지
        can_auto_init=True,       # 인스턴스 생성 시 자동 open
        dh_is_offset=1,           # 최신 펌웨어(보통 1)
        start_sdk_joint_limit=False,
        start_sdk_gripper_limit=False,
        logger_level=LogLevel.INFO,
        log_to_file=False,
        log_file_path=None,
    )
    piper.ConnectPort()           # 수신/송신 스레드 시작
    time.sleep(0.025)             # 최초 프레임 안정화 대기 (권장)

    while True:
        print(piper.GetArmJointMsgs())  # 6개 조인트 각(rad) 읽힘
        time.sleep(0.005)               # 200Hz