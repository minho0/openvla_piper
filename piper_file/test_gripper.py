#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PiPER gripper open/close ping over CAN (no ROS).
- ctrl: GripperCtrl(distance_mm, speed, code, set_zero)
  * code: 0x01 = position mode (일반)
  * set_zero: 0x00=일반 동작, 0xAE=제로 캘리브레이션
"""

import time, sys, argparse
from piper_sdk import C_PiperInterface, LogLevel

def clamp(v, lo, hi): return max(lo, min(hi, v))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can", default="can0")
    ap.add_argument("--open",  type=float, default=30.0, help="open distance (mm)")
    ap.add_argument("--close", type=float, default=5.0,  help="close distance (mm)")
    ap.add_argument("--speed", type=int,   default=1000, help="100~2000")
    ap.add_argument("--wait",  type=float, default=1.0)
    ap.add_argument("--repeat",type=int,   default=1)
    ap.add_argument("--zero",  action="store_true", help="send set_zero=0xAE once then exit")
    args = ap.parse_args()

    open_mm  = clamp(args.open,  0.0, 50.0)
    close_mm = clamp(args.close, 0.0, 50.0)
    speed    = max(100, min(2000, int(args.speed)))

    p = C_PiperInterface(
        can_name=args.can,
        judge_flag=False,
        can_auto_init=True,
        dh_is_offset=1,                 # S-V1.6-5 OK
        start_sdk_joint_limit=False,
        start_sdk_gripper_limit=False,
        logger_level=LogLevel.INFO,
        log_to_file=False,
        log_file_path=None,
    )
    p.ConnectPort()
    time.sleep(0.05)
    try:
        time.sleep(0.03)
        print("[*] Firmware:", p.GetPiperFirmwareVersion())
    except Exception:
        pass

    # 드라이버 enable (권장)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        try: p.EnableArm(7)
        except: pass
        time.sleep(0.2)

    # 제로 캘리브레이션 모드
    if args.zero:
        print("[*] Gripper ZERO set (0xAE) ...")
        p.GripperCtrl(0, speed, 0x01, 0xAE)     # set_zero=0xAE
        print("[OK] Sent zero-set command. Power-cycle 또는 초기화 절차는 장비 가이드 따르세요.")
        return

    # 일반 동작 (set_zero=0x00)
    for k in range(args.repeat):
        print(f"[{k+1}/{args.repeat}] OPEN -> {open_mm} mm")
        p.GripperCtrl(int(round(open_mm)),  speed, 0x01, 0x00)  # 마지막 0x00!
        time.sleep(args.wait)

        print(f"[{k+1}/{args.repeat}] CLOSE -> {close_mm} mm")
        p.GripperCtrl(int(round(close_mm)), speed, 0x01, 0x00)  # 마지막 0x00!
        time.sleep(args.wait)

    print("[OK] Gripper ping done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        sys.exit(1)
