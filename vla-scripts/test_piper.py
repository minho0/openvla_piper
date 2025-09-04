#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenVLA(8bit) → 7-DoF (EE-local Δpose6 + gripper_mm) → IK(j1..6 rad) → V2 ticks → PiPER(CAN, no ROS)
- Terminal instruction 입력 지원 (미지정 시)
- E-Stop: Ctrl+C 또는 'e' 키 (best-effort: 속도=0, 현재자세 재명령)

조인트 스케일: rad → milli-deg(틱) = rad * 180/pi * 1000
그리퍼 스케일: mm → m → um(틱)   = (mm/1000) * 1e6
"""

import re, math
import argparse, json, time, sys, signal, threading, tty, termios, os, math
from pathlib import Path
from typing import List, Optional
import numpy as np

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from transforms3d.euler import euler2mat, mat2euler
from ikpy.chain import Chain

# PiPER V2 SDK
from piper_sdk import C_PiperInterface_V2, LogLevel  # noqa: F401 (LogLevel for completeness)

# --------------------------
# E-Stop globals
# --------------------------
ESTOP_REQUESTED = False
_last_q_ticks: Optional[List[int]] = None  # 현재 관절(틱) 저장

def request_estop():
    global ESTOP_REQUESTED
    ESTOP_REQUESTED = True
    print("\n[ESTOP] Requested! Attempting to stop safely...")

def sigint_handler(signum, frame):
    request_estop()

# --------------------------
# OpenVLA prompt
# --------------------------
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
def get_openvla_prompt(instruction: str, model_path: str) -> str:
    if "v01" in model_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

# --------------------------
# Limits (rad) + helpers
# --------------------------
JOINT_LIMITS = [
    (-2.618,  2.618),   # j1
    ( 0.000,  3.140),   # j2
    (-2.967,  0.000),   # j3
    (-1.745,  1.745),   # j4
    (-1.220,  1.220),   # j5
    (-2.0944, 2.0944),  # j6
]
def clamp(v, lo, hi): return max(lo, min(hi, v))
def clamp_joints(q: List[float]) -> List[float]:
    return [clamp(float(q[i]), JOINT_LIMITS[i][0], JOINT_LIMITS[i][1]) for i in range(6)]

RAD2MDEG = 1000.0 * 180.0 / math.pi  # ≈ 57295.7795

def rad_to_tick(rad: float) -> int:
    return int(round(rad * RAD2MDEG))

def grip_mm_to_tick(mm: float) -> int:
    # OpenVLA grip_mm → m → um(틱)
    meters = float(mm) / 1000.0
    return int(round(meters * 1_000_000))


# JOINT_NAMES 전역 추가
def debug_chain(chain):
    print("\n[IK] Links dump (idx, is_joint, name):")
    for i, l in enumerate(chain.links):
        print(f"  {i:2d} | {getattr(l,'is_joint',False)} | {l.name}")
    print()

def resolve_joint_names(chain) -> list:
    """URDF에서 6개 조인트 이름을 결정."""
    link_names = [l.name for l in chain.links]
    candidates = [
        ["joint1","joint2","joint3","joint4","joint5","joint6"],
        ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"],
        ["J1","J2","J3","J4","J5","J6"],
        ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
         "wrist_1_joint","wrist_2_joint","wrist_3_joint"],
    ]
    for cand in candidates:
        if all(n in link_names for n in cand):
            return cand
    # 폴백: joint 타입 링크 앞의 6개
    auto = [l.name for l in chain.links if getattr(l, "is_joint", False)][:6]
    if len(auto) != 6:
        raise RuntimeError(f"URDF에서 6개 조인트를 결정하지 못했습니다. joint-count={len(auto)}")
    return auto

def set_active_mask_for_six(chain, joint_names):
    """6개 조인트 이름만 True로. is_joint 여부는 무시(ikpy가 잘못 표기하는 케이스 회피)."""
    jset = set(joint_names)
    mask = [(l.name in jset) for l in chain.links]   # <-- is_joint 조건 제거
    chain.active_links_mask = mask
    n_active = sum(mask)
    if n_active != 6:
        raise RuntimeError(f"active_links_mask={n_active} (≠6). joint_names={joint_names}")

# --------------------------
# IK / FK (URDF)
# --------------------------

def build_chain(urdf_path: str) -> Chain:
    # chain = Chain.from_urdf_file(
    #     urdf_path,
    #     base_elements=["base_link"]
    # )
    # # 링크별 활성 마스크: 조인트만 True, 나머지 False
    # mask = []
    # for lnk in chain.links:
    #     is_joint = getattr(lnk, "is_joint", False)
    #     name = getattr(lnk, "name", "")
    #     # 확실히 제외해야 하는 fixed 링크들
    #     if name in ("Base link", "base_link", "joint6_to_gripper_base", "tool0", "ee_link"):
    #         mask.append(False)
    #     else:
    #         mask.append(bool(is_joint))
    # # joint1..joint6만 남기고 나머지 조인트는 끄고 싶다면 여기를 추가:
    # JOINT_SET = set(["joint1","joint2","joint3","joint4","joint5","joint6"])
    # for i, lnk in enumerate(chain.links):
    #     if getattr(lnk, "is_joint", False) and lnk.name not in JOINT_SET:
    #         mask[i] = False

    # chain.active_links_mask = mask
    # return chain
    return Chain.from_urdf_file(urdf_path, base_elements=["base_link"])


def fk_from_q(chain: Chain, q6):
    """
    q6: [j1..j6] (rad)
    ikpy는 chain.links 길이만큼 각도를 주길 원하므로,
    joint1..6 인덱스를 찾아서 해당 위치에 q6를 꽂아줌.
    """
    names = [lnk.name for lnk in chain.links]
    q_all = [0.0] * len(chain.links)
    for i, jn in enumerate(JOINT_NAMES):
        try:
            idx = names.index(jn)
        except ValueError:
            raise RuntimeError(f"URDF/chain에 '{jn}' 조인트가 없습니다. names={names}")
        q_all[idx] = float(q6[i])
    return chain.forward_kinematics(q_all)

def ik_to_q(chain: Chain, T_target, q_init_from_curr=None):
    """
    inverse_kinematics_frame()은 전체 길이의 초기값을 받음.
    - q_init_from_curr: [j1..j6] (rad)
    반환도 [j1..j6] 순서로 뽑아서 리턴.
    """
    names = [lnk.name for lnk in chain.links]
    q_init_full = [0.0] * len(chain.links)
    if q_init_from_curr is not None:
        for i, jn in enumerate(JOINT_NAMES):
            try:
                idx = names.index(jn)
            except ValueError:
                raise RuntimeError(f"URDF/chain에 '{jn}' 조인트가 없습니다. names={names}")
            q_init_full[idx] = float(q_init_from_curr[i])

    q_all = chain.inverse_kinematics_frame(
        T_target,
        initial_position=q_init_full,
        max_iter=200,
        orientation_mode="all"  
    )

    q6 = []
    for jn in JOINT_NAMES:
        try:
            idx = names.index(jn)
        except ValueError:
            raise RuntimeError(f"URDF/chain에 '{jn}' 조인트가 없습니다. names={names}")
        q6.append(float(q_all[idx]))
    return q6

def compose_local_delta(T_current: np.ndarray, dx, dy, dz, dr, dp, dyaw, axes='sxyz') -> np.ndarray:
    """EE 로컬 Δpose 적용: T_target = T_current · ΔT_local"""
    R_delta = euler2mat(dr, dp, dyaw, axes=axes)
    Delta = np.eye(4); Delta[:3,:3] = R_delta; Delta[:3,3] = np.array([dx,dy,dz], dtype=float)
    return T_current @ Delta

# --------------------------
# OpenVLA (8-bit only)
# --------------------------
def load_model_8bit(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for 8-bit inference.")
    qcfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        quantization_config=qcfg,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return processor, model

# --------------------------
# Image loader
# --------------------------
def load_image(image_path: str = None, image_url: str = None) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")
    if image_url:
        import requests
        return Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    raise ValueError("Provide either --image_path or --image_url.")

# --------------------------
# V2 helpers (feedback + estop)
# --------------------------

_JOINT_NAME_SETS = [
    ("joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"),
    ("Joint_1","Joint_2","Joint_3","Joint_4","Joint_5","Joint_6"),
    ("joint1","joint2","joint3","joint4","joint5","joint6"),
    ("j1","j2","j3","j4","j5","j6"),
    ("q1","q2","q3","q4","q5","q6"),
]
_JOINT_REGEX = re.compile(r"Joint[_\s]?([1-6])\s*:\s*(-?\d+(?:\.\d+)?)")


def _unit_to_rad(vec6):
    m = max(abs(float(x)) for x in vec6)
    if m > 1000:               # mdeg 추정
        return [float(x)/RAD2MDEG for x in vec6]
    elif m > 10:               # deg 추정
        return [math.radians(float(x)) for x in vec6]
    else:                      # rad 추정
        return [float(x) for x in vec6]


def _extract_six_from_obj(msg):
    """ArmJoint/ArmJointCtrl/리스트/딕트/문자열 repr 대응 → [j1..j6] (rad)"""
    if msg is None:
        return None

    # 0) 리스트/튜플/넘파이
    try:
        as_list = list(msg)
        if len(as_list) >= 6:
            return _unit_to_rad(as_list[:6])
    except TypeError:
        pass

    # 1) dict
    if isinstance(msg, dict):
        # (a) 평범한 키들
        collected = []
        for names in _JOINT_NAME_SETS:
            cand = []
            for n in names:
                if n in msg: cand.append(msg[n])
            if len(cand) >= 6:
                return _unit_to_rad(cand[:6])
        # (b) 중첩 구조 펼치기
        flat = []
        def walk(x):
            if isinstance(x, dict):
                for v in x.values(): walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x: walk(v)
            else:
                try: flat.append(float(x))
                except: pass
        walk(msg)
        if len(flat) >= 6:
            return _unit_to_rad(flat[:6])

    # 2) 객체: 바로 필드로 보유
    for names in _JOINT_NAME_SETS:
        if all(hasattr(msg, n) for n in names):
            return _unit_to_rad([getattr(msg, n) for n in names])

    # 3) ArmJoint{ joint_state: ... } / ArmJointCtrl{ joint_ctrl: ... }
    for child_name in ("joint_state", "joint_ctrl"):
        if hasattr(msg, child_name):
            child = getattr(msg, child_name)
            # 3-1) 자식이 필드로 보유
            for names in _JOINT_NAME_SETS:
                if all(hasattr(child, n) for n in names):
                    return _unit_to_rad([getattr(child, n) for n in names])
            # 3-2) 자식이 벡터스러운 경우
            try:
                as_list = list(child)
                if len(as_list) >= 6:
                    return _unit_to_rad(as_list[:6])
            except TypeError:
                pass
            # 3-3) 자식 repr에서 정규식 파싱
            s = str(child)
            out = [None]*6
            for m in _JOINT_REGEX.finditer(s):
                i = int(m.group(1))-1; val = float(m.group(2))
                if 0 <= i < 6: out[i] = val
            if all(v is not None for v in out):
                return _unit_to_rad(out)

    # 4) 마지막: 자기 repr에서 정규식 파싱
    s = str(msg)
    out = [None]*6
    for m in _JOINT_REGEX.finditer(s):
        i = int(m.group(1))-1; val = float(m.group(2))
        if 0 <= i < 6: out[i] = val
    if all(v is not None for v in out):
        return _unit_to_rad(out)

    return None

def try_read_current_joints_v2(p, timeout_s=3.0, poll_hz=50.0, verbose=True):
    """
    현재 6개 관절값을 읽어서 반환.
    - 1순위: GetArmJointCtrl()  (셋포인트)
    - 2순위: GetArmJointMsgs()  (실측 피드백)
    - (참고) GetArmStatus()는 문자열/상태라 보통 관절값 없음. 로그만 찍음.

    반환: list[6] (float), 단위는 그대로(라디안/도/mdeg 감지해서 로그로 알려줌)
    """
    import time
    import math

    def _is_vec6(x):
        try:
            return x is not None and len(x) == 6 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in x)
        except Exception:
            return False

    def _unit_hint(v6):
        """대략 단위 추정 (로그용)"""
        m = max(abs(float(x)) for x in v6)
        if m > 1000:
            return "mdeg(?)"
        elif m > 10:
            return "deg(?)"
        else:
            return "rad(?)"

    def _fmt(x, n=3):
        try:
            return "[" + ", ".join(f"{float(v):.3f}" for v in x) + "]"
        except Exception:
            return str(x)

    # 당신이 이미 갖고 있는 추출기 활용하되, 실패 시 로그 남김
    def _safe_extract(tag, obj):
        if verbose:
            raw = getattr(obj, "__dict__", obj)
            print(f"[{tag}] type={type(obj).__name__} raw={raw}")
        try:
            vals = _extract_six_from_obj(obj)  # <-- 당신이 정의한 추출 함수
            if _is_vec6(vals):
                if verbose:
                    print(f"[{tag}] vec6={_fmt(vals)} (unit { _unit_hint(vals) })")
                return vals
            else:
                if verbose:
                    print(f"[{tag}] extract returned invalid vec6: {vals}")
                return None
        except Exception as e:
            if verbose:
                print(f"[{tag}] EXCEPTION in _extract_six_from_obj: {repr(e)}")
            return None

    deadline = time.time() + timeout_s
    period = max(1.0 / max(poll_hz, 1.0), 0.001)
    i = 0

    while time.time() < deadline:
        i += 1
        if verbose:
            print(f"\n--- poll #{i} ---")

        # 1) 실측 피드백(강력 추천)
        try:
            get_msgs = getattr(p, "GetArmJointMsgs", None)
            if get_msgs is not None:
                mm = get_msgs()
                vals = _safe_extract("MSGS", mm)
                if _is_vec6(vals):
                    return vals
            else:
                if verbose:
                    print("[MSGS] SKIP: p.GetArmJointMsgs not available")
        except Exception as e:
            if verbose:
                print(f"[MSGS] call EXCEPTION: {repr(e)}")

        # 2) 컨트롤(셋포인트) — TEACHING이면 대부분 0
        try:
            m = p.GetArmJointCtrl()
            vals = _safe_extract("CTRL", m)
            if _is_vec6(vals):
                return vals
        except Exception as e:
            if verbose:
                print(f"[CTRL] call EXCEPTION: {repr(e)}")

        # 3) 참고용 상태 문자열
        try:
            st = p.GetArmStatus()
            if verbose:
                txt = st if isinstance(st, str) else getattr(st, "__dict__", st)
                print(f"[STAT] {txt}")
        except Exception as e:
            if verbose:
                print(f"[STAT] call EXCEPTION: {repr(e)}")

        time.sleep(period)

    if verbose:
        print("[TIMEOUT] failed to read a valid 6-joint vector.")
    return None


def soft_estop_v2(p: Optional[C_PiperInterface_V2], q_ticks: Optional[List[int]]):
    """Best-effort: 속도스케일 0 + 현재자세 재명령"""
    try:
        if p:
            p.MotionCtrl_2(0x01, 0x01, 0, 0x00)  # speed=0
            if q_ticks and len(q_ticks) == 6:
                p.JointCtrl(q_ticks[0], q_ticks[1], q_ticks[2], q_ticks[3], q_ticks[4], q_ticks[5])
            print("[ESTOP] Hold current pose command sent (speed=0).")
    except Exception as e:
        print(f"[ESTOP] failed: {e}")

# --------------------------
# Non-blocking key listener ('e' to estop)
# --------------------------
def start_estop_key_listener(estop_key: str = 'e'):
    if not sys.stdin.isatty():
        return None
    def worker():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not ESTOP_REQUESTED:
                ch = os.read(fd, 1).decode(errors='ignore')
                if ch.lower() == estop_key:
                    request_estop()
                    break
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return th

# --------------------------
# Main
# --------------------------
def main():
    signal.signal(signal.SIGINT, sigint_handler)

    ap = argparse.ArgumentParser(description="OpenVLA(8bit)->PiPER(V2) one-shot with E-Stop & terminal prompt")
    ap.add_argument("--model_path", type=str, default="openvla/openvla-7b")
    ap.add_argument("--instruction", type=str, default=None, help="If omitted, will prompt in terminal")
    ap.add_argument("--image_path", type=str, default=None)
    ap.add_argument("--image_url",  type=str, default=None)
    ap.add_argument("--unnorm_key", type=str, default="bridge_orig")

    ap.add_argument("--can",   type=str, default="can0", help="CAN interface (e.g., can0)")
    ap.add_argument("--urdf",  type=str, default="piper_description.urdf")
    ap.add_argument("--euler_axes", type=str, default="sxyz")
    ap.add_argument("--q_init", nargs=6, type=float, help="Fallback joints(rad) if feedback read fails")
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--save_json", type=str, default=None)
    ap.add_argument("--print_prompt", action="store_true")
    ap.add_argument("--estop_key", type=str, default="e", help="Press this key to E-Stop (default: 'e')")
    ap.add_argument("--speed", type=int, default=30, help="Speed scale 0..100 for MotionCtrl_2")

    # 수동 7DoF 입력 (OpenVLA 우회)
    ap.add_argument("--action", nargs=7, type=float,
                    help="[dx dy dz droll dpitch dyaw grip_mm] (EE-local)")
    ap.add_argument("--action-file", type=str,
                    help="JSON 파일 경로 (list[7] 또는 {'action':[7]})")

    args = ap.parse_args()

    # 수동 action 로더
    def load_action_from_args(args):
        if args.action and len(args.action) == 7:
            return [float(x) for x in args.action]
        if args.action_file:
            with open(args.action_file, "r") as f:
                data = json.load(f)
            arr = data["action"] if isinstance(data, dict) else data
            if len(arr) != 7:
                raise ValueError("action must have 7 values")
            return [float(x) for x in arr]
        return None

    # 0) Prompt for instruction if missing (OpenVLA 경로에서만 쓰임)
    if not args.instruction and load_action_from_args(args) is None:
        try:
            print("\n===============================")
            print("🤖  OpenVLA → PiPER V2 (8bit, one-shot)")
            print("⌨️   Type your instruction (ENTER to confirm)")
            print("🛑  E-Stop: press 'e' or Ctrl+C anytime")
            print("===============================\n")
            args.instruction = input("👉 Instruction: ").strip()
            if not args.instruction:
                raise SystemExit("No instruction provided. Exiting.")
        except EOFError:
            raise SystemExit("No instruction provided. Exiting.")

    print(f"[*] device: cuda (8-bit only)")
    print(f"[*] model: {args.model_path}")
    if args.instruction:
        print(f"[*] instruction: {args.instruction}")

    # 키 리스너
    start_estop_key_listener(args.estop_key)

    # 1) action 결정 (수동 입력 우선)
    action = load_action_from_args(args)
    if action is None:
        # OpenVLA 경로 사용
        image = load_image(args.image_path, args.image_url)
        processor, vla = load_model_8bit(args.model_path)
        prompt = get_openvla_prompt(args.instruction, args.model_path)
        if args.print_prompt:
            print(f"\n[Prompt]\n{prompt}\n")
        inputs = processor(prompt, image).to("cuda", dtype=torch.float16)
        if ESTOP_REQUESTED: raise SystemExit("[ESTOP] Before inference.")
        t0 = time.time()
        action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
        dt = time.time() - t0
        if hasattr(action, "tolist"):
            action = action.tolist()
        action = [float(x) for x in action]
        if len(action) != 7:
            raise RuntimeError(f"Expected 7-DoF action, got len={len(action)}")
        print(f"[OK] Inference: {dt:.3f}s")
    else:
        print("[OK] Using manual action (skip OpenVLA).")

    dx, dy, dz, dr, dp, dyaw, grip_mm = action
    print("Action (EE-local): [dx, dy, dz, droll, dpitch, dyaw, grip_mm]")
    print([round(v, 5) for v in action])

    # 2) IK pipeline
    chain = build_chain(args.urdf)

    debug_chain(chain)
    global JOINT_NAMES
    JOINT_NAMES = resolve_joint_names(chain)
    print("[IK] JOINT_NAMES:", JOINT_NAMES)
    set_active_mask_for_six(chain, JOINT_NAMES)



    # V2 connect + enable + current joints
    v2 = C_PiperInterface_V2(args.can)
    v2.ConnectPort()

    # Enable 최대 3초 재시도
    print("[*] Waiting for piper to be enabled...")
    while not v2.EnablePiper():
        time.sleep(0.01)
    print("[OK] Piper enabled.")

    # 모션 모드로 진입 (arm pos mode) + 조금 기다리기
    v2.MotionCtrl_2(0x01, 0x01, int(clamp(args.speed, 0, 100)), 0x00)
    time.sleep(0.5)

    q_curr = try_read_current_joints_v2(v2, timeout_s=1.0, poll_hz=50.0)
    if q_curr is None:
        if not args.q_init:
            raise RuntimeError("현재 조인트를 SDK(V2)에서 읽지 못했습니다. --q_init j1..j6 제공 필요")
        q_curr = list(args.q_init)
    q_curr = clamp_joints(q_curr)

    # E-Stop 대비: 현재틱 저장
    global _last_q_ticks
    _last_q_ticks = [rad_to_tick(r) for r in q_curr]
    if ESTOP_REQUESTED:
        soft_estop_v2(v2, _last_q_ticks)
        raise SystemExit("[ESTOP] Before planning.")

    # FK current
    T_current = fk_from_q(chain, q_curr)
    # compose EE-local delta
    T_target = compose_local_delta(T_current, dx, dy, dz, dr, dp, dyaw, axes=args.euler_axes)
    # IK with current as init
    q_target = clamp_joints(ik_to_q(chain, T_target, q_init_from_curr=q_curr))

    # logs
    r_t, p_t, y_t = mat2euler(T_target[:3,:3], axes=args.euler_axes)
    x_t, y_t, z_t = T_target[:3,3]
    print(f"[Plan] T_target xyz=({x_t:.3f},{y_t:.3f},{z_t:.3f}) rpy=({r_t:.3f},{p_t:.3f},{y_t:.3f})")
    print(f"[Plan] q_target(rad): {[round(v,4) for v in q_target]}")
    print(f"[Plan] gripper(mm): {grip_mm:.1f}")

    if args.dryrun or ESTOP_REQUESTED:
        if ESTOP_REQUESTED:
            soft_estop_v2(v2, _last_q_ticks)
        print("[Dryrun/ESTOP] No motion sent.")
        return

    # 3) send to PiPER (V2 ticks path)
    # 모션모드: 그룹=0x01(arm), 모드=0x01(joint pos), 속도스케일=args.speed
    v2.MotionCtrl_2(0x01, 0x01, int(clamp(args.speed, 0, 100)), 0x00)

    # Arm
    q_ticks = [rad_to_tick(r) for r in q_target]
    if ESTOP_REQUESTED:
        soft_estop_v2(v2, _last_q_ticks)
        raise SystemExit("[ESTOP] Before arm command.")
    v2.JointCtrl(q_ticks[0], q_ticks[1], q_ticks[2], q_ticks[3], q_ticks[4], q_ticks[5])

    # Gripper
    if ESTOP_REQUESTED:
        soft_estop_v2(v2, _last_q_ticks)
        raise SystemExit("[ESTOP] Before gripper command.")
    g_ticks = abs(grip_mm_to_tick(grip_mm))   # mm → m → um
    v2.GripperCtrl(g_ticks, 1000, 0x01, 0x00) # set_zero=0x00(일반)

    if ESTOP_REQUESTED:
        soft_estop_v2(v2, _last_q_ticks)
        raise SystemExit("[ESTOP] After commands.")

    time.sleep(0.3)
    print("[OK] Sent to PiPER V2 (one-shot)")

    # 4) optional save json
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(
                {"action": action, "q_target": q_target, "q_ticks": q_ticks, "g_ticks": g_ticks},
                f, indent=2
            )
        print(f"[*] Saved to {args.save_json}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        request_estop()
        sys.exit(1)
