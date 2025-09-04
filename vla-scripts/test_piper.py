#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenVLA(8bit) → 7-DoF (EE-local Δpose6 + gripper_mm) → IK(j1..6 rad) → V2 ticks → PiPER(CAN, no ROS)
- Instruction 터미널 입력 지원 (미지정 시)
- Emergency Stop: Ctrl+C 또는 'e' 키 (best-effort: 현재자세 유지 재명령 + 속도 0)

조인트 스케일: rad → milli-deg(틱) = rad * 180/pi * 1000
그리퍼 스케일: meter → um(틱)     = m * 1e6  (OpenVLA는 grip_mm이므로 mm/1000 → m)

예시:
  python3 openvla_to_piper_8bit_v2_estop.py \
    --model_path openvla/openvla-7b \
    --image_path rgb.jpg \
    --can can0 \
    --urdf piper_description.urdf
"""

import argparse, json, time, sys, signal, threading, tty, termios, os, math
from pathlib import Path
from typing import List, Optional
import numpy as np

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from transforms3d.euler import euler2mat, mat2euler
from ikpy.chain import Chain

# PiPER V2 SDK (네가 성공했던 경로)
from piper_sdk import C_PiperInterface_V2, LogLevel

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

# --------------------------
# IK / FK (URDF)
# --------------------------
def build_chain(urdf_path: str, last_link: str = "gripper_base") -> Chain:
    chain = Chain.from_urdf_file(
        urdf_path,
        base_elements=["base_link"],
        last_link=last_link,
    )
    mask = [False] * len(chain.links)
    names = [lnk.name for lnk in chain.links]
    for link_name in ["link1","link2","link3","link4","link5","link6"]:
        if link_name in names:
            mask[names.index(link_name)] = True
    chain.active_links_mask = mask
    return chain

def fk_from_q(chain: Chain, q6: List[float]) -> np.ndarray:
    q_all = [0.0] * len(chain.links)
    names = [lnk.name for lnk in chain.links]
    for i, link_name in enumerate(["link1","link2","link3","link4","link5","link6"]):
        idx = names.index(link_name); q_all[idx] = float(q6[i])
    return chain.forward_kinematics(q_all)

def ik_to_q(chain: Chain, T_target: np.ndarray, q_init_from_curr: Optional[List[float]] = None) -> List[float]:
    q_init_full = [0.0] * len(chain.links)
    if q_init_from_curr is not None:
        names = [lnk.name for lnk in chain.links]
        for i, link_name in enumerate(["link1","link2","link3","link4","link5","link6"]):
            idx = names.index(link_name); q_init_full[idx] = float(q_init_from_curr[i])
    q_all = chain.inverse_kinematics_frame(
        T_target, initial_position=q_init_full, max_iter=200, target_orientation_weight=1.0
    )
    names = [lnk.name for lnk in chain.links]
    return [float(q_all[names.index(nm)]) for nm in ["link1","link2","link3","link4","link5","link6"]]

def compose_local_delta(T_current: np.ndarray, dx, dy, dz, dr, dp, dyaw, axes='sxyz') -> np.ndarray:
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
def try_read_current_joints_v2(p: C_PiperInterface_V2) -> Optional[List[float]]:
    """
    V2에서 현재 조인트(rad) 추출 시도.
    장비/SDK 빌드마다 리턴타입이 다를 수 있어 보수적으로 처리.
    네 데모에선 GetArmJointCtrl()가 잘 동작했다고 했으므로 우선 사용.
    """
    try:
        msg = p.GetArmJointCtrl()
        # 흔한 케이스 1) 리스트/튜플로 rad 값 6개
        if isinstance(msg, (list, tuple)) and len(msg) >= 6:
            return [float(msg[i]) for i in range(6)]
        # 케이스 2) 객체의 속성 경로로 탐색
        out = []
        for i in range(1, 7):
            val = None
            for path in [f"joint_{i}.position", f"motor_{i}.angle_rad",
                         f"j{i}.position", f"motor_{i}.position"]:
                obj = msg; ok = True
                for attr in path.split('.'):
                    if not hasattr(obj, attr): ok = False; break
                    obj = getattr(obj, attr)
                if ok:
                    val = float(obj); break
            if val is None: return None
            out.append(val)
        return out
    except Exception:
        return None

def soft_estop_v2(p: Optional[C_PiperInterface_V2], q_ticks: Optional[List[int]]):
    """
    Best-effort: 현재 자세 유지 재명령 + 속도스케일 0.
    (펌웨어마다 해석 다를 수 있음. 하드 E-Stop 병행 권장)
    """
    try:
        if p:
            # 속도 스케일 0 시도 (group=0x01 arm, mode=0x01 joint pos)
            p.MotionCtrl_2(0x01, 0x01, 0, 0x00)
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
    args = ap.parse_args()

    # 0) Prompt for instruction if missing
    if not args.instruction:
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
    print(f"[*] instruction: {args.instruction}")

    # 1) image + model
    image = load_image(args.image_path, args.image_url)
    processor, vla = load_model_8bit(args.model_path)

    prompt = get_openvla_prompt(args.instruction, args.model_path)
    if args.print_prompt:
        print(f"\n[Prompt]\n{prompt}\n")

    # 키 리스너
    start_estop_key_listener(args.estop_key)

    inputs = processor(prompt, image).to("cuda", dtype=torch.float16)
    if ESTOP_REQUESTED: raise SystemExit("[ESTOP] Before inference.")

    # 2) inference
    t0 = time.time()
    action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
    dt = time.time() - t0

    if hasattr(action, "tolist"):
        action = action.tolist()
    action = [float(x) for x in action]
    if len(action) != 7:
        raise RuntimeError(f"Expected 7-DoF action, got len={len(action)}")

    dx, dy, dz, dr, dp, dyaw, grip_mm = action
    print(f"[OK] Inference: {dt:.3f}s")
    print("Action (EE-local): [dx, dy, dz, droll, dpitch, dyaw, grip_mm]")
    print([round(v, 5) for v in action])

    # 3) IK pipeline
    chain = build_chain(args.urdf, last_link="gripper_base")

    # V2 connect + enable + current joints
    v2 = C_PiperInterface_V2(args.can)
    v2.ConnectPort()
    # enable 시도
    t_en = time.time()
    while time.time() - t_en < 3.0:
        try:
            if v2.EnablePiper():
                break
        except Exception:
            pass
        time.sleep(0.05)

    time.sleep(0.02)
    q_curr = try_read_current_joints_v2(v2)
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

    # 4) send to PiPER (V2 ticks path)
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
    g_ticks = grip_mm_to_tick(grip_mm)         # mm → m → um
    g_ticks = abs(g_ticks)
    v2.GripperCtrl(g_ticks, 1000, 0x01, 0x00)  # set_zero=0x00(일반)

    if ESTOP_REQUESTED:
        soft_estop_v2(v2, _last_q_ticks)
        raise SystemExit("[ESTOP] After commands.")

    time.sleep(0.3)
    print("[OK] Sent to PiPER V2 (one-shot)")

    # 5) optional save json
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"action": action, "q_target": q_target, "q_ticks": q_ticks, "g_ticks": g_ticks}, f, indent=2)
        print(f"[*] Saved to {args.save_json}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        request_estop()
        sys.exit(1)
