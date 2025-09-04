#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenVLA (8bit) → 7-DoF (EE-local Δpose6 + gripper) → IK(j1..j6) → PiPER (one-shot, no ROS)

- CUDA + bitsandbytes(8bit) 필수
- 입력: 단일 이미지 + 인스트럭션
- 출력: action[7] = [dx,dy,dz, droll,dpitch,dyaw, grip_mm] (EE 로컬 프레임)
- 처리: SDK로 현재 관절 읽기 → FK(T_current) → T_target = T_current · ΔT_local → IK → PiPER로 단발 송신
- URDF: 기본 piper_description.urdf (EE=gripper_base, joint1..6만 IK 활성)

사용 예:
  python3 openvla_to_piper_8bit.py \
    --model_path openvla/openvla-7b \
    --image_path rgb.jpg \
    --instruction "pick up the red mug" \
    --can can0 \
    --urdf piper_description.urdf
"""
import argparse, json, time, sys
from pathlib import Path
from typing import List, Optional
import numpy as np

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from transforms3d.euler import euler2mat, mat2euler
from ikpy.chain import Chain

from piper_control import piper_interface, piper_init
from piper_sdk import C_PiperInterface, LogLevel


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
# PiPER joint limits (rad)
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


# --------------------------
# IK / FK helpers (URDF)
# --------------------------
def build_chain(urdf_path: str, last_link: str = "gripper_base") -> Chain:
    chain = Chain.from_urdf_file(
        urdf_path,
        base_elements=["base_link"],
        last_link=last_link,
    )
    # joint1..joint6만 활성화 (link1..link6)
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
        idx = names.index(link_name)
        q_all[idx] = float(q6[i])
    return chain.forward_kinematics(q_all)

def ik_to_q(chain: Chain, T_target: np.ndarray, q_init_from_curr: Optional[List[float]] = None) -> List[float]:
    q_init_full = [0.0] * len(chain.links)
    if q_init_from_curr is not None:
        names = [lnk.name for lnk in chain.links]
        for i, link_name in enumerate(["link1","link2","link3","link4","link5","link6"]):
            idx = names.index(link_name)
            q_init_full[idx] = float(q_init_from_curr[i])
    q_all = chain.inverse_kinematics_frame(
        T_target, initial_position=q_init_full, max_iter=200, target_orientation_weight=1.0
    )
    names = [lnk.name for lnk in chain.links]
    return [float(q_all[names.index(nm)]) for nm in ["link1","link2","link3","link4","link5","link6"]]

def compose_local_delta(T_current: np.ndarray, dx, dy, dz, dr, dp, dyaw, axes='sxyz') -> np.ndarray:
    """EE-local delta → T_target = T_current · ΔT_local"""
    R_delta = euler2mat(dr, dp, dyaw, axes=axes)
    Delta = np.eye(4); Delta[:3,:3] = R_delta; Delta[:3,3] = np.array([dx,dy,dz], dtype=float)
    return T_current @ Delta


# --------------------------
# PiPER SDK helpers
# --------------------------
def try_read_current_joints(piper: C_PiperInterface) -> Optional[List[float]]:
    """Return 6 joint angles [rad] or None."""
    try:
        msg = piper.GetArmJointMsgs()
        if isinstance(msg, (list, tuple)) and len(msg) >= 6:
            return [float(v) for v in msg[:6]]
        out = []
        for i in range(1, 7):
            val = None
            for path in [f"motor_{i}.position", f"motor_{i}.angle_rad",
                         f"joint_{i}.position", f"joint{i}.position"]:
                obj = msg; ok = True
                for attr in path.split('.'):
                    if not hasattr(obj, attr): ok = False; break
                    obj = getattr(obj, attr)
                if ok: val = float(obj); break
            if val is None: return None
            out.append(val)
        return out
    except Exception:
        return None


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
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="OpenVLA(8bit)->PiPER one-shot")
    ap.add_argument("--model_path", type=str, default="openvla/openvla-7b")
    ap.add_argument("--instruction", type=str, required=True)
    ap.add_argument("--image_path", type=str, default=None)
    ap.add_argument("--image_url",  type=str, default=None)
    ap.add_argument("--unnorm_key", type=str, default="bridge_orig")

    ap.add_argument("--can",   type=str, default="can0", help="CAN interface (e.g., can0)")
    ap.add_argument("--urdf",  type=str, default="piper_description.urdf")
    ap.add_argument("--euler_axes", type=str, default="sxyz")
    ap.add_argument("--q_init", nargs=6, type=float, help="Fallback joints(rad) if SDK read fails")
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--save_json", type=str, default=None)
    ap.add_argument("--print_prompt", action="store_true")
    args = ap.parse_args()

    print(f"[*] device: cuda (8-bit only)")
    print(f"[*] model: {args.model_path}")
    print(f"[*] instruction: {args.instruction}")

    # 1) image + model
    image = load_image(args.image_path, args.image_url)
    processor, vla = load_model_8bit(args.model_path)

    prompt = get_openvla_prompt(args.instruction, args.model_path)
    if args.print_prompt:
        print(f"\n[Prompt]\n{prompt}\n")

    inputs = processor(prompt, image).to("cuda", dtype=torch.float16)

    # 2) inference
    t0 = time.time()
    action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
    dt = time.time() - t0

    # normalize to python list
    if hasattr(action, "tolist"): action = action.tolist()
    action = [float(x) for x in action]
    if len(action) != 7:
        raise RuntimeError(f"Expected 7-DoF action, got len={len(action)}")

    dx, dy, dz, dr, dp, dyaw, grip_mm = action
    print(f"[OK] Inference: {dt:.3f}s")
    print("Action (EE-local): [dx, dy, dz, droll, dpitch, dyaw, grip_mm]")
    print([round(v, 5) for v in action])

    # 3) IK pipeline
    chain = build_chain(args.urdf, last_link="gripper_base")

    # SDK connect + read current joints
    sdk = C_PiperInterface(
        can_name=args.can,
        judge_flag=False,
        can_auto_init=True,
        dh_is_offset=1,
        start_sdk_joint_limit=False,
        start_sdk_gripper_limit=False,
        logger_level=LogLevel.WARNING,
        log_to_file=False,
        log_file_path=None,
    )
    sdk.ConnectPort()
    q_curr = try_read_current_joints(sdk)
    if q_curr is None:
        if not args.q_init:
            raise RuntimeError("현재 조인트를 SDK에서 읽지 못했습니다. --q_init j1..j6 제공 필요")
        q_curr = list(args.q_init)
    q_curr = clamp_joints(q_curr)

    # FK current
    T_current = fk_from_q(chain, q_curr)
    # compose EE-local delta
    T_target = compose_local_delta(T_current, dx, dy, dz, dr, dp, dyaw, axes=args.euler_axes)
    # IK with current as init
    q_target = ik_to_q(chain, T_target, q_init_from_curr=q_curr)
    q_target = clamp_joints(q_target)

    # logs
    r_t, p_t, y_t = mat2euler(T_target[:3,:3], axes=args.euler_axes)
    x_t, y_t, z_t = T_target[:3,3]
    print(f"[Plan] T_target xyz=({x_t:.3f},{y_t:.3f},{z_t:.3f}) rpy=({r_t:.3f},{p_t:.3f},{y_t:.3f})")
    print(f"[Plan] q_target(rad): {[round(v,4) for v in q_target]}")
    print(f"[Plan] gripper(mm): {grip_mm:.1f}")

    if args.dryrun:
        print("[Dryrun] No motion sent.")
    else:
        # 4) send to PiPER (arm via piper_control, gripper via sdk)
        robot = piper_interface.PiperInterface(can_port=args.can)
        piper_init.reset_arm(
            robot,
            arm_controller=piper_interface.ArmController.POSITION_VELOCITY,
            move_mode=piper_interface.MoveMode.JOINT,
        )
        piper_init.reset_gripper(robot)
        robot.command_joint_positions(q_target)

        grip_mm = float(clamp(grip_mm, 0.0, 50.0))  # adjust to your hardware spec
        sdk.GripperCtrl(int(round(grip_mm)), 1000, 0x01, 1)

        time.sleep(0.3)
        print("[OK] Sent to PiPER (one-shot)")

    # 5) optional save json
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"action": action, "q_target": q_target}, f, indent=2)
        print(f"[*] Saved to {args.save_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)