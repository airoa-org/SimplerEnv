import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from simpler_env.evaluation.config import ManiSkill2Config
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.policies.base import AiroaBasePolicy


def calculate_robust_score(results: List[List[bool]], penalty_factor: float = 0.5) -> float:
    """
    成功/失敗のリストのリストから、安定性を考慮したロバストスコアを計算します。

    Args:
        results (List[List[bool]]): 各評価実行の成功結果リスト (例: [[True, False], [True, True]])
        penalty_factor (float): 標準偏差に対するペナルティ係数。

    Returns:
        float: 0から1の間のロバストスコア。
    """
    if not results:
        return 0.0

    # 各実行の成功率を計算
    success_rates = [np.mean(run) for run in results if run]

    if not success_rates:
        return 0.0

    # 平均成功率と標準偏差を計算
    mean_success_rate = np.mean(success_rates)
    std_dev = np.std(success_rates)

    # 平均から標準偏差にペナルティ係数を掛けた値を引く
    robust_score = mean_success_rate - penalty_factor * std_dev

    # スコアが負にならないように0でクリップ
    return max(0.0, robust_score)


def _run_single_evaluation(env_policy: AiroaBasePolicy, cfg: ManiSkill2Config, ckpt_path: str) -> List[bool]:
    """
    単一の評価設定でmaniskill2_evaluatorを実行し、結果を返す内部ヘルパー。
    """
    build = getattr(cfg, "additional_env_build_kwargs", None) or {}
    print(f"  ▶️  Running: env={cfg.env_name}, scene={cfg.scene_name}, kwargs={build.get('urdf_version') or build}")
    try:
        success_arr = maniskill2_evaluator(env_policy, cfg)
        rate = float(np.mean(success_arr)) if len(success_arr) else 0.0
        print(f"  ✅  Success Rate: {rate:.2%}")
        return success_arr
    except Exception as e:
        print(f"  ❌  An error occurred: {e}")
        return []


# ==============================================================================
# 各タスクの評価をまとめた内部関数
# ==============================================================================


def _evaluate_coke_can_grasping(env_policy: AiroaBasePolicy, ckpt_path: str) -> Tuple[List[List[bool]], List[List[bool]]]:
    """「Coke Can Grasping」タスクのすべてのバリエーションを評価する"""
    print("\n--- Starting: 1. Coke Can Grasping Suite ---")
    sim_results, vm_results = [], []

    coke_can_options_arr = [{"lr_switch": True}, {"upright": True}, {"laid_vertically": True}]
    urdf_version_arr = [None, "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]

    base_kwargs = {
        "robot": "google_robot_static",
        "policy_setup": "google_robot",
        "control_freq": 3,
        "sim_freq": 501,
        "max_episode_steps": 80,
        "ckpt_path": ckpt_path,
        "robot_init_x_range": [0.35, 0.35, 1],
        "robot_init_y_range": [0.20, 0.20, 1],
        "obj_init_x_range": [-0.35, -0.12, 5],
        "obj_init_y_range": [-0.02, 0.42, 5],
        "robot_init_rot_quat_center": [0.0, 0.0, 0.0, 1.0],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
    }

    for urdf_version in urdf_version_arr:
        for coke_can_option in coke_can_options_arr:
            additional_kwargs = {**coke_can_option, "urdf_version": urdf_version}

            # URDFバージョンがNoneの場合はsim、それ以外はvisual matchingとして分類
            is_visual_matching = urdf_version is not None

            cfg = ManiSkill2Config(
                **base_kwargs,
                env_name="GraspSingleOpenedCokeCanInScene-v0",
                scene_name="google_pick_coke_can_1_v4",
                rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png",
                additional_env_build_kwargs=additional_kwargs,
            )
            result = _run_single_evaluation(env_policy, cfg, ckpt_path)

            if is_visual_matching:
                vm_results.append(result)
            else:
                sim_results.append(result)

    return sim_results, vm_results


def _evaluate_drawer_placement(env_policy: AiroaBasePolicy, ckpt_path: str) -> Tuple[List[List[bool]], List[List[bool]]]:
    """「Drawer Placement」タスクのsimとoverlay評価を行う（シェルスクリプトの動作に準拠）"""
    print("\n--- Starting: 2. Drawer Placement Suite ---")
    sim_results, vm_results = [], []

    # --- 2a. Simulation ---
    print("\n[Section 2a: Simulation Variants]")

    # レイトレーシング設定を含まない、全シミュレーション共通の引数
    common_sim_kwargs = {
        "robot": "google_robot_static",
        "policy_setup": "google_robot",
        "control_freq": 3,
        "sim_freq": 513,
        "max_episode_steps": 200,
        "ckpt_path": ckpt_path,
        "additional_env_build_kwargs": {"model_ids": "apple"},
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "obj_init_x_range": [-0.08, -0.02, 3],
        "obj_init_y_range": [-0.02, 0.08, 3],
        "robot_init_x_range": [0.65, 0.65, 1],
        "robot_init_y_range": [-0.2, 0.2, 3],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
    }

    # === ケース1: Base (enable_raytracing=Trueを使用) ===
    base_case_kwargs = common_sim_kwargs.copy()
    base_case_kwargs["scene_name"] = "frl_apartment_stage_simple"
    # TrueにするとFailed to denoise OptiX Error: OPTIX_ERROR_INVALID_VALUEが出る
    base_case_kwargs["enable_raytracing"] = False
    cfg_base = ManiSkill2Config(**base_case_kwargs, env_name="PlaceIntoClosedTopDrawerCustomInScene-v0")
    sim_results.append(_run_single_evaluation(env_policy, cfg_base, ckpt_path))

    # === ケース2: Backgrounds, Lights, Cabinets (shader_dir='rt'を使用) ===
    other_eval_configs = [
        # Backgrounds
        {"scene_name": "modern_bedroom_no_roof"},
        {"scene_name": "modern_office_no_roof"},
        # Lights
        {"scene_name": "frl_apartment_stage_simple", "additional_env_build_kwargs": {"light_mode": "brighter"}},
        {"scene_name": "frl_apartment_stage_simple", "additional_env_build_kwargs": {"light_mode": "darker"}},
        # Cabinets
        {"scene_name": "frl_apartment_stage_simple", "additional_env_build_kwargs": {"station_name": "mk_station2"}},
        {"scene_name": "frl_apartment_stage_simple", "additional_env_build_kwargs": {"station_name": "mk_station3"}},
    ]

    for config_update in other_eval_configs:
        kwargs = common_sim_kwargs.copy()
        update_data = config_update.copy()

        # kwargs内の辞書を安全にコピーしてマージ処理を行う
        merged_build_kwargs = kwargs.get("additional_env_build_kwargs", {}).copy()

        # シェルの shader_dir=rt に相当する設定を追加
        merged_build_kwargs["shader_dir"] = "rt"

        # 各評価（Light, Cabinetなど）固有のビルド引数をマージ
        if "additional_env_build_kwargs" in update_data:
            merged_build_kwargs.update(update_data.pop("additional_env_build_kwargs"))

        kwargs["additional_env_build_kwargs"] = merged_build_kwargs

        # scene_name などのトップレベルの引数を更新
        kwargs.update(update_data)

        cfg = ManiSkill2Config(**kwargs, env_name="PlaceIntoClosedTopDrawerCustomInScene-v0")
        sim_results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # --- 2b. Visual Matching (Overlay) ---
    print("\n[Section 2b: Visual Matching Variants]")
    overlay_setups = [
        {
            "robot_init_x_range": [0.644, 0.644, 1],
            "robot_init_y_range": [-0.179, -0.179, 1],
            "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.03, -0.03, 1],
            "rgb_overlay_path": "./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png",
        },
        {
            "robot_init_x_range": [0.652, 0.652, 1],
            "robot_init_y_range": [0.009, 0.009, 1],
            "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "rgb_overlay_path": "./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png",
        },
        {
            "robot_init_x_range": [0.665, 0.665, 1],
            "robot_init_y_range": [0.224, 0.224, 1],
            "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "rgb_overlay_path": "./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png",
        },
    ]
    urdf_versions = ["recolor_cabinet_visual_matching_1", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", None]

    # Visual Matchingの基本設定は 'enable_raytracing' を含まないため、元のcommon_sim_kwargsが利用できる
    vm_base_kwargs = common_sim_kwargs.copy()
    vm_base_kwargs["enable_raytracing"] = False  # Visual Matchingではレイトレーシングを有効にする

    for urdf in urdf_versions:
        additional_kwargs = {
            "station_name": "mk_station_recolor",
            "light_mode": "simple",
            "disable_bad_material": True,
            "urdf_version": urdf,
            "model_ids": "baked_apple_v2",
        }
        for setup in overlay_setups:
            kwargs = vm_base_kwargs.copy()

            kwargs["env_name"] = "PlaceIntoClosedTopDrawerCustomInScene-v0"
            kwargs["scene_name"] = "dummy_drawer"

            merged_build = kwargs.get("additional_env_build_kwargs", {}).copy()
            merged_build.update(additional_kwargs)
            kwargs["additional_env_build_kwargs"] = merged_build

            kwargs.update(setup)

            cfg = ManiSkill2Config(**kwargs)
            vm_results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return sim_results, vm_results


def _evaluate_move_near(env_policy: AiroaBasePolicy, ckpt_path: str) -> Tuple[List[List[bool]], List[List[bool]]]:
    """「Move Near」タスクのsimとoverlay評価を行う"""
    print("\n--- Starting: 3. Move Near Suite ---")
    sim_results, vm_results = [], []

    base_kwargs = {
        "robot": "google_robot_static",
        "policy_setup": "google_robot",
        "control_freq": 3,
        "sim_freq": 513,
        "max_episode_steps": 80,
        "robot_init_x_range": [0.35, 0.35, 1],
        "robot_init_y_range": [0.21, 0.21, 1],
        "obj_variation_mode": "episode",
        "obj_episode_range": [0, 60],
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "ckpt_path": ckpt_path,
    }

    # --- 3a. Simulation ---
    print("\n[Section 3a: Simulation Variants]")
    sim_eval_configs = [
        {   
            "env_name": "MoveNearGoogleInScene-v0", 
            "scene_name": "google_pick_coke_can_1_v4",
            "additional_env_build_kwargs": {}
        },
        {   
            "env_name": "MoveNearGoogleInScene-v0", 
            "scene_name": "google_pick_coke_can_1_v4", 
            "additional_env_build_kwargs": {"no_distractor": True}
        },
        {
            "env_name": "MoveNearGoogleInScene-v0", 
            "scene_name": "google_pick_coke_can_1_v4_alt_background",
            "additional_env_build_kwargs": {}
        },
        {   
            "env_name": "MoveNearGoogleInScene-v0", 
            "scene_name": "google_pick_coke_can_1_v4_alt_background_2",
            "additional_env_build_kwargs": {}
        },
        {
            "env_name": "MoveNearGoogleInScene-v0",
            "scene_name": "google_pick_coke_can_1_v4",
            "additional_env_build_kwargs": {"slightly_darker_lighting": True},
        },
        {
            "env_name": "MoveNearGoogleInScene-v0",
            "scene_name": "google_pick_coke_can_1_v4",
            "additional_env_build_kwargs": {"slightly_brighter_lighting": True},
        },
        {   
            "env_name": "MoveNearGoogleInScene-v0", 
            "scene_name": "Baked_sc1_staging_objaverse_cabinet1_h870",
            "additional_env_build_kwargs": {}
        },
        {
            "env_name": "MoveNearGoogleInScene-v0", 
            "scene_name": "Baked_sc1_staging_objaverse_cabinet2_h870",
            "additional_env_build_kwargs": {}
        },
        {
            "env_name": "MoveNearAltGoogleCameraInScene-v0", 
            "scene_name": "google_pick_coke_can_1_v4",
            "additional_env_build_kwargs": {}
        },
        {
            "env_name": "MoveNearAltGoogleCamera2InScene-v0", 
            "scene_name": "google_pick_coke_can_1_v4",
            "additional_env_build_kwargs": {}
        },
    ]
    for config_update in sim_eval_configs:
        cfg = ManiSkill2Config(**base_kwargs, **config_update)
        if cfg.additional_env_build_kwargs is None:
            cfg.additional_env_build_kwargs = {}
        sim_results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # --- 3b. Visual Matching (Overlay) ---
    print("\n[Section 3b: Visual Matching Variants]")
    urdf_versions = [None, "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]
    for urdf in urdf_versions:
        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="MoveNearGoogleBakedTexInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png",
            additional_env_save_tags="baked_except_bpb_orange",
            additional_env_build_kwargs={"urdf_version": urdf},
        )
        vm_results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return sim_results, vm_results


# ==============================================================================
# メインの総合評価関数
# ==============================================================================
SIM_WEIGHT = 0.3
VISUAL_MATCHING_WEIGHT = 0.7


def run_comprehensive_evaluation(env_policy: AiroaBasePolicy, ckpt_path: str) -> Dict[str, float]:
    """
    全ての評価スクリプトを統合して実行し、ロバストスコアを計算・表示する。

    Args:
        env_policy (AiroaBasePolicy): 評価対象のポリシー。
        ckpt_path (str): 評価に使用するチェックポイントのパス。
        SIM_WEIGHT (float): シミュレーション評価のスコアに対する重み。
        VISUAL_MATCHING_WEIGHT (float): Visual Matching評価のスコアに対する重み。

    Returns:
        Dict[str, float]: 計算されたスコアを含む辞書。
    """
    print("=" * 80)
    print(f"🚀 STARTING COMPREHENSIVE EVALUATION 🚀")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Weights: Sim={SIM_WEIGHT}, VisualMatching={VISUAL_MATCHING_WEIGHT}")
    print("=" * 80)

    # 各評価スイートを実行
    coke_sim, coke_vm = _evaluate_coke_can_grasping(env_policy, ckpt_path)
    drawer_sim, drawer_vm = _evaluate_drawer_placement(env_policy, ckpt_path)
    movenear_sim, movenear_vm = _evaluate_move_near(env_policy, ckpt_path)

    # 結果をカテゴリ別に集約
    all_sim_results = coke_sim + drawer_sim + movenear_sim
    all_vm_results = coke_vm + drawer_vm + movenear_vm

    # カテゴリ別にロバストスコアを計算
    sim_score = calculate_robust_score(all_sim_results)
    vm_score = calculate_robust_score(all_vm_results)

    # 最終スコアを重み付けして計算
    # 重みの合計が0になるエッジケースを避ける
    total_weight = SIM_WEIGHT + VISUAL_MATCHING_WEIGHT
    if total_weight == 0:
        final_score = 0.0
    else:
        final_score = (sim_score * SIM_WEIGHT + vm_score * VISUAL_MATCHING_WEIGHT) / total_weight

    # --- 結果のサマリーを表示 ---
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY 📊")
    print("-" * 80)
    print(f"Simulation Score (Robust):            {sim_score:.4f}")
    print(f"  - Total Simulation Runs: {len(all_sim_results)}")
    print(f"Visual Matching Score (Robust):       {vm_score:.4f}")
    print(f"  - Total Visual Matching Runs: {len(all_vm_results)}")
    print("-" * 80)
    print(f"🏆 Final Weighted Score:               {final_score:.4f}")
    print("=" * 80)

    return {
        "final_score": final_score,
        "simulation_robust_score": sim_score,
        "visual_matching_robust_score": vm_score,
    }
