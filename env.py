from typing import List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from channel import (
    build_path_directions,
    build_rx_port_positions,
    compute_capacity,
    doppler_phase,
    generate_uav_trajectory,
    sample_paths,
    build_tx_array_offsets,
)
from config import Config


class FASEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # 预计算固定部分（无人机轨迹和接收端口位置）
        self.times, self.uav_positions = generate_uav_trajectory(cfg)
        self.rx_port_positions = build_rx_port_positions(cfg)

        # 移除预计算信道矩阵 - 改为实时计算
        self.H_current = None  # 当前时间步信道矩阵
        self.H_prev = None  # 上一步信道矩阵

        # 初始化历史状态变量
        self.last_capacity = None  # 上一步容量
        self.velocity_history = []  # 速度历史记录

        # 动作空间参数 - 添加缺失的最大跳变半径
        self.replace_shifts = [s for s in range(-(self.cfg.N - 1), self.cfg.N) if s != 0]
        self.min_port_spacing_idx = 0

        # 构建动作空间
        self.action_catalog = self._build_action_catalog()
        self.action_space = spaces.Discrete(len(self.action_catalog))

        # 观察空间维度计算␊
        # 3维位置 + 3维速度 + 1维时间 + 1维信道变化率 + 1维容量梯度 + 1维相位变化率 + N维端口状态 + 1维容量
        base_obs_dim = 3 + 3 + 1 + 1 + 1 + 1 + cfg.N + 1
        mask_dim = self.action_space.n
        total_obs_dim = base_obs_dim + mask_dim

        # 初始化环境状态
        self.t_idx = 0
        self.active_set: Set[int] = set()
        self.current_capacity = 0.0
        self.ula_baseline_capacity = 0.0

        # 初始化多径簇（在reset中实际生成）
        self.clusters = None
        dummy_obs = self._get_state()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32)

    def _build_action_catalog(self) -> List[Tuple[str, int, int]]:
        actions: List[Tuple[str, int, int]] = [("none", -1, 0)]
        # 仅保留无操作和位置替换动作
        for n in range(self.cfg.N):
            for shift in self.replace_shifts:
                actions.append(("replace", n, shift))
        return actions

    def _validate_manual_ports(self, ports: Tuple[int, ...]) -> List[int]:
        uniq = sorted(set(int(p) for p in ports))
        if len(uniq) == 0:
            raise ValueError("init_active_ports 不能为空（manual 模式下）")
        if len(uniq) != self.cfg.K:
            raise ValueError(f"manual 端口数量必须等于 K={self.cfg.K}，当前={len(uniq)}")
        if uniq[0] < 0 or uniq[-1] >= self.cfg.N:
            raise ValueError(f"manual 端口索引必须在 [0, {self.cfg.N - 1}]")
        return uniq

    def _is_spacing_legal(self, ports: List[int]) -> bool:
        return True

    def _is_port_spacing_legal_with_set(self, n: int, active_set: Set[int]) -> bool:
        return True

    def _sample_with_spacing_constraint(self, candidate_pool: np.ndarray, target_k: int) -> Set[int]:
        candidate_pool = np.array(candidate_pool, dtype=int)
        if target_k <= 0:
            return set()

        if target_k > len(candidate_pool):
            raise ValueError(f"候选端口数量不足，无法采样 {target_k} 个不同端口。")
        selected = self.rng.choice(candidate_pool, size=target_k, replace=False)
        return set(int(x) for x in selected)

    def _sample_initial_active_set(self) -> Set[int]:
        strategy = str(self.cfg.init_port_strategy).lower().strip()
        init_k = int(self.cfg.K)

        if strategy == "manual":
            if self.cfg.init_active_ports is None:
                raise ValueError("init_port_strategy='manual' 时必须提供 init_active_ports")
            return set(self._validate_manual_ports(self.cfg.init_active_ports))

        if strategy == "spread":
            selected = np.linspace(0, self.cfg.N - 1, num=init_k, dtype=int)
            if not self._is_spacing_legal(selected.tolist()):
                return self._sample_with_spacing_constraint(np.arange(self.cfg.N), init_k)
            return set(int(x) for x in selected)

        if strategy == "clustered":
            center = int(self.rng.integers(0, self.cfg.N))
            radius = max(1, int(self.cfg.init_cluster_radius))
            window = np.arange(max(0, center - radius), min(self.cfg.N, center + radius + 1))
            if len(window) < init_k:
                window = np.arange(self.cfg.N)
            return self._sample_with_spacing_constraint(window, init_k)

        # 默认 random：复用原逻辑
        center = int(self.rng.integers(0, self.cfg.N))
        radius = max(1, int(self.cfg.init_cluster_radius))
        window = np.arange(max(0, center - radius), min(self.cfg.N, center + radius + 1))
        if len(window) < init_k:
            window = np.arange(self.cfg.N)
        return self._sample_with_spacing_constraint(window, init_k)


    def _is_legal_action(self, action_idx: int, active_set: Optional[Set[int]] = None) -> bool:
        if active_set is None:
            active_set = self.active_set

        op, n, shift = self.action_catalog[action_idx]
        if op == "none":
            return True

        if op == "replace":
            if n not in active_set:
                return False
            new_n = n + shift
            if new_n < 0 or new_n >= self.cfg.N:
                return False
            if new_n in active_set:
                return False

            remaining = set(active_set)
            remaining.remove(n)
            if not self._is_port_spacing_legal_with_set(new_n, remaining):
                return False
            return True

        return False

    def _apply_action(self, action_idx: int, active_set: Optional[Set[int]] = None) -> Set[int]:
        if active_set is None:
            active_set = self.active_set
        new_set = set(active_set)

        op, n, shift = self.action_catalog[action_idx]
        if not self._is_legal_action(action_idx, new_set):
            return new_set

        if op == "none":
            return new_set
        elif op == "replace":
            new_set.remove(n)
            new_set.add(n + shift)

        return new_set

    def _action_neighbors(self, action_idx: int) -> List[int]:
        op, n, shift = self.action_catalog[action_idx]
        neighbors = {action_idx}

        # index-neighborhood
        for d in range(1, self.cfg.action_neighbor_radius + 1):
            if action_idx - d >= 0:
                neighbors.add(action_idx - d)
            if action_idx + d < self.action_space.n:
                neighbors.add(action_idx + d)

        # semantic-neighborhood: nearby port indices
        if op == "none":
            return sorted(neighbors)

        for dn in [-1, 1]:
            nn = n + dn
            if 0 <= nn < self.cfg.N:
                # replace block starts at 1
                shift_id = self.replace_shifts.index(shift)
                block = 1 + nn * len(self.replace_shifts) + shift_id
                neighbors.add(block)

        return sorted(neighbors)

    # env.py

    def _memetic_refine_action(self, action_idx: int) -> int:
        if not self.cfg.use_memetic_search:
            return action_idx

        # 1. 获取基础候选集（原有的邻域搜索）
        candidates = set(self._action_neighbors(action_idx))
        candidates.add(action_idx)
        candidates.add(0)  # 始终将“不操作”作为评估选项

        # --- [关键修改 4: 全局贪心交换逻辑] ---
        # 计算当前信道各端口的功率谱
        h_powers = np.linalg.norm(self.H_current, axis=1)  # (N,)

        # 找出当前激活集中，信道增益最差的端口
        active_list = list(self.active_set)
        weakest_port = active_list[np.argmin(h_powers[active_list])]

        # 找出所有空闲端口，并选出其中功率最强的 Top-3
        idle_ports = list(set(range(self.cfg.N)) - self.active_set)
        if idle_ports:
            # 按功率排序取后三个
            strong_indices = np.argsort(h_powers[idle_ports])[-3:]
            for idx in strong_indices:
                target_port = idle_ports[idx]
                # 计算将 weakest_port 移动到 target_port 所需的动作索引
                shift = target_port - weakest_port
                if shift != 0:
                    # 构造并检查这个“贪心动作”是否在 catalog 中
                    # action_idx = 1 + n * num_shifts + shift_pos
                    try:
                        shift_id = self.replace_shifts.index(shift)
                        # 这里的 n 是端口在 N 中的索引
                        greedy_act_idx = 1 + weakest_port * len(self.replace_shifts) + shift_id
                        if self._is_legal_action(greedy_act_idx):
                            candidates.add(greedy_act_idx)
                    except ValueError:
                        pass  # 步长超出 replace_shifts 范围

        # 评估所有候选动作，选出容量最大的
        best_action = action_idx
        best_value = -np.inf

        for a in candidates:
            if not self._is_legal_action(a): continue
            val = self._capacity_if_apply_action(a)
            if val > best_value:
                best_value = val
                best_action = a

        return best_action

    def _capacity_if_apply_action(self, action_idx: int) -> float:
        """在当前时间步信道下，估计执行某动作后的容量（不修改环境状态）"""
        if self.H_current is None:
            return self.current_capacity
        next_set = self._apply_action(action_idx, self.active_set)
        if len(next_set) == 0:
            return 0.0
        idx = np.array(sorted(next_set), dtype=int)
        H_active = self.H_current[idx, :]
        return compute_capacity(H_active, self.cfg.rho_linear, self.cfg.M)

    def _get_action_list(self) -> List[int]:
        return [a for a in range(self.action_space.n) if self._is_legal_action(a)]

    def _get_state(self) -> np.ndarray:
        # --- 1. 定义归一化常数 ---
        # 这些常数需要根据你的物理场景的大致范围进行估计
        pos_norm = 100.0  # 位置归一化分母 (假设场景在100米内)
        vel_norm = 20.0  # 速度归一化分母 (假设最大速度20m/s)
        cap_norm = 50.0  # 容量归一化分母 (假设最大容量50 bits/s/Hz)

        # --- 2. 提取基础特征 ---

        # 端口掩码
        mask = np.zeros(self.cfg.N, dtype=np.float32)
        if len(self.active_set) > 0:
            mask[list(self.active_set)] = 1.0

        # 归一化时间 [0, 1]
        normalized_time = self.t_idx / self.cfg.T

        # 归一化速度
        if self.t_idx < self.cfg.T - 1:
            velocity = (self.uav_positions[self.t_idx + 1] - self.uav_positions[self.t_idx]) / self.cfg.dt
        else:
            velocity = np.zeros(3)
        velocity = velocity / vel_norm

        # 归一化位置
        uav_pos = self.uav_positions[self.t_idx] / pos_norm

        # 归一化信道变化率 (Frobenius范数 / sqrt(N*M))
        h_diff = 0.0
        if self.H_prev is not None and self.H_current is not None:
            h_diff = np.linalg.norm(self.H_current - self.H_prev, 'fro') / np.sqrt(self.cfg.N * self.cfg.M)

        # 容量梯度 (保持原样或轻微缩放)
        capacity_gradient = 0.0
        if self.H_current is not None and len(self.active_set) > 0:
            local_grads = []
            for n in self.active_set:
                left_n = max(0, n - 1)
                right_n = min(self.cfg.N - 1, n + 1)
                left_cap = compute_capacity(self.H_current[left_n:left_n + 1, :], self.cfg.rho_linear, self.cfg.M)
                right_cap = compute_capacity(self.H_current[right_n:right_n + 1, :], self.cfg.rho_linear, self.cfg.M)
                local_grads.append(right_cap - left_cap)
            capacity_gradient = float(np.mean(local_grads)) if local_grads else 0.0

        # 相位变化率 (保持原样)
        phase_variation = 0.0
        if self.H_current is not None:
            spatial_phase = np.unwrap(np.angle(np.mean(self.H_current, axis=1)))
            if spatial_phase.shape[0] > 1:
                phase_variation = float(np.mean(np.abs(np.diff(spatial_phase))))

        # 归一化容量特征
        norm_capacity = self.current_capacity / cap_norm

        # 拼接基础观测向量
        base_obs = np.concatenate(
            [
                uav_pos.astype(np.float32),  # 3
                velocity.astype(np.float32),  # 3
                np.array([normalized_time], dtype=np.float32),  # 1
                np.array([h_diff], dtype=np.float32),  # 1
                np.array([capacity_gradient], dtype=np.float32),  # 1
                np.array([phase_variation], dtype=np.float32),  # 1
                mask,  # N
                np.array([norm_capacity], dtype=np.float32),  # 1
            ]
        ).astype(np.float32)

        # --- 3. 生成动作掩码 ---
        # 这一步非常关键：我们将合法动作的信息作为输入的一部分给神经网络
        action_mask = np.zeros(self.action_space.n, dtype=np.float32)
        legal_actions = self._get_action_list()
        action_mask[legal_actions] = 1.0

        # --- 4. 合并最终观测 ---
        # 最终维度 = 原始维度 + 动作空间维度
        return np.concatenate([base_obs, action_mask]).astype(np.float32)

    def _compute_realtime_channel_at(self, t_idx: int, rx_port_positions: np.ndarray) -> np.ndarray:
        """基于任意接收端端口位置，计算指定时间步的信道矩阵。"""
        time_s = t_idx * self.cfg.dt
        tx_offsets = build_tx_array_offsets(self.cfg)  # (M,3)
        rx_ref = np.mean(rx_port_positions, axis=0)  # RX中心
        rx_offsets = rx_port_positions - rx_ref[None, :]

        k = 2.0 * np.pi / self.cfg.wavelength
        H_t = np.zeros((rx_port_positions.shape[0], self.cfg.M), dtype=np.complex128)

        for cl in self.clusters:
            alphas = cl["alphas"]
            betas = cl["betas"]
            gains = cl["gains"]
            distance_phase = cl["distance_phase"]  # (P,) 时间不变路径相位（远场近似）
            directions = build_path_directions(alphas, betas)  # (P, 3)
            rx_phase = np.exp(-1j * k * (rx_offsets @ directions.T))  # (N, P)
            tx_phase = np.exp(-1j * k * (tx_offsets @ directions.T))  # (M, P)
            doppler = doppler_phase(self.cfg, directions, time_s, k)  # (P,)
            weighted_tx = (gains * doppler * distance_phase)[:, None] * tx_phase.T
            H_t += rx_phase @ weighted_tx

        return H_t

    def _compute_realtime_channel(self) -> np.ndarray:
        """实时计算当前时间步的信道矩阵"""
        return self._compute_realtime_channel_at(self.t_idx, self.rx_port_positions)

    def _compute_ula_baseline_capacity(self) -> float:
        """预计算当前 episode 下固定 ULA 的平均容量，作为奖励对比基准。"""
        ula_port_count = self.cfg.ula_fixed_port_count if self.cfg.ula_fixed_port_count is not None else self.cfg.K
        ula_port_count = max(1, int(ula_port_count))

        xs = np.linspace(0, self.cfg.W, ula_port_count)
        ula_rx_positions = np.zeros((ula_port_count, 3), dtype=float)
        ula_rx_positions[:, 0] = xs

        caps = []
        for t in range(self.cfg.T):
            H_t = self._compute_realtime_channel_at(t, ula_rx_positions)
            caps.append(compute_capacity(H_t, self.cfg.rho_linear, self.cfg.M))
        return float(np.mean(caps)) if caps else 0.0

    def _capacity_for_set_realtime(self, H_t: np.ndarray) -> float:
        """根据实时信道计算当前活跃端口集的容量"""
        if len(self.active_set) == 0:
            return 0.0
        idx = np.array(sorted(self.active_set), dtype=int)
        H_active = H_t[idx, :]
        return compute_capacity(H_active, self.cfg.rho_linear, self.cfg.M)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t_idx = 0
        # 初始化时重新采样cluster（每个episode重新采样，但episode内固定）
        self.clusters = sample_paths(self.cfg, self.rng)

        # 初始化活跃端口集（支持 random / spread / clustered / manual）
        self.active_set = self._sample_initial_active_set()

        # 实时计算初始容量（修改）
        H_t = self._compute_realtime_channel()
        self.H_current = H_t  # 初始化当前信道
        self.H_prev = H_t.copy()  # 初始化上一步信道（新增）
        self.current_capacity = self._capacity_for_set_realtime(H_t)
        self.ula_baseline_capacity = self._compute_ula_baseline_capacity()

        obs = self._get_state()
        info = {
            "capacity": self.current_capacity,
            "active_count": len(self.active_set),
            "active_ports": sorted(self.active_set),
            "legal_actions": self._get_action_list(),
        }
        return obs, info

    def step(self, action: int):
        # 记录旧容量用于计算增量
        old_capacity = float(self.current_capacity) if self.current_capacity is not None else 0.0
        action = int(action)

        # 获取当前合法动作列表
        legal = self._get_action_list()

        # --- [关键修改 1: 提前计算当前时刻信道] ---
        H_t = self._compute_realtime_channel()
        self.H_prev = self.H_current.copy() if self.H_current is not None else H_t.copy()
        self.H_current = H_t

        # 计算如果不做任何操作（保持上一时刻的端口配置），在当前新信道下的容量
        idx_old = np.array(sorted(self.active_set), dtype=int)
        cap_no_action = compute_capacity(H_t[idx_old, :], self.cfg.rho_linear, self.cfg.M)

        # --- [关键修改 2: 非法动作处理] ---
        # 如果动作非法，给予惩罚，不更新 active_set，直接返回
        if action not in legal:
            reward = -5.0  # 惩罚非法动作

            self.t_idx += 1
            terminated = self.t_idx >= self.cfg.T
            truncated = False

            # 如果结束，返回零向量，否则返回新状态
            obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._get_state()

            info = {
                "capacity": self.current_capacity,  # 容量没变
                "old_capacity": old_capacity,
                "capacity_increase": 0.0,
                "active_count": len(self.active_set),
                "active_ports": sorted(self.active_set),
                "policy_action": action,
                "chosen_action": action,  # 实际执行的是无效动作
                "is_illegal": True,  # 标记非法
                "reward_terms": {"illegal_penalty": -5.0, "total_reward": -5.0},
            }
            return obs, float(reward), terminated, truncated, info

        # --- [合法动作逻辑] ---

        # Memetic 搜索优化
        # 如果 use_memetic_search=True 且 memetic_eval_only=False，则在训练时也应用局部搜索
        refined = self._memetic_refine_action(action) if len(legal) > 0 else action
        use_refined = self.cfg.use_memetic_search and (not self.cfg.memetic_eval_only)
        executed_action = refined if use_refined else action

        # 应用动作
        self.active_set = self._apply_action(executed_action)

        # 计算新配置的容量
        self.current_capacity = self._capacity_for_set_realtime(H_t)

        # --- [奖励计算] ---
        # action_gain: 动作带来的纯粹增益（相对于不动的基准）
        action_gain = self.current_capacity - cap_no_action

        # reward_gain: 相对于上一步的时间增益（用于监控）
        reward_gain = self.current_capacity - old_capacity

        # baseline_contrast: 相对于 ULA 基准的优势
        baseline_contrast = self.current_capacity - getattr(self, 'ula_baseline_capacity', 0.0)

        # 最终奖励公式
        # 注意：这里 capacity 可能比较大，建议除以一个系数，或者沿用你原有的权重
        # 假设 capacity 范围 0-50，除以 5.0 后为 0-10，比较合理
        reward = action_gain * 5.0 + (self.current_capacity / 5.0)

        # 状态更新
        self.t_idx += 1
        terminated = self.t_idx >= self.cfg.T
        truncated = False

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            next_legal = []
        else:
            obs = self._get_state()
            next_legal = self._get_action_list()

        current_h_diff = 0.0
        if self.H_prev is not None and self.H_current is not None:
            current_h_diff = np.linalg.norm(self.H_current - self.H_prev, 'fro')

        info = {
            "capacity": self.current_capacity,
            "old_capacity": old_capacity,
            "capacity_increase": reward_gain,
            "active_count": len(self.active_set),
            "active_ports": sorted(self.active_set),
            "policy_action": action,
            "refined_action": refined,
            "chosen_action": executed_action,
            "chosen_action_desc": self.action_catalog[executed_action],
            "legal_actions": next_legal,
            "channel_change": current_h_diff,
            "is_illegal": False,
            "reward_terms": {
                "capacity_reward": float(self.current_capacity),
                "reward_gain": float(reward_gain),
                "baseline_contrast": float(baseline_contrast),
                "total_reward": float(reward),
            },
        }
        return obs, float(reward), terminated, truncated, info