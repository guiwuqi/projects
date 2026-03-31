from typing import List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from channel import (
    build_path_directions,
    build_rx_port_positions,
    compute_capacity,
    compute_channel_matrix,
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
        self.neighbor_index_radius = max(1, int(round(cfg.neighbor_radius_multiples_delta)))
        self.replace_shifts = [s for s in range(-self.neighbor_index_radius, self.neighbor_index_radius + 1) if s != 0]
        self.max_jump_radius = 50  # 添加最大跳变半径控制
        self.min_port_spacing_idx = max(
            1, int(np.ceil(self.cfg.min_port_spacing_in_lambda / self.cfg.delta_port_in_lambda))
        )

        # 构建动作空间
        self.action_catalog = self._build_action_catalog()
        self.action_space = spaces.Discrete(len(self.action_catalog))

        # 观察空间维度计算
        # 3维位置 + 3维速度 + 1维时间 + 1维信道变化率 + N维端口状态 + 1维容量
        obs_dim = 3 + 3 + 1 + 1 + cfg.N + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 初始化环境状态
        self.t_idx = 0
        self.active_set: Set[int] = set()
        self.current_capacity = 0.0

        # 初始化多径簇（在reset中实际生成）
        self.clusters = None

    def _build_action_catalog(self) -> List[Tuple[str, int, int]]:
        actions: List[Tuple[str, int, int]] = [("none", -1, 0)]
        # 添加无操作动作
        for n in range(self.cfg.N):
            actions.append(("add", n, 0))
        for n in range(self.cfg.N):
            actions.append(("remove", n, 0))
        for n in range(self.cfg.N):
            for shift in self.replace_shifts:
                actions.append(("replace", n, shift))
        return actions

    def _validate_manual_ports(self, ports: Tuple[int, ...]) -> List[int]:
        uniq = sorted(set(int(p) for p in ports))
        if len(uniq) == 0:
            raise ValueError("init_active_ports 不能为空（manual 模式下）")
        if len(uniq) < self.cfg.Kmin or len(uniq) > self.cfg.Kmax:
            raise ValueError(
                f"manual 端口数量必须在 [Kmin, Kmax]=[{self.cfg.Kmin}, {self.cfg.Kmax}]，当前={len(uniq)}"
            )
        if uniq[0] < 0 or uniq[-1] >= self.cfg.N:
            raise ValueError(f"manual 端口索引必须在 [0, {self.cfg.N - 1}]")
        if not self._is_spacing_legal(uniq):
            raise ValueError(
                f"manual 端口不满足最小间距约束：任意两端口索引差必须 >= {self.min_port_spacing_idx} "
                f"(约 {self.cfg.min_port_spacing_in_lambda}λ)"
            )
        return uniq

    def _is_spacing_legal(self, ports: List[int]) -> bool:
        sorted_ports = sorted(int(p) for p in ports)
        for i in range(1, len(sorted_ports)):
            if sorted_ports[i] - sorted_ports[i - 1] < self.min_port_spacing_idx:
                return False
        return True

    def _is_port_spacing_legal_with_set(self, n: int, active_set: Set[int]) -> bool:
        return all(abs(n - a) >= self.min_port_spacing_idx for a in active_set)

    def _sample_with_spacing_constraint(self, candidate_pool: np.ndarray, target_k: int) -> Set[int]:
        candidate_pool = np.array(candidate_pool, dtype=int)
        if target_k <= 0:
            return set()

        for _ in range(200):
            perm = self.rng.permutation(candidate_pool)
            selected: List[int] = []
            for p in perm:
                if all(abs(int(p) - s) >= self.min_port_spacing_idx for s in selected):
                    selected.append(int(p))
                    if len(selected) == target_k:
                        return set(selected)
        raise ValueError(
            f"无法在当前参数下采样到 {target_k} 个满足最小间距的端口。"
            "请减小 Kmax/Kmin 或减小 min_port_spacing_in_lambda。"
        )
    def _sample_initial_active_set(self) -> Set[int]:
        strategy = str(self.cfg.init_port_strategy).lower().strip()
        init_k = int(self.rng.integers(self.cfg.Kmin, self.cfg.Kmax + 1))

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

    def _is_neighbor_of_set(self, n: int, active_set: Set[int]) -> bool:
        """修改为允许更大的跳变，而不是严格相邻"""
        if len(active_set) == 0:
            return True
        min_gap = min(abs(n - a) for a in active_set)
        # 允许在最大跳变半径内的端口激活
        return min_gap <= self.max_jump_radius

    def _is_legal_action(self, action_idx: int, active_set: Optional[Set[int]] = None) -> bool:
        if active_set is None:
            active_set = self.active_set

        op, n, shift = self.action_catalog[action_idx]
        K = len(active_set)

        if op == "none":
            return True

        if op == "add":
            if K >= self.cfg.Kmax:
                return False
            if n in active_set:
                return False
            if not self._is_port_spacing_legal_with_set(n, active_set):
                return False
            return self._is_neighbor_of_set(n, active_set)  # 使用修改后的邻域判断

        if op == "remove":
            if K <= self.cfg.Kmin:
                return False
            return n in active_set

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
            if len(remaining) > 0:
                # 使用修改后的邻域判断
                return self._is_neighbor_of_set(new_n, remaining)

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
        elif op == "add":
            new_set.add(n)
        elif op == "remove":
            new_set.remove(n)
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
                if op == "add":
                    # add block starts at 1, range [1, N]
                    neighbors.add(1 + nn)
                elif op == "remove":
                    # remove block starts at 1 + N
                    neighbors.add(1 + self.cfg.N + nn)
                else:
                    # replace block starts at 2N
                    shift_id = self.replace_shifts.index(shift)
                    block = 1 + 2 * self.cfg.N + nn * len(self.replace_shifts) + shift_id
                    neighbors.add(block)

        return sorted(neighbors)

    def _memetic_refine_action(self, action_idx: int) -> int:
        if not self.cfg.use_memetic_search:
            return action_idx

        candidates = set()
        candidates.add(action_idx)
        # 显式添加 none 动作（索引 0）
        # candidates.add(0)

        for a in self._action_neighbors(action_idx):
            candidates.add(a)

        best_action = action_idx
        best_value = -np.inf
        for a in candidates:
            if not self._is_legal_action(a):
                continue
            next_set = self._apply_action(a)

            # 使用当前实时信道计算容量（修改）
            idx = np.array(sorted(next_set), dtype=int)
            H_active = self.H_current[idx, :]  # 使用当前实时信道
            c = compute_capacity(H_active, self.cfg.rho_linear, self.cfg.M)

            r = c / len(next_set)
            # r = c - self.cfg.lambda_port_penalty * len(next_set)
            if r > best_value:
                best_value = r
                best_action = a

        return best_action

    def _get_action_list(self) -> List[int]:
        return [a for a in range(self.action_space.n) if self._is_legal_action(a)]

    def _get_state(self) -> np.ndarray:
        mask = np.zeros(self.cfg.N, dtype=np.float32)
        if len(self.active_set) > 0:
            mask[list(self.active_set)] = 1.0

        # 添加时间步信息（归一化）
        normalized_time = self.t_idx / self.cfg.T  # [0,1]之间

        # 添加无人机速度（如果当前是最后一步，则速度为0）
        if self.t_idx < self.cfg.T - 1:
            velocity = (self.uav_positions[self.t_idx + 1] - self.uav_positions[self.t_idx]) / self.cfg.dt
        else:
            velocity = np.zeros(3)

        # 计算信道变化率（新增）
        h_diff = 0.0
        if self.H_prev is not None and self.H_current is not None:
            h_diff = np.linalg.norm(self.H_current - self.H_prev, 'fro')

        return np.concatenate(
            [
                self.uav_positions[self.t_idx].astype(np.float32),  # 3维位置
                velocity.astype(np.float32),  # 3维速度
                np.array([normalized_time], dtype=np.float32),  # 1维时间
                np.array([h_diff], dtype=np.float32),  # 1维信道变化率（新增）
                mask,  # N维端口状态
                np.array([self.current_capacity], dtype=np.float32),  # 1维容量
            ]
        ).astype(np.float32)

    def _compute_realtime_channel(self) -> np.ndarray:
        """实时计算当前时间步的信道矩阵"""
        # 获取当前时间步的无人机位置（注意：uav_positions是预计算的，因为无人机轨迹是确定的）
        uav_pos_t = self.uav_positions[self.t_idx]  # (3,)
        time_s = self.t_idx * self.cfg.dt
        tx_offsets = build_tx_array_offsets(self.cfg)  # (M,3)
        tx_positions = uav_pos_t[None, :] + tx_offsets  # (M,3)

        k = 2.0 * np.pi / self.cfg.wavelength
        H_t = np.zeros((self.cfg.N, self.cfg.M), dtype=np.complex128)

        for cl in self.clusters:
            alphas = cl["alphas"]
            betas = cl["betas"]
            gains = cl["gains"]
            directions = build_path_directions(alphas, betas)  # (P,3)
            rx_phase = np.exp(-1j * k * (self.rx_port_positions @ directions.T))  # (N, P)
            tx_phase = np.exp(-1j * k * (tx_positions @ directions.T))  # (M, P)
            doppler = doppler_phase(self.cfg, directions, time_s, k)
            weighted_tx = (gains * doppler)[:, None] * tx_phase.T
            H_t += rx_phase @ weighted_tx

        return H_t

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

        obs = self._get_state()
        info = {
            "capacity": self.current_capacity,
            "active_count": len(self.active_set),
            "active_ports": sorted(self.active_set),
            "legal_actions": self._get_action_list(),
        }
        return obs, info

    def step(self, action: int):
        action = int(action)
        legal = self._get_action_list()
        if len(legal) > 0 and action not in legal:
            # 非法动作时随机回退，而不是固定 no-op，避免长期“完全不变”
            action = int(self.rng.choice(np.array(legal, dtype=int)))
        refined = self._memetic_refine_action(action) if len(legal) > 0 else action
        if len(legal) > 0 and not self._is_legal_action(refined):
            refined = action

        # 保存当前配置和容量（改变前）
        old_config = self.active_set.copy()
        old_capacity = self.current_capacity

        # 应用动作（改变端口配置）
        self.active_set = self._apply_action(refined)

        # 保存当前信道矩阵作为上一步的信道
        if self.H_current is not None:
            self.H_prev = self.H_current.copy()

        # 实时计算当前时间步的信道
        H_t = self._compute_realtime_channel()
        self.H_current = H_t

        # 计算新配置的容量
        self.current_capacity = self._capacity_for_set_realtime(H_t)

        # 奖励函数：直接最大化瞬时信道容量（避免对小端口数的隐式偏置）
        reward = self.current_capacity


        self.t_idx += 1
        terminated = self.t_idx >= self.cfg.T
        truncated = False
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            next_legal = []
        else:
            obs = self._get_state()
            next_legal = self._get_action_list()

        # 计算信道变化率（用于监控）
        current_h_diff = 0.0
        if self.H_prev is not None and self.H_current is not None:
            current_h_diff = np.linalg.norm(self.H_current - self.H_prev, 'fro')

        info = {
            "capacity": self.current_capacity,
            "old_capacity": old_capacity,
            "capacity_increase": self.current_capacity - old_capacity,
            "active_count": len(self.active_set),
            "active_ports": sorted(self.active_set),
            "chosen_action": refined,
            "chosen_action_desc": self.action_catalog[refined],
            "legal_actions": next_legal,
            "channel_change": current_h_diff,
        }
        return obs, float(reward), terminated, truncated, info