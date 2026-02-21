# fas_environment.py
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.special import iv

# 固定参数
Q = 30  # 端口总数固定


def reward_function(state, C, delta_error):
    """改进的奖励函数，包含模型误差"""
    # 标准化容量奖励
    capacity_norm = 0.1 * C

    # 模型误差惩罚 (论文公式14，误差越小越好)
    # 注意：delta_error 是负的dB值，所以我们需要将其转为正的惩罚项
    error_penalty = -0.1 * abs(delta_error)  # 误差越大惩罚越大

    # 改进间距惩罚
    if state['delta_d'] < 0.01:  # 过小间距
        spacing_penalty = -10.0 * (0.3 - state['delta_d'])
    elif state['delta_d'] > 2.0:  # 较大但可接受
        spacing_penalty = -10.0 * (state['delta_d'] - 2.0)
    else:  # 理想范围 [0.01, 2.0]
        spacing_penalty = 0.0  # 适当奖励

    total_reward = (
            capacity_norm +  # 主要奖励
            error_penalty +  # 模型误差项
            spacing_penalty
    )

    if state['delta_d'] < 0.01 or state['delta_d'] > 2.0:
        total_reward -= 8.0  # 严重违反物理约束

    return total_reward


class FASA2GEnv(gym.Env):
    def __init__(self, use_local_search=False, search_radius=1):
        super(FASA2GEnv, self).__init__()
        self.use_local_search = use_local_search
        self.search_radius = search_radius

        # 添加固定角度参数
        self.psi_R_fixed = np.pi / 3  # 固定水平偏转角
        self.theta_R_fixed = np.pi / 3  # 固定竖直偏转角

        # 修改状态空间 - 移除角度变量
        self.observation_space = spaces.Dict({
            'W': spaces.Box(low=2, high=20.0, shape=(1,), dtype=np.float32),
            'Q_sub': spaces.Box(low=1, high=30, shape=(1,), dtype=np.int32),
            'delta_d': spaces.Box(low=0.01, high=2.0, shape=(1,), dtype=np.float32),
            'SNR': spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32)  # 移除了角度变量
        })

        # 修改动作空间 - 移除角度配置动作
        self.action_space = spaces.MultiDiscrete([5, 7])  # 从[5,7,15]改为[5,7]

        # 固定参数 (基于论文)
        self.v_T = 3.0  # 固定无人机速度
        self.P = 4  # UAV天线数
        self.H0 = 20.0  # 初始高度
        self.D0 = 50.0  # 初始距离
        self.fc = 5e9  # 载波频率
        self.lambda_wave = 3e8 / self.fc  # 波长
        self.c = 3e8  # 光速

        # UAV天线参数
        self.psi_T = np.pi / 2  # UAV方位角
        self.theta_T = np.pi / 3  # UAV仰角
        self.delta_T = self.lambda_wave / 2  # UAV天线间距
        self.gamma_T = np.pi / 3  # 运动方位角
        self.eta_T = np.pi / 3  # 运动仰角

        # 散射体参数
        self.L = 3  # 簇数量
        self.N_per_cluster = 10  # 每簇路径数
        self.kappa = 2.0  # von Mises分布参数

        self.max_steps = 64
        self.current_step = 0

        # 预计算散射体位置
        self._generate_scatterers()

        self.reset()

    def _generate_scatterers(self):
        """生成散射体位置（基于von Mises分布）"""
        self.scatterers = []
        self.scatterer_angles = []  # 存储每个散射体的AoD和AoA角度

        for l in range(self.L):
            cluster_scatterers = []
            cluster_angles = []
            # 簇中心角度
            mu_alpha = np.random.uniform(0, 2 * np.pi)
            mu_beta = np.random.uniform(0, np.pi)

            for n in range(self.N_per_cluster):
                # 使用von Mises分布生成角度
                alpha = self._sample_von_mises(mu_alpha, self.kappa)
                beta = self._sample_von_mises(mu_beta, self.kappa)

                # 生成随机相位 φ_{ℓn} ~ U[-π, π)
                phi = np.random.uniform(-np.pi, np.pi)

                # 散射体距离 - 假设在50-100米范围内
                distance = np.random.uniform(50, 100)

                # 转换为3D坐标
                x = self.D0 + distance * np.cos(alpha) * np.cos(beta)
                y = distance * np.sin(alpha) * np.cos(beta)
                z = distance * np.sin(beta)

                scatterer_pos = np.array([x, y, z])
                cluster_scatterers.append(scatterer_pos)

                # 存储散射体角度
                cluster_angles.append({
                    'alpha_R': alpha,  # 到达方位角
                    'beta_R': beta,  # 到达仰角
                    'phi': phi  # 随机相位
                })

            self.scatterers.append(cluster_scatterers)
            self.scatterer_angles.append(cluster_angles)

    def _sample_von_mises(self, mu, kappa):
        """从von Mises分布中采样"""
        from scipy.stats import vonmises
        return vonmises.rvs(kappa, loc=mu)

    def _calculate_ULA_baseline(self, Q_sub):
        """计算ULA基线模型信道矩阵，按照论文公式(8)的正确实现"""
        P = self.P
        t = 2.0
        delta_R = 0.5  # ULA端口间距固定为半波长

        H_ULA = np.zeros((Q_sub, P), dtype=complex)

        # UAV在t时刻的位置
        d_T = self.v_T * t * np.array([
            np.cos(self.eta_T) * np.cos(self.gamma_T),
            np.cos(self.eta_T) * np.sin(self.gamma_T),
            np.sin(self.eta_T)
        ])

        for p in range(P):
            # UAV天线元素位置 (公式3)
            k_p = (P - 2 * p + 1) / 2
            d_T_p = k_p * self.delta_T * np.array([
                np.cos(self.theta_T) * np.cos(self.psi_T),
                np.cos(self.theta_T) * np.sin(self.psi_T),
                np.sin(self.theta_T)
            ])
            d_p = d_T_p - d_T

            for q in range(Q_sub):
                k_q = (Q_sub - 2 * q + 1) / 2

                # ULA端口位置（使用固定角度）
                d_q = k_q * delta_R * np.array([
                    np.cos(self.theta_R_fixed) * np.cos(self.psi_R_fixed),  # 使用固定值
                    np.cos(self.theta_R_fixed) * np.sin(self.psi_R_fixed),  # 使用固定值
                    np.sin(self.theta_R_fixed)  # 使用固定值
                ]) - np.array([self.D0, 0, 0])

                # 按照论文公式(8)计算信道
                cluster_sum = 0j

                for l_idx, cluster in enumerate(self.scatterers):
                    for n_idx, scatterer in enumerate(cluster):
                        # 获取存储的角度和相位
                        angles = self.scatterer_angles[l_idx][n_idx]
                        phi = angles['phi']

                        # 计算距离（论文中的ξ_T和ξ_R）
                        xi_T = np.linalg.norm(scatterer - d_p)  # ξ_T,ℓn(t)
                        xi_R = np.linalg.norm(d_q - scatterer)  # ξ_R,ℓn

                        # 计算从UAV到散射体的角度（AoD）
                        vec_T = scatterer - d_p
                        dist_T = np.linalg.norm(vec_T)
                        if dist_T > 0:
                            unit_vec_T = vec_T / dist_T
                            # 全局坐标系下的角度
                            alpha_T_global = np.arctan2(unit_vec_T[1], unit_vec_T[0])
                            beta_T_global = np.arcsin(np.clip(unit_vec_T[2], -1.0, 1.0))

                            # 相对角度（论文公式8中使用了相对角度）
                            alpha_T = alpha_T_global - self.psi_T
                        else:
                            alpha_T = 0
                            beta_T_global = 0

                        # 计算从散射体到ULA端口的角度（使用固定角度）
                        vec_R = d_q - scatterer
                        dist_R = np.linalg.norm(vec_R)
                        if dist_R > 0:
                            # 使用存储的到达角和固定接收角度
                            alpha_R = angles['alpha_R'] - self.psi_R_fixed  # 使用固定值
                            beta_R = angles['beta_R']
                        else:
                            alpha_R = 0
                            beta_R = 0

                        # 按照论文公式(8)计算相位项
                        # 1. 随机相位和距离相位
                        phase1 = phi - (2 * np.pi / self.lambda_wave) * (xi_T + xi_R)

                        # 2. UAV天线阵列相位
                        phase2 = (2 * np.pi / self.lambda_wave) * k_p * self.delta_T * (
                                np.cos(alpha_T) * np.cos(beta_T_global) * np.cos(self.theta_T) +
                                np.sin(beta_T_global) * np.sin(self.theta_T)
                        )

                        # 3. ULA端口阵列相位（使用固定角度）
                        phase3 = (2 * np.pi / self.lambda_wave) * k_q * delta_R * (
                                np.cos(alpha_R) * np.cos(beta_R) * np.cos(self.theta_R_fixed) +  # 使用固定值
                                np.sin(beta_R) * np.sin(self.theta_R_fixed)  # 使用固定值
                        )

                        # 4. UAV运动相位
                        phase4 = (2 * np.pi / self.lambda_wave) * self.v_T * t * (
                                np.cos(alpha_T_global - self.gamma_T) * np.cos(beta_T_global) * np.cos(self.eta_T) +
                                np.sin(beta_T_global) * np.sin(self.eta_T)
                        )

                        # 总相位
                        total_phase = phase1 + phase2 + phase3 + phase4

                        # 累加
                        cluster_sum += np.exp(1j * total_phase)

                # 归一化
                H_ULA[q, p] = cluster_sum / (self.L * self.N_per_cluster)

        return H_ULA

    def _calculate_modeling_error(self, H_FAS):
        """
        按照论文公式(10)正确计算模型误差：
        δ_error = 10 log10( Σ_{p=1}^P Σ_{q=1}^{Q_sub} |h_FAS - h_ULA| / |h_ULA| )
        修正：使用正确的求和和归一化
        """
        Q_sub = int(self.state['Q_sub'][0])

        # 计算ULA基线模型
        H_ULA = self._calculate_ULA_baseline(Q_sub)

        # 避免除以零
        H_ULA_abs = np.abs(H_ULA)
        H_ULA_abs[H_ULA_abs == 0] = 1e-12

        # 计算每个元素的相对误差
        abs_diff = np.abs(H_FAS - H_ULA)
        relative_error = abs_diff / H_ULA_abs

        # 对所有元素求和（论文中的双重求和）
        sum_relative_error = np.sum(relative_error)

        # 计算对数误差（论文公式10）
        if sum_relative_error > 0:
            delta_error = 10 * np.log10(sum_relative_error)
        else:
            delta_error = -np.inf  # 如果误差为0，返回负无穷

        return np.real(delta_error)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = {
            'W': np.array([10.0], dtype=np.float32),
            'Q_sub': np.array([15], dtype=np.int32),
            'delta_d': np.array([0.5], dtype=np.float32),
            'SNR': np.array([10], dtype=np.float32)  # 移除了角度变量
        }
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        raw_action = np.array(action, dtype=np.int32)

        # 改进版：在PPO动作基础上执行局部搜索精修动作
        if self.use_local_search:
            action = self._refine_action_with_local_search(raw_action)
        else:
            action = raw_action

        # 执行动作，更新状态
        action_dict = {
            'W_change': int(action[0]),
            'Q_sub_change': int(action[1])
        }
        self._apply_action(action_dict)

        # 计算信道容量
        H_FAS = self._calculate_channel_matrix()
        C = self._calculate_channel_capacity(H_FAS)

        # 计算模型误差
        delta_error = self._calculate_modeling_error(H_FAS)

        # 计算奖励 (现在包含模型误差)
        reward = self._calculate_reward(C, delta_error)

        # 检查是否结束
        terminated = self._check_done()
        truncated = False  # 可以根据需要设置截断条件

        self.current_step += 1

        info = {
            'channel_capacity': C,
            'modeling_error': delta_error,
            'raw_action': raw_action,
            'executed_action': action,
            'local_search_enabled': self.use_local_search
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _evaluate_action_reward(self, action):
        """仅用于局部搜索：计算动作即时奖励，不改变环境状态。"""
        state_backup = {
            key: np.copy(value) for key, value in self.state.items()
        }
        current_step_backup = self.current_step

        action_dict = {
            'W_change': int(action[0]),
            'Q_sub_change': int(action[1])
        }
        self._apply_action(action_dict)
        H_FAS = self._calculate_channel_matrix()
        C = self._calculate_channel_capacity(H_FAS)
        delta_error = self._calculate_modeling_error(H_FAS)
        reward = self._calculate_reward(C, delta_error)

        self.state = state_backup
        self.current_step = current_step_backup
        return reward

    def _refine_action_with_local_search(self, action):
        """
        PPO + 局部搜索（Memetic PPO）:
        在离散动作邻域内挑选即时奖励最大的动作。
        """
        best_action = np.array(action, dtype=np.int32)
        best_reward = self._evaluate_action_reward(best_action)

        action0_max = self.action_space.nvec[0] - 1
        action1_max = self.action_space.nvec[1] - 1

        for dw in range(-self.search_radius, self.search_radius + 1):
            for dq in range(-self.search_radius, self.search_radius + 1):
                candidate = np.array([
                    np.clip(best_action[0] + dw, 0, action0_max),
                    np.clip(best_action[1] + dq, 0, action1_max)
                ], dtype=np.int32)

                candidate_reward = self._evaluate_action_reward(candidate)
                if candidate_reward > best_reward:
                    best_reward = candidate_reward
                    best_action = candidate

        return best_action

    def _apply_action(self, action):
        """根据动作更新状态 - 移除角度更新部分"""
        # action现在是一个字典，包含'W_change'和'Q_sub_change'键

        # W_change: [0,1,2,3,4] -> [-0.2, -0.05, 0, 0.05, 0.2]
        w_changes = [-2, -1, 0, 1, 2]
        w_change = w_changes[action['W_change']]  # 使用字典键访问
        min_W = 0.01 * (Q - 1)
        new_W = np.clip(self.state['W'][0] + w_change, min_W, 5.0)
        self.state['W'] = np.array([new_W], dtype=np.float32)

        # 更新端口间距
        normalized_spacing = new_W / (Q - 1)
        self.state['delta_d'] = np.array([normalized_spacing], dtype=np.float32)

        # Q_sub_change: [0,1,2,3,4,5,6] -> [-3,-2,-1,0,+1,+2,+3]
        q_sub_changes = [-3, -2, -1, 0, 1, 2, 3]
        q_sub_change = q_sub_changes[action['Q_sub_change']]  # 使用字典键访问
        new_Q_sub = np.clip(self.state['Q_sub'][0] + q_sub_change, 1, Q)
        self.state['Q_sub'] = np.array([new_Q_sub], dtype=np.int32)

        # 移除了角度更新部分

    def _get_obs(self):
        """返回当前观测值 - 移除角度变量"""
        return {
            'W': self.state['W'].astype(np.float32),
            'Q_sub': self.state['Q_sub'].astype(np.int32),
            'delta_d': self.state['delta_d'].astype(np.float32),
            'SNR': self.state['SNR'].astype(np.float32)
        }

    def _calculate_reward(self, C, delta_error):
        """计算奖励，现在包含模型误差"""
        return reward_function({
            'Q_sub': self.state['Q_sub'][0],
            'delta_d': self.state['delta_d'][0]
        }, C, delta_error)

    def _check_done(self):
        """检查是否结束"""
        return self.current_step >= self.max_steps

    def _calculate_channel_capacity(self, H):
        Q_sub = int(self.state['Q_sub'][0])
        SNR_linear = 10

        # 计算 H * H^H（Q_sub × Q_sub）
        HH = H @ H.conj().T

        # 正则化
        HH += 1e-8 * np.eye(Q_sub)

        # 计算容量
        I = np.eye(Q_sub)
        C = np.log2(np.linalg.det(I + (SNR_linear / self.P) * HH))

        return np.real(C)

    def _calculate_channel_matrix(self):
        """计算完整信道矩阵 - 按照论文公式(8)实现"""
        return self._calculate_NLoS_component()

    def _calculate_NLoS_component(self):
        """
        按照论文公式(8)实现NLoS信道矩阵计算
        """
        P = self.P
        Q_sub = int(self.state['Q_sub'][0])
        t = 2.0  # 固定时间

        H = np.zeros((Q_sub, P), dtype=complex)

        # UAV在t时刻的位置（公式4）
        d_T = self.v_T * t * np.array([
            np.cos(self.eta_T) * np.cos(self.gamma_T),
            np.cos(self.eta_T) * np.sin(self.gamma_T),
            np.sin(self.eta_T)
        ])

        for p in range(P):
            # UAV天线元素位置（公式2和3）
            k_p = (P - 2 * p + 1) / 2
            d_T_p = k_p * self.delta_T * np.array([
                np.cos(self.theta_T) * np.cos(self.psi_T),
                np.cos(self.theta_T) * np.sin(self.psi_T),
                np.sin(self.theta_T)
            ])
            d_p = d_T_p - d_T

            for q in range(Q_sub):
                # FAS端口位置（使用固定角度）
                k_q = (Q_sub - 2 * q + 1) / 2
                d_q = k_q * self.state['delta_d'][0] * np.array([
                    np.cos(self.theta_R_fixed) * np.cos(self.psi_R_fixed),  # 使用固定值
                    np.cos(self.theta_R_fixed) * np.sin(self.psi_R_fixed),  # 使用固定值
                    np.sin(self.theta_R_fixed)  # 使用固定值
                ]) - np.array([self.D0, 0, 0])

                # 按照论文公式(8)计算信道
                cluster_sum = 0j

                for l_idx, cluster in enumerate(self.scatterers):
                    for n_idx, scatterer in enumerate(cluster):
                        # 获取存储的角度和相位
                        angles = self.scatterer_angles[l_idx][n_idx]
                        phi = angles['phi']

                        # 计算距离（论文中的ξ_T和ξ_R）
                        xi_T = np.linalg.norm(scatterer - d_p)  # ξ_T,ℓn(t)
                        xi_R = np.linalg.norm(d_q - scatterer)  # ξ_R,ℓn

                        # 计算从UAV到散射体的角度（AoD）
                        vec_T = scatterer - d_p
                        dist_T = np.linalg.norm(vec_T)
                        if dist_T > 0:
                            unit_vec_T = vec_T / dist_T
                            # 全局坐标系下的角度
                            alpha_T_global = np.arctan2(unit_vec_T[1], unit_vec_T[0])
                            beta_T_global = np.arcsin(np.clip(unit_vec_T[2], -1.0, 1.0))

                            # 相对角度（论文公式8中使用了相对角度）
                            alpha_T = alpha_T_global - self.psi_T
                        else:
                            alpha_T = 0
                            beta_T_global = 0

                        # 计算从散射体到FAS端口的角度（使用固定角度）—— 移到外面
                        vec_R = d_q - scatterer
                        dist_R = np.linalg.norm(vec_R)
                        if dist_R > 0:
                            # 使用存储的到达角和固定接收角度
                            alpha_R = angles['alpha_R'] - self.psi_R_fixed  # 使用固定值
                            beta_R = angles['beta_R']
                        else:
                            alpha_R = 0
                            beta_R = 0

                        # 按照论文公式(8)计算相位项
                        # 1. 随机相位和距离相位
                        phase1 = phi - (2 * np.pi / self.lambda_wave) * (xi_T + xi_R)

                        # 2. UAV天线阵列相位
                        phase2 = (2 * np.pi / self.lambda_wave) * k_p * self.delta_T * (
                                np.cos(alpha_T) * np.cos(beta_T_global) * np.cos(self.theta_T) +
                                np.sin(beta_T_global) * np.sin(self.theta_T)
                        )

                        # 3. FAS端口阵列相位（使用固定角度）
                        phase3 = (2 * np.pi / self.lambda_wave) * k_q * self.state['delta_d'][0] * (
                                np.cos(alpha_R) * np.cos(beta_R) * np.cos(self.theta_R_fixed) +  # 使用固定值
                                np.sin(beta_R) * np.sin(self.theta_R_fixed)  # 使用固定值
                        )

                        # 4. UAV运动相位
                        phase4 = (2 * np.pi / self.lambda_wave) * self.v_T * t * (
                                np.cos(alpha_T_global - self.gamma_T) * np.cos(beta_T_global) * np.cos(self.eta_T) +
                                np.sin(beta_T_global) * np.sin(self.eta_T)
                        )

                        # 总相位
                        total_phase = phase1 + phase2 + phase3 + phase4

                        # 累加
                        cluster_sum += np.exp(1j * total_phase)

                    # 归一化
                    H[q, p] = cluster_sum / (self.L * self.N_per_cluster)

        # 信道矩阵归一化（论文要求的normalized channel matrix）
        # 使平均功率为1
        power = np.mean(np.abs(H) ** 2)
        if power > 0:
            H = H / np.sqrt(power)

        return H
