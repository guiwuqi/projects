from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    # Physical constants
    c: float = 3e8
    fc: float = 28e9

    # UAV transmitter
    M: int = 4
    Tend: float = 3.0
    T: int = 30
    uav_speed: float = 10.0
    uav_start: Tuple[float, float, float] = (0.0, 0.0, 20.0)
    uav_direction_deg: Tuple[float, float] = (45.0, 15.0)  # azimuth, elevation

    # FAS receiver
    # FAS receiver
    N: int = 200
    W_in_lambda: float = 3.0
    K: int = 5
    # 初始激活端口策略:
    # - "random": 原始随机窗口采样
    # - "spread": 尽量分散
    # - "clustered": 尽量集中
    # - "manual": 使用 init_active_ports 显式指定
    init_port_strategy: str = "spread"
    # init_active_ports: Optional[Tuple[int, ...]] = None
    init_active_ports = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499)
    # clustered 模式的窗口半径（索引单位）
    init_cluster_radius: int = 50

    # IO
    out_dir: str = "results_fas_memetic_ppo"
    # FAS vs ULA 对比时的 ULA 固定端口个数（None 表示跟随 FAS 初始端口数）
    ula_fixed_port_count: Optional[int] = 5

    # NLOS channel
    L: int = 2
    Npath: int = 10
    kappa: float = 2.0
    cluster_mean_aoa_deg: Tuple[float, float, float] = (0.0, 60.0, 20.0)
    # 显式建模路径俯仰角 beta_{T,l_n}(t) 的统计中心（单位：度）
    cluster_mean_eoa_deg: Tuple[float, float, float] = (5.0, 5.0, 5.0)

    # SNR
    snr_db: float = 2.0

    # RL
    seed: int = 42
    total_timesteps: int = 20_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    clip_range: float = 0.1
    ent_coef: float = 0.05

    # Memetic local search
    use_memetic_search: bool = True
    action_neighbor_radius: int = 5
    # True: 仅评估时启用 memetic refine；训练时执行 policy 原始动作，避免 credit assignment 错位
    memetic_eval_only: bool = False

    # 最小端口间距约束（单位：lambda），例如 0.5 表示 >= lambda/2
    min_port_spacing_in_lambda: float = 0.5

    # Reward shaping（提升动作区分度）
    reward_gain_weight: float = 2.0
    reward_positive_gain_bonus: float = 5.0
    reward_contrast_weight: float = 0.5
    reward_contrast_sample_k: int = 10


    @property
    def wavelength(self) -> float:
        return self.c / self.fc

    @property
    def dt(self) -> float:
        return self.Tend / self.T

    @property
    def W(self) -> float:
        return self.W_in_lambda * self.wavelength

    @property
    def delta_port_in_lambda(self) -> float:
        return self.W_in_lambda / (self.N - 1)

    @property
    def delta_port(self) -> float:
        return self.W / (self.N - 1)

    @property
    def rho_linear(self) -> float:
        return 10 ** (self.snr_db / 10.0)