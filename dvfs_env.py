import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DVFSEnv(gym.Env):
    """Custom Environment untuk Simulasi DVFS berbasis Reinforcement Learning"""
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(DVFSEnv, self).__init__()
        
        # Aksi (Action Space): Level frekuensi CPU
        # 0: Low Frequency (Hemat daya, performa rendah)
        # 1: Medium Frequency (Seimbang)
        # 2: High Frequency (Performa tinggi, boros daya)
        self.action_space = spaces.Discrete(3)
        
        # Observasi / State (Observation Space):
        # [utilisasi_cpu, panjang_antrean_beban, latensi_inferensi]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 50.0, 1000.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Inisialisasi state awal
        self.state = np.array([30.0, 5.0, 100.0], dtype=np.float32)
        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Reset state acak atau default
        self.state = np.array([np.random.uniform(20.0, 50.0), 5.0, 100.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        
        # Simulasikan efek aksi terhadap frekuensi CPU
        # Aksi 0 (Low), 1 (Med), 2 (High)
        freq_multiplier = [0.6, 1.0, 1.4][action]
        
        # Update state berdasarkan beban kerja dan aksi frekuensi
        cpu_util = np.clip(np.random.uniform(40.0, 90.0) / freq_multiplier, 0.0, 100.0)
        queue_load = np.maximum(0.0, self.state[1] + np.random.uniform(-2.0, 3.0) - (action * 2))
        latency = (queue_load * 20.0) / freq_multiplier
        
        self.state = np.array([cpu_util, queue_load, latency], dtype=np.float32)
        
        # Fungsi Reward: Keseimbangan antara penalti konsumsi daya dan latensi
        power_consumption = (freq_multiplier ** 3) * 10.0 # Daya proporsional terhadap kubik frekuensi
        latency_penalty = latency * 0.1
        reward = - (power_consumption + latency_penalty)
        
        # Kondisi selesai (terminated atau truncated)
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"Step: {self.current_step} | State (CPU, Queue, Latency): {self.state}")
