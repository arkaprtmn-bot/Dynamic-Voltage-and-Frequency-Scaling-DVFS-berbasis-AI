from stable_baselines3 import PPO
from dvfs_env import DVFSEnv

# 1. Inisialisasi custom environment DVFS
env = DVFSEnv()

# 2. Buat model agen Reinforcement Learning menggunakan PPO
model = PPO("MlpPolicy", env, verbose=1)

print("Memulai pelatihan agen RL untuk DVFS...")
# 3. Latih model selama 10,000 langkah (timesteps)
model.learn(total_timesteps=10000)

# 4. Simpan model yang sudah dilatih
model.save("ppo_dvfs_agent")
print("Pelatihan selesai! Model berhasil disimpan sebagai ppo_dvfs_agent.zip")
