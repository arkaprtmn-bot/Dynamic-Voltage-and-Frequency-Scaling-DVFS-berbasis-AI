from stable_baselines3 import PPO
from dvfs_env import DVFSEnv

# 1. Muat environment dan model yang sudah dilatih
env = DVFSEnv()
model = PPO.load("ppo_dvfs_agent")

print("\n--- Memulai Pengujian Agen DVFS ---")
obs, _ = env.reset()

# 2. Jalankan simulasi selama 1 episode (misal 50 langkah)
for step in range(50):
    # Agen memilih aksi berdasarkan state saat ini
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Cetak hasil aksi agen
    freq_names = ["Low (Hemat)", "Medium (Seimbang)", "High (Performa)"]
    print(f"Langkah {step+1}: Aksi Dipilih -> {freq_names[action]} | CPU/Queue/Latency: {obs}")
    
    if terminated or truncated:
        break

print("\nPengujian selesai!")
