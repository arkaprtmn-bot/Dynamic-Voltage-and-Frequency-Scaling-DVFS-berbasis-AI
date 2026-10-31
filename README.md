Dynamic Voltage and Frequency Scaling (DVFS) berbasis AI: Membuat modul pengatur daya prosesor yang menggunakan Reinforcement Learning (RL) untuk menentukan kapan harus menaikkan/menurunkan performa CPU/GPU saat menjalankan inferensi AI agar hemat baterai. 


mkdir dvfs_project
cd dvfs_project
sudo apt update
sudo apt install python3-venv python3-pip -y
python3 -m venv dvfs_env
source dvfs_env/bin/activate
python3 -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium stable-baselines3 numpy pandas
from stable_baselines3 import PPO
from dvfs_env import DVFSEnv

env = DVFSEnv()
model = PPO("MlpPolicy", env, verbose=1)

print("Memulai pelatihan agen RL untuk DVFS...")
model.learn(total_timesteps=10000)

model.save("ppo_dvfs_agent")
print("Pelatihan selesai! Model berhasil disimpan sebagai ppo_dvfs_agent.zip")
python3 train.py
