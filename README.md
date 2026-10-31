Dynamic Voltage and Frequency Scaling (DVFS) berbasis AI: Membuat modul pengatur daya prosesor yang menggunakan Reinforcement Learning (RL) untuk menentukan kapan harus menaikkan/menurunkan performa CPU/GPU saat menjalankan inferensi AI agar hemat baterai. 

Buka Terminal & Buat Folder Proyek:Buka Terminal di Ubuntu (atau klik kanan di File Manager $\rightarrow$ Open in Terminal).Buat folder proyek baru dan masuk ke foldernya:Bashmkdir dvfs_project
cd dvfs_project
Install Package python3-venv (Jika Belum Ada):Ketik perintah berikut untuk memastikan modul virtual environment siap pakai:Bashsudo apt update
sudo apt install python3-venv python3-pip -y
Membuat & Mengaktifkan Virtual Environment:Buat environment bernama dvfs_env:Bashpython3 -m venv dvfs_env
Aktifkan environment-nya:Bashsource dvfs_env/bin/activate
(Tanda berhasil: Muncul (dvfs_env) di awal baris perintah terminal).Menginstal Library yang Dibutuhkan:Jalankan perintah install library pendukung:Bashpython3 -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium stable-baselines3 numpy pandas
📍 Tahap 4: Membuat Kode Program (nano)Membuat File Lingkungan Simulasi (dvfs_env.py):Ketik: nano dvfs_env.pyMasukkan logika kelas DVFSEnv (beban CPU, suhu, power, reward).   Simpan dengan Ctrl + O $\rightarrow$ Enter, lalu keluar dengan Ctrl + X.Membuat File Pelatihan (train.py):Ketik: nano train.pyMasukkan kode training yang memanggil DVFSEnv dan algoritma PPO dari stable-baselines3:   Pythonfrom stable_baselines3 import PPO
from dvfs_env import DVFSEnv

env = DVFSEnv()
model = PPO("MlpPolicy", env, verbose=1)

print("Memulai pelatihan agen RL untuk DVFS...")
model.learn(total_timesteps=10000)

model.save("ppo_dvfs_agent")
print("Pelatihan selesai! Model berhasil disimpan sebagai ppo_dvfs_agent.zip")
Simpan dengan Ctrl + O $\rightarrow$ Enter, lalu keluar dengan Ctrl + X.📍 Tahap 5: Eksekusi & PengujianMenjalankan Pelatihan Agen:Pastikan (dvfs_env) masih aktif, lalu ketik:Bashpython3 train.py
