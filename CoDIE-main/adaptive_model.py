import subprocess
import itertools

# 参数范围配置
alphas = [0.8, 1.0, 1.2]
betas  = [0.05, 0.1, 0.15]
gammas = [0.3, 0.5, 0.7]
deltas = [0.3, 0.5, 0.7]

param_combinations = list(itertools.product(alphas, betas, gammas, deltas))
log_dir = "logs/"
script_path = "F:\\ayejiantuxiangronghe\diguangtuxiangronghe\colie-main\colie-main\hybrid_main_dehaze.py"
input_folder = "F:\\ayejiantuxiangronghe\diguangtuxiangronghe\colie-main\colie-main\input"
output_folder = "F:\\ayejiantuxiangronghe\diguangtuxiangronghe\colie-main\colie-main\onput"

for i, (a, b, g, d) in enumerate(param_combinations):
    print(f"🔧 测试组合 [{i+1}/{len(param_combinations)}] alpha={a}, beta={b}, gamma={g}, delta={d}")
    log_file = f"{log_dir}/log_{i+1}_α{a}_β{b}_γ{g}_δ{d}.txt"

    command = [
        "python", script_path,
        "--input_folder", input_folder,
        "--output_folder", output_folder,
        "--alpha", str(a),
        "--beta", str(b),
        "--gamma", str(g),
        "--delta", str(d)
    ]

    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
