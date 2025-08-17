import os
import pandas as pd
import matplotlib.pyplot as plt

# === 路径设置 ===
BASE_CSV = "./coco/eval_base_only.csv"
LORA_CSV = "./coco/eval_lora_only.csv"
SAVE_DIR = "./coco"
os.makedirs(SAVE_DIR, exist_ok=True)

# === 加载数据 ===
df_base = pd.read_csv(BASE_CSV)
df_lora = pd.read_csv(LORA_CSV)

# === 合并 DataFrame 方便统一处理 ===
df = pd.DataFrame({
    "bleu_base": df_base["bleu_base"],
    "bleu_lora": df_lora["bleu_lora"],
    "bertscore_base": df_base["bertscore_base"],
    "bertscore_lora": df_lora["bertscore_lora"],
})

x = range(len(df))

# === 计算平均值 ===
bleu_base_avg = df["bleu_base"].mean()
bleu_lora_avg = df["bleu_lora"].mean()
bertscore_base_avg = df["bertscore_base"].mean()
bertscore_lora_avg = df["bertscore_lora"].mean()

# === 图 1：BLEU 折线图 ===
plt.figure(figsize=(10, 4))
plt.plot(x, df["bleu_base"], label="BLEU Base", color="gray", linestyle="--", alpha=0.6)
plt.plot(x, df["bleu_lora"], label="BLEU LoRA", color="red", linestyle="-", alpha=0.9)
plt.title("BLEU Score per Sample")
plt.xlabel("Sample Index")
plt.ylabel("BLEU Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "bleu_curve.png"), dpi=300)
plt.close()

# === 图 2：BERTScore 折线图 ===
plt.figure(figsize=(10, 4))
plt.plot(x, df["bertscore_base"], label="BERTScore Base", color="gray", linestyle="--", alpha=0.6)
plt.plot(x, df["bertscore_lora"], label="BERTScore LoRA", color="red", linestyle="-", alpha=0.9)
plt.title("BERTScore per Sample")
plt.xlabel("Sample Index")
plt.ylabel("BERTScore")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "bertscore_curve.png"), dpi=300)
plt.close()

# === 图 3：BLEU 平均柱状图 ===
plt.figure(figsize=(6, 4))
plt.bar(["Base", "LoRA"], [bleu_base_avg, bleu_lora_avg], color=["gray", "red"])
plt.title("Average BLEU Score")
plt.ylim(0, 1)
for i, v in enumerate([bleu_base_avg, bleu_lora_avg]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "bleu_bar.png"), dpi=300)
plt.close()

# === 图 4：BERTScore 平均柱状图 ===
plt.figure(figsize=(6, 4))
plt.bar(["Base", "LoRA"], [bertscore_base_avg, bertscore_lora_avg], color=["gray", "red"])
plt.title("Average BERTScore")
plt.ylim(0, 1)
for i, v in enumerate([bertscore_base_avg, bertscore_lora_avg]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "bertscore_bar.png"), dpi=300)
plt.close()

print("✅ All plots saved to coco/ folder.")
