import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 生成示例数据与掩码（模拟联合掩码：isfinite & target_seq_valid_mask）
L, K = 256, 2
x = np.arange(L)

# 构造目标与预测（含 NaN/Inf 模拟无效值）
y_true = (np.sin(x / 20.0)[:, None] * np.array([1.0, 0.5])[None, :])
y_pred = y_true + 0.2 * np.random.randn(L, K)
y_true[50:60, 0] = np.nan
y_pred[70:80, 1] = np.inf

# 标签有效掩码（部分时间无效）
target_seq_valid_mask = np.ones((L, K), dtype=bool)
target_seq_valid_mask[:20, :] = False
target_seq_valid_mask[200:210, 0] = False

# 联合掩码
valid = np.isfinite(y_pred) & np.isfinite(y_true) & target_seq_valid_mask

# 应用掩码：无效位置置为 NaN（绘图不连线，避免零线）
y_true_masked = y_true.copy()
y_pred_masked = y_pred.copy()
y_true_masked[~valid] = np.nan
y_pred_masked[~valid] = np.nan

# 绘图并保存
fig, axes = plt.subplots(K, 1, figsize=(10, 2 * K))
if K == 1:
    axes = [axes]
for k in range(K):
    axes[k].plot(x, y_true_masked[:, k], 'k-', label='true', linewidth=1)
    axes[k].plot(x, y_pred_masked[:, k], 'b-', label='pred', linewidth=1, alpha=0.8)
    axes[k].set_title(f'Device {k+1}')
    axes[k].legend()

os.makedirs('preview', exist_ok=True)
out_path = os.path.join('preview', 'mask_viz.png')
fig.savefig(out_path, dpi=120)
print(f'Saved: {out_path}')