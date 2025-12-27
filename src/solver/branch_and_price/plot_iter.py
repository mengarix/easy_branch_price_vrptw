import matplotlib.pyplot as plt

class IterRecord:
    def __init__(self, iter_idx, best_obj):
        self.iter_idx = iter_idx
        self.best_obj = best_obj
    
    def __repr__(self):
        return f"IterRecord(iter_idx={self.iter_idx}, best_obj={self.best_obj})"

def plot_iter_records(records, buffer_ratio=0.2, plot_id=""):
    # 提取原始数据
    x = [r.iter_idx for r in records]
    y = [r.best_obj for r in records]

    # 检测是否存在极大值（异常值）
    non_infinite = [v for v in y if v < max(y)]
    if len(non_infinite) == 0:
        max_normal = max(y)
    else:
        max_normal = max(non_infinite)
    visual_max = max_normal * (1 + buffer_ratio)

    # 判断哪些值需要压缩
    y_plot = [min(val, visual_max) for val in y]

    # 作图
    plt.figure(figsize=(6, 3))
    
    plt.plot(x, y_plot, markersize=2, marker='o', label='lower bound')

    # 标记被压缩的点
    for i, (x_i, y_i, raw_y) in enumerate(zip(x, y_plot, y)):
        if raw_y >= visual_max:
            plt.scatter(x_i, y_i, s=5, color='red', marker='o', label='Compressed' if i == 0 else "", zorder=10)
            if i == 0:
                plt.text(x_i, y_i+0.5, f"{raw_y:.0e}", ha='center', va='bottom', fontsize=8, color='red')
    
    # 特别标注最后一个点的值
    last_x = x[-1]
    last_y = y_plot[-1]
    last_raw_y = y[-1]

    # 动态计算文本偏移量（基于数据范围）
    y_range = max(y_plot) - min(y_plot)
    y_offset = y_range * 0.05 if y_range > 0 else 0.5
    
    # 根据点是否被压缩决定颜色
    color = 'red' if last_raw_y >= visual_max else 'black'
    
    # 添加标注（使用箭头连接）
    plt.annotate(f"{last_raw_y}", 
                 xy=(last_x, last_y),
                 xytext=(last_x, last_y + y_offset),
                 ha='center', va='bottom',
                 fontsize=9, color=color,
                 arrowprops=dict(arrowstyle='->', color=color, linewidth=0.8))

    plt.xlabel("Iteration",fontsize=8)
    plt.ylabel("Lower Bound",fontsize=8)
    plt.title(f"Lower bound over Iterations ({plot_id})", fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'bp_plot_{plot_id}.png')
    plt.show()
# 示例使用
if __name__ == "__main__":
    records = [
        IterRecord(0, 1000000.0),
        IterRecord(1, 1000000.0),
        IterRecord(2, 1000000.0),
        IterRecord(3, 1000000.0),
        IterRecord(4, 1000000.0),
        IterRecord(5, 75.1),
        IterRecord(6, 70.5),
        IterRecord(7, 65.8),
        IterRecord(8, 60.2),
        IterRecord(9, 55.7),
        IterRecord(10, 50.3),
    ]

    plot_iter_records(records)