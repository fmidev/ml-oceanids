import matplotlib.pyplot as plt


# Plot each metric-category pair in a separate plot
metric_category_pairs = [
    ('RMSE', 'WG_PT24H_MAX'),
    ('RMSE', 'TP_PT24H_SUM'),
    ('MAE', 'WG_PT24H_MAX'),
    ('MAE', 'TP_PT24H_SUM'),
]

# Colors for each plot
colors = {
    'RMSE': ['#4C72B0', '#55A868', '#C44E52', '#8172B2'],  # Distinct colors
    'MAE-WG_PT24H_MAX': ['#4C72B0', '#6D9EC1', '#98B8D9', '#BCCFE7'],  # Shades of blue
    'MAE-TP_PT24H_SUM': ['#55A868', '#79BD89', '#A1D4A5', '#C8E2C8'],  # Shades of green
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Plotting each pair
for idx, (metric, category) in enumerate(metric_category_pairs):
    ax = axes[idx]
    method_values = [rearranged_values[method][idx % 2] for method in cluster_categories]  # Select appropriate values
    color_key = metric if metric == 'RMSE' else f'{metric}-{category}'
    
    ax.bar(cluster_categories, method_values, color=colors[color_key], width=0.6)
    ax.set_title(f'{metric} for {category}', fontsize=12, weight='bold')
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xlabel('Methods', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(method_values) + 0.5)

plt.tight_layout()
plt.show()
