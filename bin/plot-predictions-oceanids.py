import pandas as pd
import matplotlib.pyplot as plt
import sys

file_path = sys.argv[1]
pred = file_path.split('-')[-5]
print(pred)

# Read the CSV file
df = pd.read_csv(file_path)

# Set the 'utctime' column as the index
df.set_index('utctime', inplace=True)

# Plot the data
ax = df.drop(columns=[pred,'Predicted','pr_sum_mean']).plot(figsize=(10, 6), alpha=0.75)  # Plot all columns except 'pred' and 'Predicted'
df['Predicted'].plot(ax=ax, color='red', linewidth=2)  # Plot 'Predicted' column in the foreground
df[pred].plot(ax=ax, color='blue', linewidth=2, alpha=0.5)  # Plot 'pred' column in the foreground with adjusted opacity
df['pr_sum_mean'].plot(ax=ax, color='pink', linewidth=2, alpha=1)  # Plot 'tasmax_sum_mean' column in the foreground with adjusted opacity



plt.title('Predictions vs training data')
plt.xlabel('Date')
plt.ylabel(pred)
plt.legend(title='Columns')
plt.grid(True)
plt.show()
plt.savefig(file_path.replace('.csv', '.png'))