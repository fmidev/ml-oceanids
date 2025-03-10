import pandas as pd
import sklearn.metrics
import sys
import os

file_name = sys.argv[1]

dir = '/home/ubuntu/data/ML/results/OCEANIDS/cordex/'

# Extract model name, prediction type, and location from the file name
parts = file_name.split('-')
model = parts[1]+'-'+parts[2]
pred = parts[4]
location = parts[3]

data = pd.read_csv(dir+file_name)
data = data.dropna()

# Calculate the errors
data['error'] = data['Predicted'] - data[pred]

mean_error = data['error'].mean()
mae = sklearn.metrics.mean_absolute_error(data[pred], data['Predicted'])
r2 = sklearn.metrics.r2_score(data[pred], data['Predicted'])
variance = sklearn.metrics.explained_variance_score(data[pred], data['Predicted'])

# Save the results to a new CSV file
results = pd.DataFrame({
    'Model': [model],
    'Pred': [pred],
    'Location': [location],
    'Mean_Error': [mean_error],
    'MAE': [mae],
    'R2': [r2],
    'Explained_Variance': [variance]
})

output_file = f'cordex-{model}-{pred}-{location}-metrics.csv'
results.to_csv(f'/home/ubuntu/data/ML/results/OCEANIDS/metrics/{output_file}', index=False)

print(f'Results saved to {output_file}')