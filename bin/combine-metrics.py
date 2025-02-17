import pandas as pd
import glob
from pathlib import Path

def combine_metrics():
    # Get all metric files
    files = glob.glob('cordex-*-metrics.csv')
    
    # Initialize result DataFrame with column names
    columns = ['Model']
    for pred in ['WG', 'TN', 'TX', 'TP']:
        for metric in ['ME', 'MAE', 'R2']:
            columns.append(f'{pred}_{metric}')
    
    results = []
    
    # Get unique models from filenames
    models = set('-'.join(Path(f).stem.split('-')[1:4]) for f in files)
    
    # Process each model
    for model in models:
        row = {'Model': model}
        
        # Process each prediction type
        for pred in ['WG_PT24H_MAX', 'TN_PT24H_MIN', 'TX_PT24H_MAX', 'TP_PT24H_SUM']:
            prefix = pred.split('_')[0]
            filename = f'cordex-{model}-{pred}-Vuosaari-metrics.csv'
            
            try:
                df = pd.read_csv(filename)
                row[f'{prefix}_ME'] = df.iloc[0]['Mean_Error']
                row[f'{prefix}_MAE'] = df.iloc[0]['MAE']
                row[f'{prefix}_R2'] = df.iloc[0]['R2']
            except:
                row[f'{prefix}_ME'] = None
                row[f'{prefix}_MAE'] = None
                row[f'{prefix}_R2'] = None
        
        results.append(row)
    
    # Create final DataFrame and save
    df = pd.DataFrame(results)[columns]
    df.to_csv('Vuosaari_metrics.csv', index=False)

if __name__ == '__main__':
    combine_metrics()