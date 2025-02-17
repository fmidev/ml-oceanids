import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Wind Gust Analysis",
    page_icon="🌪️",
    layout="wide"
)

# Add a title and description
st.title("Wind Gust Analysis Dashboard")
st.write("Analysis of wind gust forecasts for different locations")

# Sidebar with location selector with Vuosaari as default
st.sidebar.header("Settings")
location = st.sidebar.selectbox(
    "Select Location",
    ["Vuosaari", "Bremerhaven", "Malaga", "Raahe", "Rauma"]  # Vuosaari first
)

# Load data based on location
@st.cache_data
def load_data(location):
    location_files = {
        "Bremerhaven": "/home/ubuntu/data/ML/ECXSF_202501_WG_PT24H_MAX_Bremerhaven_004885.csv",
        "Malaga": "/home/ubuntu/data/ML/ECXSF_202501_WG_PT24H_MAX_Malaga_000231.csv",
        "Raahe": "/home/ubuntu/data/ML/ECXSF_202501_WG_PT24H_MAX_Raahe_101785.csv",
        "Rauma": "/home/ubuntu/data/ML/ECXSF_202501_WG_PT24H_MAX_Rauma_101061.csv",
        "Vuosaari": "/home/ubuntu/data/ML/ECXSF_202501_WG_PT24H_MAX_Vuosaari_151028.csv"
    }
    return pd.read_csv(location_files[location])

data = load_data(location)

# Convert valid_time to datetime
data['valid_time'] = pd.to_datetime(data['valid_time'])

# Time series analysis at the top
st.subheader("Time Series Analysis")
col1, col2 = st.columns(2)

with col1:
    # Threshold analysis
    st.write("Threshold Analysis:")
    threshold = st.slider("Wind Gust Threshold (m/s)", 
                         min_value=0,  # min value
                         max_value=30,  # max value
                         value=15,      # default value
                         step=1)        # integer steps
    
    percentage_threshold = st.slider("Probability Threshold (%)", 
                                   min_value=0,    # min value
                                   max_value=100,  # max value
                                   value=50,       # default value
                                   step=1)         # integer steps
    
    # Calculate percentage of ensemble members above threshold for each timestamp
    forecast_columns = [col for col in data.columns if 'WG_PT24H_MAX_' in col]
    percentage_above = (data[forecast_columns] > threshold).mean(axis=1) * 100
    
    # Plot percentage time series
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['valid_time'], percentage_above, 'r-', label=f'% Above {threshold} m/s')
    ax.axhline(y=percentage_threshold, color='b', linestyle='--', label=f'{percentage_threshold}% Threshold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Percentage of Ensemble Members (%)')
    ax.set_title(f'Percentage of Ensemble Members Above {threshold} m/s')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Create percentage_stats DataFrame first
    percentage_stats = pd.DataFrame({
        'Timestamp': data['valid_time'],
        'Percentage_Above': percentage_above
    })
    
    # Add monthly statistics
    st.write("\nMonthly forecasts:")
    
    # Add month column to percentage_stats
    percentage_stats['Month'] = percentage_stats['Timestamp'].dt.strftime('%Y-%m')
    
    # Calculate monthly counts
    monthly_stats = pd.DataFrame({
        f'Days >{percentage_threshold}%': percentage_stats[percentage_stats['Percentage_Above'] > percentage_threshold].groupby('Month').size(),
        'Total Days': percentage_stats.groupby('Month').size(),
        'Percentage': (percentage_stats[percentage_stats['Percentage_Above'] > percentage_threshold].groupby('Month').size() / percentage_stats.groupby('Month').size() * 100).round(1),
    }).fillna(0)
    
    # Format the columns and reorder them
    monthly_stats['Total Days'] = monthly_stats['Total Days'].astype(int)
    monthly_stats[f'Days >{percentage_threshold}%'] = monthly_stats[f'Days >{percentage_threshold}%'].astype(int)
    monthly_stats['Percentage'] = monthly_stats['Percentage'].map('{:.1f}%'.format)
    
    # Reorder columns to put Percentage before Total Days
    monthly_stats = monthly_stats[['Percentage',f'Days >{percentage_threshold}%', 'Total Days']]
    
    # Display the monthly statistics table with more height
    styled_monthly_stats = monthly_stats.style.set_properties(**{
        'text-align': 'center !important',
        'margin-left': 'auto',
        'margin-right': 'auto'
    }).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center !important')]},
        {'selector': 'td', 'props': [('text-align', 'center !important')]},
        {'selector': 'td:nth-child(1)', 'props': [
            ('font-weight', 'bold'),
            ('text-align', 'center !important')
        ]}
    ]).format(precision=0)

    st.dataframe(styled_monthly_stats, use_container_width=True, height=300)  # Increased height from 150 to 300
    
    # Show probability statistics after monthly forecasts
    st.write("\nProbability Statistics:")
    high_risk_days = percentage_stats[percentage_stats['Percentage_Above'] > percentage_threshold]
    
    if not high_risk_days.empty:
        st.write(f"Days with >{percentage_threshold}% members above {threshold} m/s:")
        st.write(high_risk_days)
    else:
        st.write(f"No days with more than {percentage_threshold}% members above {percentage_threshold} m/s")

# Load training data before statistical analysis with error handling
@st.cache_data
def load_training_data(location):
    location_files = {
        "Bremerhaven": "/home/ubuntu/data/ML/training-data/OCEANIDS/Bremerhaven/training_data_oceanids_Bremerhaven-sf-addpreds.csv",
        "Malaga": "/home/ubuntu/data/ML/training-data/OCEANIDS/Malaga/training_data_oceanids_Malaga-sf-addpreds.csv",
        "Raahe": "/home/ubuntu/data/ML/training-data/OCEANIDS/Raahe/training_data_oceanids_Raahe-sf-addpreds.csv", 
        "Rauma": "/home/ubuntu/data/ML/training-data/OCEANIDS/Rauma/training_data_oceanids_Rauma-sf-addpreds.csv",
        "Vuosaari": "/home/ubuntu/data/ML/training-data/OCEANIDS/Vuosaari/training_data_oceanids_Vuosaari-sf-addpreds.csv"
    }
    try:
        return pd.read_csv(location_files[location], parse_dates=['utctime'])
    except FileNotFoundError:
        return None

train_data = load_training_data(location)
has_observations = train_data is not None

# Statistical Analysis
st.subheader("Statistical Analysis of Seasonal and Historical Data")

# Create single distribution plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot ensemble forecasts 
for col in forecast_columns:
    member_number = int(col.split('_')[-1])
    if member_number == 0:
        color = 'black'
        sns.kdeplot(data=data, x=col, label='Control', linewidth=1.5, color=color, ax=ax)
    else:
        color = 'gray'
        sns.kdeplot(data=data, x=col, linewidth=0.5, alpha=0.3, color=color, ax=ax)

# Add observations distribution if available
if has_observations:
    sns.kdeplot(data=train_data, x='WG_PT24H_MAX', label='Historical Observations', 
                color='red', linestyle='--', linewidth=2, ax=ax)

ax.set_xlabel('Wind Gust Speed (m/s)')
ax.set_ylabel('Density')
ax.set_title('Distribution of Ensemble Members' + 
            (' vs Historical Observations' if has_observations else ''))
ax.set_xlim(0, 30)
ax.legend()

plt.tight_layout()
st.pyplot(fig)

# Add Training Data Analysis section only if data is available
if has_observations:
    st.subheader("Training Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"/home/ubuntu/data/ML/results/OCEANIDS/{location}/fscore_{location}_xgb_era5_oceanids-QE.png")
    with col2:
        st.image(f"/home/ubuntu/data/ML/results/OCEANIDS/{location}/shap_{location}_xgb_era5_oceanids-QE.png")
else:
    st.info("Training data analysis not available for this location")

# Add interactive data table
st.subheader("Raw Data")
if st.checkbox("Show raw data"):
    st.write(data)

# Add ensemble member visualization
st.subheader("Individual Ensemble Members")
fig, ax = plt.subplots(figsize=(15, 8))

# Plot each ensemble member
for column in forecast_columns:
    ax.plot(data['valid_time'], data[column], alpha=0.3, linewidth=0.5)

# Add mean line with different color and thickness
data['ensemble_mean'] = data[forecast_columns].mean(axis=1)
ax.plot(data['valid_time'], data['ensemble_mean'], 'r-', 
        linewidth=2, label='Ensemble Mean')

ax.set_xlabel('Date')
ax.set_ylabel('Wind Gust (m/s)')
ax.set_title('All Ensemble Members Over Time')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Add heatmap visualization
st.subheader("Ensemble Heatmap")
fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(data[forecast_columns].T, 
            cmap='YlOrRd',
            xticklabels=50,
            yticklabels=True)
plt.xlabel('Time Step')
plt.ylabel('Ensemble Member')
plt.title('Heatmap of Ensemble Members')
plt.tight_layout()
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Wind Gust Analysis Tool - Built with Streamlit ❤️")