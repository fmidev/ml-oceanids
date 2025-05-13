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

# NEW: Define sidebar selectboxes only once with unique keys as needed
st.sidebar.header("Settings")
forecast_type = st.sidebar.selectbox(
    "Select Forecast Type",
    ["Wind Gust Max", "Max Temperature", "Min Temperature", "Total Precipitation"],
    index=0,
    key="forecast_type"
)
location = st.sidebar.selectbox(
    "Select Location",
    [ "Raahe", "Rauma", "Vuosaari", "Antwerpen", "Bremerhaven", "Malaga-Puerto", "PontaDelgada", "PraiaDaVittoria", "Saint-Guenole", "Plaisance" ],
    key="location"
)
# Update: Added "March" to prediction month options, now including "April"
pred_month = st.sidebar.selectbox(
    "Select Prediction Month",
    ["January", "February", "March", "April"],
    index=1,
    key="pred_month"
)

# Update: Compute month code based on selection
if pred_month == "January":
    month_code = "202501"
elif pred_month == "February":
    month_code = "202502"
elif pred_month == "March":
    month_code = "202503"
elif pred_month == "April":
    month_code = "202504"

# NEW: Display selected harbor plot in the sidebar using use_container_width
harbor_plot = f"/home/ubuntu/data/ML/results/OCEANIDS/{location}/{location}_training-locs.png"
st.sidebar.markdown("### Harbor Plot")
st.sidebar.image(harbor_plot, use_container_width=True)

# Now that variables are defined, update title dynamically
st.title(f"{forecast_type} Analysis Dashboard")
st.write("Analysis of forecasts for different locations")

# Update: Recompute month code based on selection (duplicate block updated similarly)
if pred_month == "January":
    month_code = "202501"
elif pred_month == "February":
    month_code = "202502"
elif pred_month == "March":
    month_code = "202503"
elif pred_month == "April":
    month_code = "202504"

forecast_map = {
    "Wind Gust Max": "WG_PT24H_MAX",
    "Max Temperature": "TA_PT24H_MAX",
    "Min Temperature": "TA_PT24H_MIN",
    "Total Precipitation": "TP_PT24H_ACC"
}
forecast_prefix = forecast_map[forecast_type]

# Display location image
location_images = {
    "Bremerhaven": "/home/ubuntu/ml-oceanids/Bremerhaven_points.png",
    "Malaga": "/home/ubuntu/ml-oceanids/Malaga_points.png",
    "Raahe": "/home/ubuntu/ml-oceanids/Raahe_points.png",
        "Rauma": "/home/ubuntu/ml-oceanids/Rauma.png",
        "Vuosaari": "/home/ubuntu/ml-oceanids/Vuosaari_points.png"
    }
def load_data(location):
    # Dynamically build file path based on selected month, forecast type and location
    file_path = f"/home/ubuntu/data/ML/ECXSF_{month_code}_{forecast_prefix}_{location}.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Data file for {location} ({pred_month}) not found.")
        return pd.DataFrame()

data = load_data(location)

# NEW: Check for the 'utctime' column. If missing, notify the user and stop execution.
if 'utctime' in data.columns:
    data['utctime'] = pd.to_datetime(data['utctime'])
else:
    st.error("The required 'utctime' column is missing in the loaded data. Please check your data source.")
    st.stop()

# Time series analysis at the top
st.subheader("Time Series Analysis")
col1, col2 = st.columns(2)

# After loading data, define forecast_columns using the selected prefix
forecast_columns = [col for col in data.columns if col.startswith(f"{forecast_prefix}_")]

with col1:
    # NEW: Set slider parameters and operator based on forecast type
    if forecast_type == "Wind Gust Max":
        threshold_text = "Wind Gust Threshold (m/s)"
        slider_params = {"min_value": 0, "max_value": 30, "value": 15, "step": 1}
        calc_operator = "above"
        unit_text = "m/s"
    elif forecast_type == "Max Temperature":
        threshold_text = "Temperature Threshold (°C)"
        slider_params = {"min_value": -30, "max_value": 50, "value": 20, "step": 1}
        calc_operator = "above"
        unit_text = "°C"
    elif forecast_type == "Min Temperature":
        threshold_text = "Temperature Threshold (°C)"
        slider_params = {"min_value": -30, "max_value": 50, "value": 10, "step": 1}
        calc_operator = "below"
        unit_text = "°C"
    elif forecast_type == "Total Precipitation":
        threshold_text = "Precipitation Threshold (mm)"
        slider_params = {"min_value": 0, "max_value": 100, "value": 50, "step": 1}
        calc_operator = "above"
        unit_text = "mm"
    
    st.write("Threshold Analysis:")
    threshold = st.slider(threshold_text, **slider_params)
    
    percentage_threshold = st.slider("Probability Threshold (%)", min_value=0, max_value=100, value=50, step=1)
    
    if calc_operator == "above":
        percentage_above = (data[forecast_columns] > threshold).mean(axis=1) * 100
        line_label = f"% Above {threshold} {unit_text}"
    else:
        percentage_above = (data[forecast_columns] < threshold).mean(axis=1) * 100
        line_label = f"% Below {threshold} {unit_text}"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['utctime'], percentage_above, 'r-', label=line_label)
    ax.axhline(y=percentage_threshold, color='b', linestyle='--', label=f'{percentage_threshold}% Threshold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Percentage of Ensemble Members (%)')
    ax.set_title(f'Percentage of Ensemble Members {"Above" if calc_operator=="above" else "Below"} {threshold} {unit_text}')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Create percentage_stats DataFrame first
    percentage_stats = pd.DataFrame({
        'Timestamp': data['utctime'],
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
        if calc_operator == "above":
            st.write(f"Days with >{percentage_threshold}% members above {threshold} {unit_text}:")
        else:
            st.write(f"Days with >{percentage_threshold}% members below {threshold} {unit_text}:")
        st.write(high_risk_days)
    else:
        st.write(f"No days with more than {percentage_threshold}% members " +
                 (f"above {threshold} {unit_text}" if calc_operator=="above" else f"below {threshold} {unit_text}"))

# Load training data before statistical analysis with error handling
@st.cache_data
def load_training_data(location):
    location_files = {
        "Antwerpen": "/home/ubuntu/data/ML/training-data/OCEANIDS/Antwerpen/training_data_oceanids_Antwerpen-sf-addpreds.csv.gz",
        "Bremerhaven": "/home/ubuntu/data/ML/training-data/OCEANIDS/Bremerhaven/training_data_oceanids_Bremerhaven-sf-addpreds.csv.gz",
        "Malaga": "/home/ubuntu/data/ML/training-data/OCEANIDS/Malaga/training_data_oceanids_Malaga-sf-addpreds.csv.gz",
        "Malaga-Puerto": "/home/ubuntu/data/ML/training-data/OCEANIDS/Malaga-Puerto/training_data_oceanids_Malaga-Puerto-sf-addpreds.csv.gz",
        "PraiaDaVittoria": "/home/ubuntu/data/ML/training-data/OCEANIDS/PraiaDaVittoria/training_data_oceanids_PraiaDaVittoria-sf-addpreds.csv.gz",
        "PontaDelgada": "/home/ubuntu/data/ML/training-data/OCEANIDS/PontaDelgada/training_data_oceanids_PontaDelgada-sf-addpreds.csv.gz",
        "Plaisance": "/home/ubuntu/data/ML/training-data/OCEANIDS/Plaisance/training_data_oceanids_Plaisance-sf-addpreds.csv.gz",
        "Saint-Guenole": "/home/ubuntu/data/ML/training-data/OCEANIDS/Saint-Guenole/training_data_oceanids_Saint-Guenole-sf-addpreds.csv.gz",
        "Raahe": "/home/ubuntu/data/ML/training-data/OCEANIDS/Raahe/training_data_oceanids_Raahe-sf-addpreds.csv.gz", 
        "Rauma": "/home/ubuntu/data/ML/training-data/OCEANIDS/Rauma/training_data_oceanids_Rauma-sf-addpreds.csv.gz",
        "Vuosaari": "/home/ubuntu/data/ML/training-data/OCEANIDS/Vuosaari/training_data_oceanids_Vuosaari-sf-addpreds.csv.gz"
    }
    try:
        return pd.read_csv(location_files[location], parse_dates=['utctime'])
    except FileNotFoundError:
        return None

train_data = load_training_data(location)
has_observations = train_data is not None

# NEW: Convert training grid-point temperatures from Kelvin to Celsius for temperature forecasts
if has_observations and forecast_type in ["Max Temperature", "Min Temperature"]:
    training_prefix = "mx2t-" if forecast_type == "Max Temperature" else "mn2t-"
    for i in range(1, 5):
        col_name = f"{training_prefix}{i}"
        if col_name in train_data.columns:
            train_data[col_name] = train_data[col_name] - 273.15

# NEW: Convert training grid-point total precipitation from meters to millimeters for Total Precipitation forecast
if has_observations and forecast_type == "Total Precipitation":
    training_prefix = "tp-"
    for i in range(1, 5):
        col_name = f"{training_prefix}{i}"
        if col_name in train_data.columns:
            train_data[col_name] = train_data[col_name] * 1000

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

# Add historical observations distribution if available
if has_observations:
    if train_data[forecast_prefix].dropna().empty:
        st.write("Training data forecast column has only missing values. Distribution cannot be plotted.")
    else:
        sns.kdeplot(data=train_data, x=forecast_prefix, label='Historical Observations', 
                    color='red', linestyle='--', linewidth=2, ax=ax)
        
    # NEW: Plot training grid-point distributions from the training_data file.
    if forecast_type == "Wind Gust Max":
        training_prefix = "fg10-"
    elif forecast_type == "Max Temperature":
        training_prefix = "mx2t-"
    elif forecast_type == "Min Temperature":
        training_prefix = "mn2t-"
    elif forecast_type == "Total Precipitation":
        training_prefix = "tp-"
    training_cols = [f"{training_prefix}{i}" for i in range(1,5)]
    for col in training_cols:
        if col in train_data.columns:
            sns.kdeplot(data=train_data, x=col, label=f'Training {col}', linestyle=':', linewidth=1.5, ax=ax)

ax.set_xlabel(f"{forecast_type} Forecast")
ax.set_ylabel('Density')
if forecast_type == "Wind Gust Max":
    ax.set_title('Distribution of Ensemble Members vs Historical Observations' if has_observations else 'Distribution of Ensemble Members')
    ax.set_xlim(0, 30)
else:
    ax.set_title(f'Distribution of Ensemble {forecast_type} Forecasts' + (' vs Historical Observations' if has_observations else ''))
ax.legend()

plt.tight_layout()
st.pyplot(fig)

# Add interactive data table
st.subheader("Raw Data")
if st.checkbox("Show raw data"):
    st.write(data)

# Add ensemble member visualization
st.subheader("Individual Ensemble Members")
fig, ax = plt.subplots(figsize=(15, 8))

# Plot each ensemble member
for column in forecast_columns:
    ax.plot(data['utctime'], data[column], alpha=0.3, linewidth=0.5)

# Add mean line with different color and thickness
data['ensemble_mean'] = data[forecast_columns].mean(axis=1)
ax.plot(data['utctime'], data['ensemble_mean'], 'r-', 
        linewidth=2, label='Ensemble Mean')

ax.set_xlabel('Date')
ax.set_ylabel(f"{forecast_type} Forecast")
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

# NEW: In the Training Data Analysis section, update metrics by loading the provided CSV file example.
if has_observations:
    st.subheader("Training Data Analysis")
    # Updated to load metrics from CSV instead of computing them from training data.
    metrics_file = f"/home/ubuntu/data/ML/results/OCEANIDS/metrics/{location}-metrics.csv"
    try:
        metrics_df = pd.read_csv(metrics_file)
        st.table(metrics_df)
    except FileNotFoundError:
        st.info("Metrics CSV file not found for this location.")
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(f"/home/ubuntu/data/ML/results/OCEANIDS/{location}/fscore_{location}_{forecast_prefix}_xgb_era5_oceanids-QE.png")
    with col2:
        st.image(f"/home/ubuntu/data/ML/results/OCEANIDS/{location}/shap_{location}_{forecast_prefix}_xgb_era5_oceanids-QE.png")
    with col3:
        st.image(f"/home/ubuntu/data/ML/results/OCEANIDS/{location}/beeswarm_{location}_{forecast_prefix}_xgb_era5_oceanids-QE.png")
else:
    st.info("Training data analysis not available for this location")
    
# Footer
st.markdown("---")
st.markdown("Wind Gust Analysis Tool - Built with Streamlit ❤️")