import pandas as pd

def is_360_day_calendar(data, date_col):
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data['month'] = data[date_col].dt.month
    data['day'] = data[date_col].dt.day
    feb_30_exists = ((data['month'] == 2) & (data['day'] == 30)).any()
    months_with_31_days = [1, 3, 5, 7, 8, 10, 12]
    missing_31st_days = any(
        not ((data['month'] == month) & (data['day'] == 31)).any()
        for month in months_with_31_days
    )
    data.drop(columns=['month', 'day'], inplace=True)
    return feb_30_exists or missing_31st_days

def revert_360_day_calendar(data, date_col):
    data[date_col] = pd.to_datetime(data[date_col])
    months_with_31_days = [1, 3, 5, 7, 8, 10, 12]
    adjusted_data = []
    for year, group in data.groupby(data[date_col].dt.year):
        year = int(year)
        jan_31 = group[(group[date_col].dt.month == 2) & (group[date_col].dt.day == 1)].copy()
        if not jan_31.empty:
            jan_31[date_col] = pd.to_datetime(f"{year}-01-31")
            adjusted_data.append(jan_31)
        is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        if is_leap:
            feb_29 = group[(group[date_col].dt.month == 2) & (group[date_col].dt.day == 28)].copy()
            if not feb_29.empty:
                feb_29[date_col] = pd.to_datetime(f"{year}-02-29")
                adjusted_data.append(feb_29)
        mar_1 = group[
            (group[date_col].dt.month == 2) &
            (group[date_col].dt.day == (29 if is_leap else 28))
        ].copy()
        if not mar_1.empty:
            mar_1[date_col] = pd.to_datetime(f"{year}-03-01")
            adjusted_data.append(mar_1)
        for month in months_with_31_days:
            day_30 = group[(group[date_col].dt.month == month) & (group[date_col].dt.day == 30)].copy()
            if not day_30.empty:
                day_30[date_col] = pd.to_datetime(f"{year}-{month:02d}-31")
                adjusted_data.append(day_30)
        adjusted_data.append(group)
    adjusted_df = pd.concat(adjusted_data).sort_values(by=date_col).reset_index(drop=True)
    return adjusted_df
