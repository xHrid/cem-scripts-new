import argparse
import os
import re
import pandas as pd
import sys
import numpy as np
from scipy.stats import spearmanr, pearsonr, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pytz
from astral import LocationInfo
from astral.sun import sunrise, sunset
import calendar
import warnings
warnings.filterwarnings('ignore')


def get_expected_count(rule):
    """Returns expected segments in 24h based on sampling rule."""
    if rule == '30R30W': return 1440
    if rule == '5R5W': return 144
    if rule == '2R4W': return 240
    return 1440


def zscore_on_active_days(df_plot, value_col, active_days_set, date_col="Date_Only", out_col="z"):
    """Z-score computed only on active days. Discarded days forced to NaN."""
    valid_values = df_plot.loc[df_plot[date_col].isin(active_days_set), value_col]
    mu = valid_values.mean()
    sigma = valid_values.std()

    if sigma and sigma > 0:
        df_plot[out_col] = (df_plot[value_col] - mu) / sigma
    else:
        df_plot[out_col] = 0

    df_plot.loc[~df_plot[date_col].isin(active_days_set), out_col] = np.nan
    return df_plot


def get_active_months(df_data):
    """Identifies months where average activity > 20% of peak."""
    df_data = df_data.copy()
    df_data['Date_Only'] = pd.to_datetime(df_data['Date_Only'])
    df_data['month_idx'] = df_data['Date_Only'].dt.month

    monthly_avg = df_data.groupby('month_idx')['call_count'].mean()

    if len(monthly_avg) == 0 or monthly_avg.max() == 0:
        return []

    threshold = monthly_avg.max() * 0.20
    active_indices = monthly_avg[monthly_avg > threshold].index

    return [calendar.month_name[i] for i in sorted(active_indices)]


def clean_filename(name: str):
    """Clean filename for safe file saving."""
    name = name.strip().lower()
    name = name.replace(" ", "_").replace("-", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


# ============================================================================
# PART 1: DATA LOADING & PREPROCESSING (Overhauled for Web App)
# ============================================================================

def load_and_preprocess(detection_csv, weather_csv, output_dir):
    """Load detection and weather data, extract temporal info safely."""
    print("\n=== PART 1: Data Loading & Preprocessing ===")

    # Load detection data
    results_df = pd.read_csv(detection_csv)
    
    # Fallback cleanup just in case old filenames slipped through
    if 'filename' in results_df.columns:
        results_df['filename'] = results_df['filename'].str.replace("STOP", "SPOT", regex=False)

        # Extract date from filename if not explicitly provided by the new BirdNET script
        if 'Date' not in results_df.columns:
            date_info = results_df['filename'].str.extract(r'_(\d{8})_')
            results_df['Date'] = pd.to_datetime(date_info[0], format='%Y%m%d', errors='coerce')
    else:
        # Construct a Date column from year/month/day if filename is missing
        results_df['Date'] = pd.to_datetime(results_df[['year', 'month', 'day']])
        
    results_df['Date_Only'] = results_df['Date'].dt.date

    # Rely on the new 'spot' column from BirdNET, fallback to regex if missing
    if 'spot' in results_df.columns:
        results_df['Spot'] = results_df['spot'].str.upper()
    elif 'filename' in results_df.columns:
        results_df['Spot'] = results_df['filename'].str.extract(r'(SPOT\d+)', expand=False, flags=re.IGNORECASE).str.upper()
    else:
        results_df['Spot'] = "UNKNOWN_SPOT"

    results_df.dropna(subset=['Spot', 'Date'], inplace=True)

    # --- Graceful Weather Handling ---
    sanjay_van_data = None
    if weather_csv and os.path.exists(weather_csv):
        sanjay_van_data = pd.read_csv(weather_csv)
        # Dynamically find the date column instead of hallucinating dates
        possible_date_cols = ['Date', 'date', 'Date_Only', 'time', 'Time']
        found_date_col = next((c for c in possible_date_cols if c in sanjay_van_data.columns), None)
        
        if found_date_col:
            sanjay_van_data['Date_Only'] = pd.to_datetime(sanjay_van_data[found_date_col], errors='coerce').dt.date
            sanjay_van_data.dropna(subset=['Date_Only'], inplace=True)
        else:
            print(f"WARNING: Weather file {weather_csv} has no recognizable Date column. Skipping weather analysis.")
            sanjay_van_data = None
    else:
        print("WARNING: No valid weather file provided or found. Skipping weather analysis.")

    print(f"Loaded {len(results_df)} detections from {len(results_df['common_name'].unique())} species.")
    print(f"Date range: {results_df['Date_Only'].min()} to {results_df['Date_Only'].max()}")

    return results_df, sanjay_van_data


# ============================================================================
# PART 2: BEHAVIORAL STICKINESS ANALYSIS
# ============================================================================

def analyze_temporal_stickiness(results_df, output_dir):
    """Calculate temporal stickiness (predictability of daily routines)."""
    print("\n=== PART 2: Temporal Stickiness Analysis ===")

    results_df['Date_Only'] = pd.to_datetime(results_df['Date']).dt.date

    ACTIVITY_HOURS = range(5, 20)
    activity_df = results_df[(results_df['confidence'] >= 0.3) & (results_df['hour'].isin(ACTIVITY_HOURS))].copy()

    daily_counts = activity_df.groupby(['common_name', 'Date_Only']).size().reset_index(name='daily_count')
    daily_counts = daily_counts[daily_counts['daily_count'] > 10]

    valid_birds = daily_counts['common_name'].value_counts()[lambda x: x > 5].index
    daily_counts = daily_counts[daily_counts['common_name'].isin(valid_birds)]

    activity_df = activity_df.merge(
        daily_counts[['common_name', 'Date_Only']],
        on=['common_name', 'Date_Only'],
        how='inner'
    )

    species_list = activity_df['common_name'].unique()
    spot_list = activity_df['Spot'].unique()
    date_list = sorted(activity_df['Date'].unique())

    temporal_stickiness = {}

    for species in species_list:
        species_spot_correlations = []
        for spot in spot_list:
            spot_day_correlations = []
            for i in range(len(date_list) - 1):
                day_k = date_list[i]
                day_k_plus_1 = date_list[i+1]

                series_k = activity_df[
                    (activity_df['common_name'] == species) &
                    (activity_df['Spot'] == spot) &
                    (activity_df['Date'] == day_k)
                ]['hour'].value_counts().reindex(ACTIVITY_HOURS, fill_value=0)

                series_k_plus_1 = activity_df[
                    (activity_df['common_name'] == species) &
                    (activity_df['Spot'] == spot) &
                    (activity_df['Date'] == day_k_plus_1)
                ]['hour'].value_counts().reindex(ACTIVITY_HOURS, fill_value=0)

                if series_k.sum() > 0 and series_k_plus_1.sum() > 0:
                    corr, _ = spearmanr(series_k, series_k_plus_1)
                    spot_day_correlations.append(corr)

            if spot_day_correlations:
                species_spot_correlations.append(np.mean(spot_day_correlations))

        if species_spot_correlations:
            temporal_stickiness[species] = np.mean(species_spot_correlations)

    temporal_stickiness_df = pd.DataFrame(
        list(temporal_stickiness.items()),
        columns=['common_name', 'Temporal_Stickiness']
    ).sort_values(by='Temporal_Stickiness', ascending=False)

    output_file = os.path.join(output_dir, 'temporal_stickiness.csv')
    temporal_stickiness_df.to_csv(output_file, index=False)
    print(f"Saved temporal stickiness to {output_file}")

    return temporal_stickiness_df


# ============================================================================
# PART 3: SPATIAL STICKINESS ANALYSIS
# ============================================================================

def analyze_spatial_stickiness(results_df, output_dir):
    """Calculate spatial stickiness (preference for specific spots)."""
    print("\n=== PART 3: Spatial Stickiness Analysis ===")

    results_df['Date_Only'] = pd.to_datetime(results_df['Date']).dt.date

    spatial_df = results_df[results_df['confidence'] >= 0.3].copy()

    spot_list = sorted(spatial_df['Spot'].unique())
    date_list = sorted(spatial_df['Date'].unique())

    if len(spot_list) < 2:
        print("WARNING: Need data from at least 2 spots for spatial analysis. Skipping spatial stickiness.")
        return None

    spatial_stickiness = {}

    for species in spatial_df['common_name'].unique():
        daily_rank_correlations = []

        for i in range(len(date_list) - 1):
            day_k = date_list[i]
            day_k_plus_1 = date_list[i + 1]

            counts_k = spatial_df[
                (spatial_df['common_name'] == species) &
                (spatial_df['Date'] == day_k)
            ].groupby('Spot').size().reindex(spot_list, fill_value=0)

            counts_k_plus_1 = spatial_df[
                (spatial_df['common_name'] == species) &
                (spatial_df['Date'] == day_k_plus_1)
            ].groupby('Spot').size().reindex(spot_list, fill_value=0)

            if counts_k.nunique() > 1 and counts_k_plus_1.nunique() > 1:
                corr, _ = spearmanr(counts_k, counts_k_plus_1)
                if not np.isnan(corr):
                    daily_rank_correlations.append(corr)

        if daily_rank_correlations:
            spatial_stickiness[species] = np.mean(daily_rank_correlations)

    spatial_stickiness_df = pd.DataFrame(
        list(spatial_stickiness.items()),
        columns=['common_name', 'Spatial_Stickiness']
    ).sort_values(by='Spatial_Stickiness', ascending=False)

    output_file = os.path.join(output_dir, 'spatial_stickiness.csv')
    spatial_stickiness_df.to_csv(output_file, index=False)
    print(f"Saved spatial stickiness to {output_file}")

    return spatial_stickiness_df


# ============================================================================
# PART 4: PEAK ACTIVITY ANALYSIS (with Sunrise/Sunset)
# ============================================================================

def analyze_peak_activity(results_df, output_dir, lat, lon):
    """Calculate peak activity hours and correlate with sunrise/sunset."""
    print("\n=== PART 4: Peak Activity Analysis ===")

    results_df['Date_Only'] = pd.to_datetime(results_df['Date']).dt.date

    daily_counts = results_df.groupby(['common_name', 'Date_Only']).size().reset_index(name='daily_count')
    daily_counts = daily_counts[daily_counts['daily_count'] > 10]

    valid_birds = daily_counts['common_name'].value_counts()[lambda x: x > 5].index
    daily_counts = daily_counts[daily_counts['common_name'].isin(valid_birds)]

    filtered_results = results_df.merge(
        daily_counts[['common_name', 'Date_Only']],
        on=['common_name', 'Date_Only'],
        how='inner'
    )

    peak_activity = filtered_results.groupby(['common_name', 'Date_Only'])['hour'].agg(
        lambda x: x.value_counts().idxmax()
    ).reset_index()
    peak_activity.rename(columns={'hour': 'peak_hour'}, inplace=True)

    city = LocationInfo("Sanjay Van", "India", "Asia/Kolkata", lat, lon)
    timezone = pytz.timezone("Asia/Kolkata")

    unique_dates = peak_activity['Date_Only'].unique()
    sunrise_data = []
    sunset_data = []

    for d in unique_dates:
        try:
            sr_dt = sunrise(city.observer, date=d, tzinfo=timezone)
            ss_dt = sunset(city.observer, date=d, tzinfo=timezone)

            sr_decimal = sr_dt.hour + sr_dt.minute/60 + sr_dt.second/3600
            ss_decimal = ss_dt.hour + ss_dt.minute/60 + ss_dt.second/3600

            sunrise_data.append({'Date_Only': d, 'sunrise_decimal': sr_decimal})
            sunset_data.append({'Date_Only': d, 'sunset_decimal': ss_decimal})
        except:
            continue

    df_sunrise = pd.DataFrame(sunrise_data)
    df_sunset = pd.DataFrame(sunset_data)

    if not df_sunrise.empty and not df_sunset.empty:
        merged_sunrise = pd.merge(peak_activity, df_sunrise, on='Date_Only', how='inner')
        merged_sunset = pd.merge(peak_activity, df_sunset, on='Date_Only', how='inner')

        corr_results_sunrise = []
        corr_results_sunset = []

        for bird in merged_sunrise['common_name'].unique():
            subset_sr = merged_sunrise[merged_sunrise['common_name'] == bird]
            subset_ss = merged_sunset[merged_sunset['common_name'] == bird]

            if len(subset_sr) > 2:
                pr, pp = pearsonr(subset_sr['peak_hour'], subset_sr['sunrise_decimal'])
                corr_results_sunrise.append({
                    'Bird': bird,
                    'Pearson_r': round(pr, 3),
                    'P_Value': round(pp, 4),
                    'Samples': len(subset_sr)
                })

            if len(subset_ss) > 2:
                pr, pp = pearsonr(subset_ss['peak_hour'], subset_ss['sunset_decimal'])
                corr_results_sunset.append({
                    'Bird': bird,
                    'Pearson_r': round(pr, 3),
                    'P_Value': round(pp, 4),
                    'Samples': len(subset_ss)
                })

        corr_sr_df = pd.DataFrame(corr_results_sunrise).sort_values('Pearson_r', ascending=False) if corr_results_sunrise else pd.DataFrame()
        corr_ss_df = pd.DataFrame(corr_results_sunset).sort_values('Pearson_r', ascending=False) if corr_results_sunset else pd.DataFrame()

        if not corr_sr_df.empty: corr_sr_df.to_csv(os.path.join(output_dir, 'peak_activity_vs_sunrise.csv'), index=False)
        if not corr_ss_df.empty: corr_ss_df.to_csv(os.path.join(output_dir, 'peak_activity_vs_sunset.csv'), index=False)

        print("Saved peak activity correlations")
        return peak_activity, corr_sr_df, corr_ss_df

    return peak_activity, pd.DataFrame(), pd.DataFrame()


# ============================================================================
# PART 5: MIGRATORY STATUS CLASSIFICATION
# ============================================================================

def classify_migratory_status(results_df, output_dir):
    """Classify birds as migratory or resident using SCI, Kurtosis, and PMR."""
    print("\n=== PART 5: Migratory Status Classification ===")

    results_df['Date_Only'] = pd.to_datetime(results_df['Date']).dt.date

    # Safely handle sampling_rule. If missing, assume a default for classification math
    if 'sampling_rule' not in results_df.columns:
        results_df['sampling_rule'] = '2R4W'

    spot_activity = results_df.groupby(['Date_Only', 'Spot', 'sampling_rule']).size().reset_index(name='segment_count')
    spot_activity['expected'] = spot_activity['sampling_rule'].apply(get_expected_count)
    spot_activity['is_valid_day'] = spot_activity['segment_count'] >= (spot_activity['expected'] * (16/24))

    valid_spots_df = spot_activity[spot_activity['is_valid_day'] == True].copy()
    daily_spot_density = valid_spots_df.groupby('Date_Only').size().reset_index(name='active_spot_count')

    filtered_results = results_df.merge(
        valid_spots_df[['Date_Only', 'Spot']],
        on=['Date_Only', 'Spot'],
        how='inner'
    )

    global_dates_with_data = set(daily_spot_density['Date_Only'].unique())
    all_dates = pd.date_range(
        start=results_df['Date_Only'].min(),
        end=results_df['Date_Only'].max()
    ).date

    unique_birds = filtered_results['common_name'].unique()
    all_classifications = []

    for bird in unique_birds:
        bird_counts = filtered_results[filtered_results['common_name'] == bird].groupby('Date_Only').size().reset_index(name='raw_calls')
        bird_norm = bird_counts.merge(daily_spot_density, on='Date_Only', how='left')
        bird_norm['call_count'] = bird_norm['raw_calls'] / bird_norm['active_spot_count']

        df_plot = pd.DataFrame({'Date_Only': all_dates})
        df_plot = df_plot.merge(bird_norm[['Date_Only', 'call_count']], on='Date_Only', how='left')
        df_plot.loc[df_plot['Date_Only'].isin(global_dates_with_data), 'call_count'] = df_plot['call_count'].fillna(0)

        ts_data = df_plot.loc[df_plot['Date_Only'].isin(global_dates_with_data), 'call_count'].values

        if np.sum(ts_data) == 0:
            continue

        window_size = min(60, len(ts_data))
        if window_size == 0: continue
        
        padded_data = np.concatenate((ts_data, ts_data[:window_size-1]))
        window_sums = np.convolve(padded_data, np.ones(window_size), mode='valid')
        sci_score = np.max(window_sums) / np.sum(ts_data)

        df_reg = df_plot.dropna(subset=['call_count']).copy()
        if len(df_reg) >= 10:
            df_reg['t'] = pd.to_datetime(df_reg['Date_Only']).map(pd.Timestamp.toordinal)
            X = df_reg[['t']]
            y = df_reg['call_count']
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            kurt_score = kurtosis(residuals, fisher=True)
        else:
            kurt_score = np.nan

        peak_val = np.max(ts_data)
        median_val = np.median(ts_data)
        pmr_score = peak_val / (median_val + 1e-4)

        status = "Resident"
        if sci_score > 0.9 or kurt_score > 15 or pmr_score > 50:
            status = "Migratory"
        elif sci_score > 0.7 or (8 < kurt_score <= 15) or (20 < pmr_score <= 50):
            status = "Semi-Resident"

        all_classifications.append({
            'Species': bird,
            'Status': status,
            'SCI_Score': round(sci_score, 3),
            'Kurtosis': round(kurt_score, 3) if not np.isnan(kurt_score) else np.nan,
            'PMR_Score': round(pmr_score, 3)
        })

    if all_classifications:
        classification_df = pd.DataFrame(all_classifications).sort_values('SCI_Score', ascending=False)
        classification_df.to_csv(os.path.join(output_dir, 'migratory_classification.csv'), index=False)
        print(f"Classified {len(classification_df)} species")
        return classification_df
    else:
        print("Not enough data to classify migratory status.")
        return pd.DataFrame()


# ============================================================================
# PART 6: TEMPERATURE & WEATHER CORRELATION
# ============================================================================

def analyze_temperature_correlation(results_df, weather_df, output_dir):
    """Correlate peak activity time with temperature."""
    if weather_df is None or 'temp_celsius' not in weather_df.columns:
        print("\nSkipping temperature analysis (no valid weather data found).")
        return None

    print("\n=== PART 6: Temperature Correlation ===")

    results_df['Date_Only'] = pd.to_datetime(results_df['Date']).dt.date

    daily_counts = results_df.groupby(['common_name', 'Date_Only']).size().reset_index(name='daily_count')
    daily_counts = daily_counts[daily_counts['daily_count'] > 10]

    valid_birds = daily_counts['common_name'].value_counts()[lambda x: x > 5].index
    filtered_results = results_df.merge(
        daily_counts[daily_counts['common_name'].isin(valid_birds)][['common_name', 'Date_Only']],
        on=['common_name', 'Date_Only'],
        how='inner'
    )

    peak_activity = filtered_results.groupby(['common_name', 'Date_Only'])['hour'].agg(
        lambda x: x.value_counts().idxmax()
    ).reset_index()
    peak_activity.rename(columns={'hour': 'peak_hour'}, inplace=True)

    merged_df = pd.merge(peak_activity, weather_df[['Date_Only', 'temp_celsius']], on='Date_Only', how='inner')

    temp_corr_results = []

    for bird in merged_df['common_name'].unique():
        subset = merged_df[merged_df['common_name'] == bird].dropna(subset=['peak_hour', 'temp_celsius'])

        if len(subset) >= 8:
            X = subset[['temp_celsius']].values
            y = subset['peak_hour'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            r2 = r2_score(y_test, model.predict(X_test))
            mae = mean_absolute_error(y_test, model.predict(X_test))

            temp_corr_results.append({
                'Species': bird,
                'R2_Score': round(r2, 3),
                'MAE_Hours': round(mae, 2),
                'Temp_Coef': round(model.coef_[0], 4),
                'Samples': len(subset)
            })

    if temp_corr_results:
        temp_corr_df = pd.DataFrame(temp_corr_results).sort_values('R2_Score', ascending=False)
        temp_corr_df.to_csv(os.path.join(output_dir, 'temperature_correlation.csv'), index=False)
        print(f"Analyzed temperature correlation for {len(temp_corr_df)} species")
        return temp_corr_df

    return None


# ============================================================================
# PART 7: SPOT PREFERENCE ANALYSIS
# ============================================================================

def analyze_spot_preference(results_df, output_dir):
    """Identify preferred spots for each species."""
    print("\n=== PART 7: Spot Preference Analysis ===")

    species_spot_counts = results_df.groupby(['common_name', 'Spot']).size().unstack(fill_value=0)
    species_total = results_df.groupby('common_name').size()

    species_spot_ratio = species_spot_counts.div(species_total, axis=0)

    preferred = species_spot_ratio > 0.5
    preferred_pairs = preferred.stack()
    preferred_pairs = preferred_pairs[preferred_pairs]

    preferred_df = preferred_pairs.reset_index()
    preferred_df.columns = ["Species", "Preferred_Spot", "Flag"]

    preferred_df["Ratio"] = preferred_df.apply(
        lambda row: species_spot_ratio.loc[row["Species"], row["Preferred_Spot"]],
        axis=1
    )

    preferred_df = preferred_df[["Species", "Preferred_Spot", "Ratio"]].sort_values('Ratio', ascending=False)

    output_file = os.path.join(output_dir, 'spot_preferences.csv')
    preferred_df.to_csv(output_file, index=False)
    print(f"Identified spot preferences for {len(preferred_df)} species-spot pairs")

    return preferred_df


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(detection_csv, weather_csv, output_dir, lat=28.53, lon=77.18):
    """Main analysis pipeline."""
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(detection_csv):
        print(f"ERROR: Detections CSV not found at {detection_csv}")
        sys.exit(1)

    results_df, weather_df = load_and_preprocess(detection_csv, weather_csv, output_dir)

    if len(results_df) == 0:
        print("ERROR: Detection CSV contains no valid, parsed data. Halting analysis.")
        sys.exit(1)

    # Run analyses
    try:
        analyze_temporal_stickiness(results_df, output_dir)
        analyze_spatial_stickiness(results_df, output_dir)
        analyze_peak_activity(results_df, output_dir, lat, lon)
        classify_migratory_status(results_df, output_dir)
        analyze_temperature_correlation(results_df, weather_df, output_dir)
        analyze_spot_preference(results_df, output_dir)
    except Exception as e:
        print(f"\n⚠️ WARNING: A sub-analysis failed during execution: {e}")

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive acoustic biodiversity analysis pipeline."
    )
    parser.add_argument(
        "--detection-csv",
        required=True,
        help="Path to BirdNET classification CSV file"
    )
    parser.add_argument(
        "--weather-csv",
        default=None,
        help="Path to weather data CSV file (optional)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=28.53,
        help="Latitude for sunrise/sunset calculations"
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=77.18,
        help="Longitude for sunrise/sunset calculations"
    )

    args = parser.parse_args()

    main(
        detection_csv=args.detection_csv,
        weather_csv=args.weather_csv,
        output_dir=args.output_dir,
        lat=args.lat,
        lon=args.lon
    )