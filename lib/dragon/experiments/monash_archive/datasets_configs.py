from dragon.experiments.monash_archive.meta_model import FeedCellModel

nn5_daily_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "nn5_daily_dataset_without_missing_values.tsf",
    "DatasetName": "nn5_daily",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 9,
    "NumEpochs": 10,
    "Device": "cpu",
    "ExternalForecastHorizon": None,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

tourism_yearly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "tourism_yearly_dataset.tsf",
    "DatasetName": "tourism_yearly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 2,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": None,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

tourism_quarterly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "tourism_quarterly_dataset.tsf",
    "DatasetName": "tourism_quarterly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 5,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": None,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

tourism_monthly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "tourism_monthly_dataset.tsf",
    "DatasetName": "tourism_monthly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 15,
    "NumEpochs": 1,
    "Device": "cpu",
    "ExternalForecastHorizon": None,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

m1_yearly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m1_yearly_dataset.tsf",
    "DatasetName": "m1_yearly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 2,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m1_quarterly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m1_quarterly_dataset.tsf",
    "DatasetName": "m1_quarterly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 5,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m1_monthly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m1_monthly_dataset.tsf",
    "DatasetName": "m1_monthly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 15,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m3_yearly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m3_yearly_dataset.tsf",
    "DatasetName": "m3_yearly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 2,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m3_quarterly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m3_quarterly_dataset.tsf",
    "DatasetName": "m3_quarterly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 15,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m3_monthly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m3_monthly_dataset.tsf",
    "DatasetName": "m3_monthly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 15,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m3_other_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m3_other_dataset.tsf",
    "DatasetName": "m3_other",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 2,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m4_quarterly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m4_quarterly_dataset.tsf",
    "DatasetName": "m4_quarterly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 5,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m4_monthly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m4_monthly_dataset.tsf",
    "DatasetName": "m4_monthly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 15,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m4_weekly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m4_weekly_dataset.tsf",
    "DatasetName": "m4_weekly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 65,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m4_daily_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m4_daily_dataset.tsf",
    "DatasetName": "m4_daily",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 9,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

m4_hourly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "m4_hourly_dataset.tsf",
    "DatasetName": "m4_hourly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 210,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': None,
    "IntegerConversion": False,
    "Save": True
}

car_parts_config = {
    "PathName":  "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "car_parts_dataset_without_missing_values.tsf",
    "DatasetName": "car_parts",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 15,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': 12,
    "IntegerConversion": True,
    "Save": True
}

hospital_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "hospital_dataset.tsf",
    "DatasetName": "hospital",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 15,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': 12,
    "IntegerConversion": True,
    "Save": True
}

fred_md_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "fred_md_dataset.tsf",
    "DatasetName": "fred_md",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 15,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': 12,
    "IntegerConversion": False,
    "Save": True
}

nn5_weekly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "nn5_weekly_dataset.tsf",
    "DatasetName": "nn5_weekly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 65,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': 8,
    "IntegerConversion": False,
    "Save": True
}

traffic_weekly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "traffic_weekly_dataset.tsf",
    "DatasetName": "traffic_weekly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 65,
    "NumEpochs": 100,
    "Device": "cpu",
    "Model": FeedCellModel,
    'ExternalForecastHorizon': 8,
    "IntegerConversion": False,
    "Save": True
}

electricity_weekly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "electricity_weekly_dataset.tsf",
    "DatasetName": "electricity_weekly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 65,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 8,
    "Model": FeedCellModel,
    "IntegerConversion": True,
    "Save": True
}

solar_weekly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "solar_weekly_dataset.tsf",
    "DatasetName": "solar_weekly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 6,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 5,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

kaggle_web_traffic_weekly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "kaggle_web_traffic_weekly_dataset.tsf",
    "DatasetName": "kaggle_web_traffic_weekly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 10,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 8,
    "Model": FeedCellModel,
    "IntegerConversion": True,
    "Save": True
}

dominick_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "dominick_dataset.tsf",
    "DatasetName": "dominick",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 10,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 8,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

us_births_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "us_births_dataset.tsf",
    "DatasetName": "us_births",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 9,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 30,
    "Model": FeedCellModel,
    "IntegerConversion": True,
    "Save": True
}

saugeen_river_flow_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "saugeenday_dataset.tsf",
    "DatasetName": "saugeen_river_flow",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 9,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 30,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

sunspot_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "sunspot_dataset_without_missing_values.tsf",
    "DatasetName": "sunspot",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 9,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 30,
    "Model": FeedCellModel,
    "IntegerConversion": True,
    "Save": True
}

covid_deaths_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "sunspot_dataset_without_missing_values.tsf",
    "DatasetName": "covid_deaths",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 9,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 30,
    "Model": FeedCellModel,
    "IntegerConversion": True,
    "Save": True
}

weather_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "weather_dataset.tsf",
    "DatasetName": "weather",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 9,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 30,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

traffic_hourly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "traffic_hourly_dataset.tsf",
    "DatasetName": "traffic_hourly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 30,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 168,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

electricity_hourly_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "electricity_hourly_dataset.tsf",
    "DatasetName": "electricity_hourly",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 30,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 168,
    "Model": FeedCellModel,
    "IntegerConversion": True,
    "Save": True
}

solar_10_minutes_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "solar_10_minutes_dataset.tsf",
    "DatasetName": "solar_10_minutes",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 50,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 1008,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

kdd_cup_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "kdd_cup_2018_dataset_without_missing_values.tsf",
    "DatasetName": "kdd_cup",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 210,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 168,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

melbourne_pedestrian_counts_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "pedestrian_counts_dataset.tsf",
    "DatasetName": "melbourne_pedestrian_counts",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 210,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 24,
    "Model": FeedCellModel,
    "IntegerConversion": True,
    "Save": True
}

bitcoin_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "bitcoin_dataset_without_missing_values.tsf",
    "DatasetName": "bitcoin",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 9,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 30,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

vehicle_trips_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "vehicle_trips_dataset_without_missing_values.tsf",
    "DatasetName": "vehicle_trips",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 9,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 30,
    "Model": FeedCellModel,
    "IntegerConversion": True,
    "Save": True
}

aus_elecdemand_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "australian_electricity_demand_dataset.tsf",
    "DatasetName": "aus_elecdemand",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 420,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 336,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

rideshare_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "rideshare_dataset_without_missing_values.tsf",
    "DatasetName": "rideshare",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 210,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 168,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}

temperature_rain_config = {
    "PathName": "lib/dragon/experiments/monash_archive/raw_data/",
    "FileName": "temperature_rain_dataset_without_missing_values.tsf",
    "DatasetName": "temperature_rain",
    "Target": "series_value",
    "TimeCol": "start_timestamp",
    "Lag": 9,
    "NumEpochs": 100,
    "Device": "cpu",
    "ExternalForecastHorizon": 30,
    "Model": FeedCellModel,
    "IntegerConversion": False,
    "Save": True
}


SEASONALITY_MAP = {
    "minutely": [1440, 10080, 525960],
    "10_minutes": [144, 1008, 52596],
    "half_hourly": [48, 336, 17532],
    "hourly": [24, 168, 8766],
    "daily": 7,
    "weekly": 365.25 / 7,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
}

# Frequencies used by GluonTS framework
FREQUENCY_MAP = {
    "minutely": "1min",
    "10_minutes": "10min",
    "half_hourly": "30min",
    "hourly": "1H",
    "daily": "1D",
    "weekly": "1W",
    "monthly": "1M",
    "quarterly": "1Q",
    "yearly": "1Y"
}

dataset_configs = {
    'nn5_daily': nn5_daily_config,
    'tourism_yearly': tourism_yearly_config,
    'tourism_quarterly':tourism_quarterly_config,
    'tourism_monthly': tourism_monthly_config,
    'm1_yearly': m1_yearly_config,
    'm1_quarterly': m1_quarterly_config,
    'm1_monthly': m1_monthly_config,
    'm3_yearly': m3_yearly_config,
    'm3_quarterly': m3_quarterly_config,
    'm3_monthly': m3_monthly_config,
    'm3_other': m3_other_config,
    'm4_quarterly': m4_quarterly_config,
    'm4_monthly': m4_monthly_config,
    'm4_weekly': m4_weekly_config,
    'm4_daily': m4_daily_config,
    'm4_hourly': m4_hourly_config,
    'car_parts': car_parts_config,
    'hospital': hospital_config,
    'fred_md': fred_md_config,
    'nn5_weekly': nn5_weekly_config,
    'traffic_weekly': traffic_weekly_config,
    'electricity_weekly': electricity_weekly_config,
    'solar_weekly': solar_weekly_config,
    'kaggle_web_traffic_weekly': kaggle_web_traffic_weekly_config,
    'dominick': dominick_config,
    'us_births': us_births_config,
    'saugeen_river_flow': saugeen_river_flow_config,
    'sunspot': sunspot_config,
    'covid_deaths': covid_deaths_config,
    'weather': weather_config,
    'traffic_hourly': traffic_hourly_config,
    'electricity_hourly': electricity_hourly_config,
    'solar_10_minutes': solar_10_minutes_config,
    'kdd_cup': kdd_cup_config,
    'melbourne_pedestrian_counts': melbourne_pedestrian_counts_config,
    'bitcoin': bitcoin_config,
    'vehicle_trips': vehicle_trips_config,
    'aus_elecdemand': aus_elecdemand_config,
    'rideshare': rideshare_config,
    'temperature_rain': temperature_rain_config
}