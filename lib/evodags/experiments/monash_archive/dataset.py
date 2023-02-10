from datetime import datetime

import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

import lib.evodags.experiments.monash_archive.tsforecastinggit.tsf_loader as loader
from lib.evodags.experiments.monash_archive.datasets_configs import FREQUENCY_MAP, SEASONALITY_MAP
from lib.evodags.utils.tools import logger, read_nn


def gluonts_dataset(config):
    logger.info("Started loading " + config["DatasetName"])

    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = loader.convert_tsf_to_dataframe(
        config['PathName'] + config['FileName'], 'NaN', config['Target'])

    train_series_list = []
    test_series_list = []
    train_series_full_list = []
    test_series_full_list = []

    if frequency is not None:
        freq = FREQUENCY_MAP[frequency]
        seasonality = SEASONALITY_MAP[frequency]
    else:
        freq = "1Y"
        seasonality = 1

    if isinstance(seasonality, list):
        seasonality = min(seasonality)  # Use to calculate MASE

    external_forecast_horizon = config['ExternalForecastHorizon']
    # If the forecast horizon is not given within the .tsf file, then it should be provided as a function input
    if forecast_horizon is None:
        if external_forecast_horizon is None:
            raise Exception("Please provide the required forecast horizon")
        else:
            forecast_horizon = external_forecast_horizon

    for index, row in df.iterrows():
        if config['TimeCol'] in df.columns:
            train_start_time = row[config['TimeCol']]
        else:
            train_start_time = datetime.strptime('1900-01-01 00-00-00',
                                                 '%Y-%m-%d %H-%M-%S')  # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False

        series_data = row[config['Target']]

        # Creating training and test series. Test series will be only used during evaluation
        train_series_data = series_data[:len(series_data) - forecast_horizon]
        test_series_data = series_data[(len(series_data) - forecast_horizon): len(series_data)]

        train_series_list.append(train_series_data)
        test_series_list.append(test_series_data)

        # We use full length training series to train the model as we do not tune hyperparameters
        train_series_full_list.append({
            FieldName.TARGET: train_series_data,
            FieldName.START: pd.Timestamp(train_start_time, freq=freq)
        })

        test_series_full_list.append({
            FieldName.TARGET: series_data,
            FieldName.START: pd.Timestamp(train_start_time, freq=freq)
        })

    train_ds = ListDataset(train_series_full_list, freq=freq)
    test_ds = ListDataset(test_series_full_list, freq=freq)
    config['Freq'] = freq
    config['ForecastHorizon'] = forecast_horizon
    config['TrainSeries'] = train_series_list
    config['TestSeries'] = test_series_list
    config['Seasonality'] = seasonality
    return train_ds, test_ds, config


def generate_args_from_string(string):
    model = string[0]
    matrix = read_nn(model[0])
    args = [matrix, model[1], int(model[2])]
    return [args]


def generate_args_and_loss_from_file(filename, sep=";"):
    data = pd.read_table(filename, sep=sep)
    best_model = data[data["loss"] == data["loss"].min()]
    matrix = read_nn(best_model['Cell'].item())
    args = [matrix, best_model['NN Activation'].item(), int(best_model['Seed'].item())]
    loss = best_model['loss'].item()
    return args, loss
