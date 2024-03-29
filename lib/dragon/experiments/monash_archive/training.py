from datetime import timedelta
import logging
import warnings

import gluonts

warnings.filterwarnings("ignore")

import csv
import os
import subprocess
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from gluonts.evaluation import make_evaluation_predictions
from dragon.utils.tools import logger
from dragon.experiments.monash_archive.meta_model import FeedCellEstimator, FeedCellLightningModule


class GluontsNet:
    def __init__(self, train_ds, test_ds, config):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.config = config

    def get_nn_forecast(self, args):
        path_name = self.config["SaveDir"] + str(args[-1])
        final_forecasts = []
        forecast_horizon = self.config['ForecastHorizon']

        pl.seed_everything(args[-1])

        gts_logger = logging.getLogger(gluonts.__name__)
        gts_logger.setLevel(logging.ERROR)
        params = {
            "batch_size": 64,
            "device": self.config['Device']
        }
        estimator = FeedCellEstimator(
            model=self.config['Model'],
            lightning_module=FeedCellLightningModule,
            freq=self.config['Freq'],
            prediction_length=forecast_horizon,
            context_length=self.config["Lag"],
            args=args,
            trainer_kwargs={
                "max_epochs": self.config['NumEpochs'],
                "enable_progress_bar": False,
                "devices": "auto", 
                "accelerator": "auto"
            },
            **params
        )

        try:
            predictor = estimator.train(training_data=self.train_ds, enable_progress_bar=False)
            forecast_it, ts_it = make_evaluation_predictions(dataset=self.test_ds, predictor=predictor, num_samples=100)

            # Time series predictions
            forecasts = list(forecast_it)

            # Get median (0.5 quantile) of the 100 sample forecasts as final point forecasts
            for f in forecasts:
                final_forecasts.append(f.mean)

            if self.config["IntegerConversion"]:
                final_forecasts = np.round(final_forecasts)

            if not os.path.exists(path_name + "/results/fixed_horizon_forecasts/"):
                os.makedirs(path_name + "/results/fixed_horizon_forecasts/")

            # write the forecasting results to a file
            file_name = self.config["DatasetName"] + "_lag_" + str(self.config["Lag"])
            forecast_file_path = path_name + "/results/fixed_horizon_forecasts/" + file_name + ".txt"

            with open(forecast_file_path, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(final_forecasts)
            temp_dataset_path = path_name + "/results/fixed_horizon_forecasts/" + self.config["DatasetName"] + \
                                "_dataset.txt"
            temp_results_path = path_name + "/results/fixed_horizon_forecasts/" + self.config["DatasetName"] + \
                                "_results.txt"

            with open(temp_dataset_path, "w") as output_dataset:
                writer = csv.writer(output_dataset, lineterminator='\n')
                writer.writerows(self.config['TrainSeries'])

            with open(temp_results_path, "w") as output_results:
                writer = csv.writer(output_results, lineterminator='\n')
                writer.writerows(self.config['TestSeries'])

            if not os.path.exists(path_name + f"/results/fixed_horizon_errors/"):
                os.makedirs(path_name + f"/results/fixed_horizon_errors/")
            try:
                smape = subprocess.check_output(
                    ["Rscript", "--vanilla", "lib/dragon/experiments/monash_archive/tsforecastinggit/error_calc_helper.R",
                     self.config['PathName'],
                     forecast_file_path, temp_results_path, temp_dataset_path, str(self.config['Seasonality']),
                     file_name])
            except subprocess.CalledProcessError as e:
                raise e
            smape = float(str(smape).split(' ')[-1].split("\\")[0])
            logger.info(f"Mean MASE: {smape}")
            # Remove intermediate files
            os.system("rm " + temp_dataset_path)
            os.system("rm " + temp_results_path)
            os.system("rm -r " + path_name)
        except ValueError as e:
            logger.info(f"Mean MASE: NAN")
            try:
                os.system("rm -r " + path_name)
            except Exception:
                pass
            smape = np.inf
        return smape
