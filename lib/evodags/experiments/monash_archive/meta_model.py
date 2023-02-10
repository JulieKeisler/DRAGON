from typing import List, Optional, Iterable, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    Transformation,
    AddObservedValuesIndicator,
    InstanceSampler,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
)
from gluonts.torch.util import (
    IterableDataset,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)

from evodags.search_space.bricks.basics import MLP
from evodags.search_space.cells import CandidateOperation, AdjCell

PREDICTION_INPUT_NAMES = [
    "past_target",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


def mean_abs_scaling(seq, min_scale=1e-5):
    return seq.abs().mean(1).clamp(min_scale, None).unsqueeze(1)


class FeedCellModel(nn.Module):
    def __init__(self, prediction_length: int, context_length: int, args: Any,
                 distr_output=StudentTOutput()):
        super().__init__()
        assert prediction_length > 0
        assert context_length > 0
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.distr_output = distr_output
        self.cell = AdjCell(args[0], (self.context_length,))
        self.output = CandidateOperation("add", MLP(self.cell.output_shape,
                                                    out_channels=self.cell.output_shape * prediction_length),
                                         self.cell.output_shape, activation=args[1])
        self.args_proj = self.distr_output.get_args_proj(self.cell.output_shape)

    def forward(self, context: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        scale = mean_abs_scaling(context)
        scaled_context = context / scale
        try:
            cell_out = self.cell(scaled_context)
            output_out = self.output(cell_out)
            output_out_reshaped = output_out.reshape(-1, self.prediction_length, self.cell.output_shape)
            distr_args = self.args_proj(output_out_reshaped)
        except RuntimeError as e:
            raise e
        return distr_args, torch.zeros_like(scale), scale


class FeedCellLightningModule(pl.LightningModule):
    def __init__(self, model: FeedCellModel,
                 loss: DistributionLoss = NegativeLogLikelihood(),
                 lr: float = 1e-3,
                 weight_decay: float = 1e-8):
        super().__init__()
        self.save_hyperparameters()
        if 'model' not in self.hparams:
            self.hparams['model'] = model
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay

    def _compute_loss(self, batch):
        context = batch["past_target"]
        target = batch["future_target"]
        observed_target = batch["future_observed_values"]

        assert context.shape[-1] == self.model.context_length
        assert target.shape[-1] == self.model.prediction_length

        distr_args, loc, scale = self.model(context)
        distr = self.model.distr_output.distribution(distr_args, loc, scale)

        return (
                       self.loss(distr, target) * observed_target
               ).sum() / torch.maximum(torch.tensor(1.0), observed_target.sum())

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss = self._compute_loss(batch)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=False,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self._compute_loss(batch)
        self.log(
            "val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=False, logger=False,
        )
        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


class FeedCellEstimator(PyTorchLightningEstimator):

    def __init__(self,
                 prediction_length: int,
                 context_length: int,
                 args: List,
                 device,
                 model=FeedCellModel,
                 distr_output: DistributionOutput = StudentTOutput(),
                 loss: DistributionLoss = NegativeLogLikelihood(),
                 batch_size: int = 32,
                 num_batches_per_epoch: int = 50,
                 trainer_kwargs: Optional[Dict[str, Any]] = None,
                 train_sampler: Optional[InstanceSampler] = None,
                 validation_sampler: Optional[InstanceSampler] = None,
                 ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)
        self.context_length = context_length or 10 * prediction_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.loss = loss
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.device = device
        self.meta_model = model

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

        self.args = args

    def create_transformation(self) -> Transformation:
        return SelectFields(
            [
                FieldName.ITEM_ID,
                FieldName.INFO,
                FieldName.START,
                FieldName.TARGET,
            ],
            allow_missing=True,
        ) + AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )

    def create_lightning_module(self) -> pl.LightningModule:
        model = self.meta_model(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            args=self.args,
            distr_output=self.distr_output,
        )
        return FeedCellLightningModule(
            model=model,
            loss=self.loss,
        )

    def _create_instance_splitter(
            self, module: FeedCellLightningModule, mode: str
    ):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.OBSERVED_VALUES,
            ],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
            self,
            data: Dataset,
            module: FeedCellLightningModule,
            shuffle_buffer_length: Optional[int] = None,
            **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            module, "training"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )

        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=self.batch_size,
                    **kwargs,
                )
            ),
            self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
            self,
            data: Dataset,
            module: FeedCellLightningModule,
            **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(module, "validation") + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

    def create_predictor(
            self,
            transformation: Transformation,
            module,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module.model,
            forecast_generator=DistributionForecastGenerator(
                self.distr_output
            ),
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device=self.device
        )
