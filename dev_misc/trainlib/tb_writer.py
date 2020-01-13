"""This contains `MetricWriter`, a subclass of `SummaryWriter` that integrates `Metric` and `Metrics` instances.
"""

from typing import Union

from torch.utils.tensorboard import SummaryWriter

from dev_misc.trainlib import Metric, Metrics


class MetricWriter(SummaryWriter):

    def add_metrics(self, metrics: Union[Metric, Metrics], global_step: int):
        if isinstance(metrics, Metric):
            metric_iterator = iter([metrics])
        else:
            metric_iterator = iter(metrics)

        for metric in metric_iterator:
            value = metric.mean if metric.report_mean else metric.value
            # TODO(j_luo) Hierarchical metric naming system can better organize tensorboard scalars.
            self.add_scalar(metric.name, value, global_step=global_step)
