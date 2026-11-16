"""V7 objectives in the existing W&B layout."""
from v7._core.model_v5.monitor import Monitor as BaseMonitor, epoch_metrics as base_metrics


class Monitor(BaseMonitor):
    def __init__(self, args, config, saved=None):
        super().__init__(args, config, saved)
        if self.run is not None:
            self.run.tags = ('v7', 'anatomy-supervision', 'anatomy-oversampling')


def epoch_metrics(record, weights):
    result = base_metrics(record, weights)
    for name in ('anatomy', 'vista_nodule'):
        result[f'01_components/{name}'] = record[name]
    return result
