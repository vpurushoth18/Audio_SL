"""Shared loader mixin for tasks whose data is not on the Hub yet.

MTEB's `load_data` goes straight to `datasets.load_dataset` with the metadata
path and revision, which cannot be satisfied before a dataset is published.
Setting SIMTEB_AUDIO_LOCAL makes a task read the prepared copy from disk
instead, so the benchmark is runnable while the data is still local.
"""

import os
from typing import Any


class LocalDatasetMixin:
    local_dir_name: str

    def load_data(self, **kwargs: Any) -> None:
        root = os.environ.get("SIMTEB_AUDIO_LOCAL")
        if not root:
            super().load_data(**kwargs)
            return

        from datasets import load_from_disk

        self.dataset = load_from_disk(os.path.join(root, self.local_dir_name))
        self.dataset_transform()
        self.data_loaded = True
