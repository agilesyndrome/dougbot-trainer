from datasets import load_dataset, concatenate_datasets
from torch.onnx.symbolic_opset11 import chunk

from lib import StringUtils


class HuggingFaceDataset:
    def __init__(self, config=None):
        self.ds = None

        if config:
            self.token = config.HF_TOKEN
            self.dataset = config.HF_DATASET

        self.load()

    def load(self):
        if self.dataset is None:
            raise Exception('No dataset specified')

        ds = load_dataset(self.dataset, token=self.token)

        if 'train' in ds:
            self.ds = ds['train']
            for key in ds.keys():
                if key != 'train':
                    setattr(self, f'ds_{key}', ds[key])

        else:
            self.ds = ds

    def concat(self, additions):

        if isinstance(additions, list):
            for addition in additions:
                self.ds = concatenate_datasets(self.ds, addition)
        else:
            self.ds = concatenate_datasets(self.ds, additions)
        return self


#
#from datasets import load_dataset_builder
#ds_builder = load_dataset_builder("rotten_tomatoes")

# Inspect dataset description
#ds_builder.info.description
#Movie Review Dataset. This is a dataset of containing 5,331 positive and 5,331 negative processed sentences from Rotten Tomatoes movie reviews. This data was first used in Bo Pang and Lillian Lee, ``Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.'', Proceedings of the ACL, 2005.

# Inspect dataset features
#ds_builder.info.features
#{'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
# 'text': Value(dtype='string', id=None)}
