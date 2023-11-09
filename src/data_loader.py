from torch.utils import data
from os.path import isfile, splitext
from typing import Union, List, Tuple
import json
import pickle
import pandas as pd
class Dataset(data.Dataset):
    '''Convenience class for loading dataset'''
    def __init__(self, data: Union[str, List[tuple]]):
        '''
        data: either a filepath or a list of tuples, where each tuple is a
            mention-entity pair
        '''
        if isinstance(data, str) and isfile(data):
            # filepath provided, load file
            if splitext(data)[1] == ".pkl":
                self.data = pickle.load(open(data, "rb"))
            elif splitext(data)[1] == ".json":
                self.data = json.load(open(data, "r"))["data"]
            elif splitext(data)[1] == ".json":
                self.data =pd.read_csv(data)
            else: raise AssertionError("file not pickle or json or csv")

        elif isinstance(data, list):
            # list of tuples provided
            self.data = data
        else: raise AssertionError("either list of tuples or filepath to pkl")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]