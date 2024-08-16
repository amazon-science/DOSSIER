import os
import sys
import torch
import numpy as np
import json
from fact_check import constants
import pathlib
from pathlib import PosixPath
import collections

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

def to_device(obj, device):
    if torch.is_tensor(obj) or isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return [obj.to(device) for i in obj]
    elif isinstance(obj, (dict, collections.abc.Mapping)):
        return {a: b.to(device) if torch.is_tensor(b) or isinstance(b, torch.nn.Module) else b for a,b in obj.items() }
    else:
        raise ValueError("invalid object type passed to to_device")
    
def path_serial(obj):
    if isinstance(obj, (pathlib.Path, PosixPath)):
        return str(obj)
    raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")

class NumpyEncoder(json.JSONEncoder):
    # https://github.com/hmallen/numpyencoder/blob/f8199a61ccde25f829444a9df4b21bcb2d1de8f2/numpyencoder/numpyencoder.py
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None

        return json.JSONEncoder.default(self, obj)    