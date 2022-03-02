import numpy as np
import random
import torch
from argparse import ArgumentParser
from typing import List, Tuple, Optional

from helpers.basic_classes import PoolType, DataSet


def convert_underscore_to_camelcase(word: str):
    return ''.join(x.capitalize() or '_' for x in word.split('_'))


def record_args(args: ArgumentParser) -> str:
    """
        print the arguments and creates the name of the run
        :param args: ArgumentParser - command line inputs
        :return: name_of_run: str
    """
    name_of_run = ''

    for arg in vars(args):
        arg_value = getattr(args, arg)
        print(f"{arg}: {arg_value}", flush=True)

        if arg == 'dataset':
            name_of_run += DataSet(arg_value).name.capitalize() + '_'
            continue
        if arg == 'pool_type' and arg_value is not PoolType.NONE:
            name_of_run += '_' + PoolType(arg_value).name + 'Pool'
            continue

        camelcase_arg = convert_underscore_to_camelcase(word=arg)

        if isinstance(arg_value, bool):
            if arg_value:
                name_of_run += '_' + camelcase_arg
        else:
            name_of_run += '_' + camelcase_arg + str(arg_value)
    print(flush=True)
    return name_of_run


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_results_list(results_list: List[Tuple[str, float]], epoch: Optional[int] = None):
    result_str = ", ".join(key + ': ' + str(value) for key, value in results_list)
    if epoch is None:
        result_str = 'Final, ' + result_str
    else:
        result_str = f'Epoch: {epoch}, ' + result_str
    print(result_str, flush=True)
