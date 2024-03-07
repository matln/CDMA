"""
Copyright 2019 Snowdar
          2022 Jianchen Li
"""

import os
import csv
import sys
import math
import time
import copy
import torch
import random
import pickle
import shutil
import signal
import inspect
import logging
import torchaudio
import numpy as np
import pandas as pd
from typing import Optional
import torch.distributed as dist
from hyperpyyaml import resolve_references


# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def to_bool(variable):
    """Transform string to bool if variable is not bool
    """
    if not isinstance(variable, bool):
        if not isinstance(variable, str):
            raise TypeError("variable is not str or bool type")
        else:
            return True if variable == "true" or variable == "True" else False
    else:
        return variable


def parse_gpu_id_option(gpu_id):
    """
    @gpu_id: str: 1,2,3 or 1-2-3 or "1 2 3"
              int: 1
              list/tuple: [1,2,3] or ("1","2","3")
    """
    if isinstance(gpu_id, str):
        gpu_id = gpu_id.replace("-", " ")
        gpu_id = gpu_id.replace(",", " ")
        gpu_id = [int(x) for x in gpu_id.split()]
    elif isinstance(gpu_id, int):
        gpu_id = [gpu_id]
    elif isinstance(gpu_id, (list, tuple)):
        gpu_id = [int(x) for x in gpu_id]
    else:
        raise TypeError("Expected str, int or list/tuple, bug got {}.".format(gpu_id))
    return gpu_id


def select_model_device(model, use_gpu, gpu_id="", benchmark=False, log=True):
    """ Auto select device (cpu/GPU) for model
    @use_gpu: bool or 'true'/'false' string
    """
    model.cpu()

    use_gpu = to_bool(use_gpu)
    benchmark = to_bool(benchmark)

    if use_gpu:
        torch.backends.cudnn.benchmark = benchmark

        if gpu_id == "":
            logger.info(
                "The use_gpu is true and gpu id is not specified, so select gpu device automatically."
            )
            import libs.support.GPU_Manager as gpu

            gm = gpu.GPUManager()
            gpu_id = [gm.auto_choice()]
        else:
            # Get a gpu id list.
            gpu_id = parse_gpu_id_option(gpu_id)
            if is_main_training() and log:
                logger.info("The use_gpu is true and training will use GPU {0}.".format(gpu_id))

        # Multi-GPU with DDP.
        if len(gpu_id) > 0 and use_ddp():
            if dist.get_world_size() != len(gpu_id):
                raise ValueError(
                    "To run DDP with {} nj, "
                    "but {} GPU ids ({}) are given.".format(
                        dist.get_world_size(), len(gpu_id), gpu_id
                    )
                )
            torch.cuda.set_device(gpu_id[dist.get_rank()])
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[gpu_id[dist.get_rank()]], output_device=dist.get_rank(),
            )
            return model

        # DataParallel
        elif len(gpu_id) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_id))
            model = torch.nn.DataParallel(model)  # Multiple GPUs
            # model = convert_model(model)
        else:
            # One process in one GPU.
            # os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
            torch.cuda.set_device(gpu_id[0])
        model.cuda()
    return model


def to_device(device_object, tensor):
    """
    Select device for non-parameters tensor w.r.t model or tensor which has been specified a device.
    """
    if isinstance(device_object, torch.nn.Module):
        try:
            device = next(device_object.parameters()).device
            return tensor.to(device)
        except StopIteration:
            return tensor
    elif isinstance(device_object, torch.Tensor):
        device = device_object.device
        return tensor.to(device)


def get_device(model):
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    return device


def get_device_from_optimizer(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            return p.device


def get_tensors(tensor_sets):
    """Get a single tensor list from a nested tensor_sets list/tuple object,
    such as transforming [(tensor1,tensor2),tensor3] to [tensor1,tensor2,tensor3]
    """
    tensors = []
    other_param = []

    for this_object in tensor_sets:
        # Only tensor
        if isinstance(this_object, torch.Tensor):
            tensors.append(this_object)
        elif isinstance(this_object, np.ndarray):
            tensors.append(torch.from_numpy(this_object))
        elif isinstance(this_object, list) or isinstance(this_object, tuple):
            tensors.extend(get_tensors(this_object))
        else:
            other_param.append(this_object)

    return tensors, other_param


def for_device_free(function):
    """
    A decorator to make class-function with input-tensor device-free
    Used in libs.nnet.framework.TopVirtualNnet
    """

    def wrapper(self, *param_sets):
        transformed = []

        tensors, other_param = get_tensors(param_sets)
        # 申请实例后，self 会被替换成实例
        for tensor in tensors:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                transformed.append(tensor.cuda())
            else:
                transformed.append(to_device(self, tensor))

        return function(self, *transformed, *other_param)

    return wrapper


def import_module(model_blueprint, model_creation=""):
    """
    Used in pipeline/train.py and pipeline/onestep/extract_emdeddings.py and it makes config of nnet
    more free with no-change of training and other common scripts.

    @model_blueprint: string type, a *.py file path which includes the instance of nnet, such as examples/xvector.py
    @model_creation: string type, a command to create the model class according to the class declaration
                     in model_blueprint, such as using 'Xvector(40,2)' to create an Xvector nnet.
                     Note, it will return model_module if model_creation is not given, else return model.
    """
    if not os.path.exists(model_blueprint):
        raise TypeError("Expected {} to exist.".format(model_blueprint))
    if os.path.getsize(model_blueprint) == 0:
        raise TypeError("There is nothing in {}.".format(model_blueprint))

    sys.path.insert(0, os.path.dirname(model_blueprint))
    model_module_name = os.path.basename(model_blueprint).split(".")[0]
    # 动态导入模块就是只知道str类型的模块名字符串，通过这个字符串导入模块
    model_module = __import__(model_module_name)

    if model_creation == "":
        return model_module
    else:
        model = eval("model_module.{0}".format(model_creation))
        return model


def write_nnet_config(model_blueprint: str, model_creation: str, nnet_config: str):
    dataframe = pd.DataFrame(
        [model_blueprint, model_creation], index=["model_blueprint", "model_creation"]
    )
    dataframe.to_csv(nnet_config, header=None, sep="\t")
    logger.info(
        f"Save nnet_config to [bold cyan]{nnet_config}[/bold cyan] done.", extra={"markup": True},
    )


def read_nnet_config(nnet_config: str, log=True):
    if log:
        logger.info("Read nnet_config from {0}".format(nnet_config))
    # Use ; sep to avoid some problem in spliting.
    dataframe = pd.read_csv(nnet_config, header=None, index_col=0, sep="\t")
    model_blueprint = dataframe.loc["model_blueprint", 1]
    model_creation = dataframe.loc["model_creation", 1]

    return model_blueprint, model_creation


def create_model_dir(model_dir: str, debug: bool = False):
    if not debug and is_main_training():
        if not os.path.exists("{0}/log".format(model_dir)):
            os.makedirs("{0}/log".format(model_dir), exist_ok=True)

        if not os.path.exists("{0}/backup".format(model_dir)):
            os.makedirs("{0}/backup".format(model_dir), exist_ok=True)

        if not os.path.exists("{0}/config".format(model_dir)):
            os.makedirs("{0}/config".format(model_dir), exist_ok=True)

        if not os.path.exists("{0}/checkpoints".format(model_dir)):
            os.makedirs("{0}/checkpoints".format(model_dir), exist_ok=True)


def draw_list_to_png(list_x, list_y, out_png_file, color="r", marker=None, dpi=256):
    """ Draw a piture for some values.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(list_x, list_y, color=color, marker=marker)
    plt.savefig(out_png_file, dpi=dpi)
    plt.close()


def read_file_to_list(file_path, every_bytes=10000000):
    list = []
    with open(file_path, "r") as reader:
        while True:
            lines = reader.readlines(every_bytes)
            if not lines:
                break
            for line in lines:
                list.append(line)
    return list


def write_list_to_file(this_list, file_path, mod="w"):
    """
    @mod: could be 'w' or 'a'
    """
    if not isinstance(this_list, list):
        this_list = [this_list]

    with open(file_path, mod) as writer:
        writer.write("\n".join(str(x) for x in this_list))
        writer.write("\n")


def save_checkpoint(checkpoint_path, **kwargs):
    """Save checkpoint to file for training. Generally, The checkpoint includes
        epoch:<int>
        iter:<int>
        model_path:<string>
        optimizer:<optimizer.state_dict>
        lr_scheduler:<lr_scheduler.state_dict>
    """
    state_dict = {}
    state_dict.update(kwargs)
    torch.save(state_dict, checkpoint_path)


def format(x, str):
    """To hold on the None case when formating float to string.
    @x: a float value or None or any others, should be consistent with str
    @str: a format such as {:.2f}
    """
    if x is None:
        return "-"
    else:
        return str.format(x)


def set_seed(seed=None, deterministic=True):
    """This is refered to https://github.com/lonePatient/lookahead_pytorch/blob/master/tools.py.
    """
    if seed is not None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = deterministic


def key_to_value(adict, key, return_none=True):
    assert isinstance(adict, dict)

    if key in adict.keys():
        return adict[key]
    elif return_none:
        return None
    else:
        return key


def assign_params_dict(default_params: dict, params: dict, force_check=False, support_unknow=True):
    default_params = copy.deepcopy(default_params)
    default_keys = set(default_params.keys())

    # Should keep force_check=False to use support_unknow
    if force_check:
        for key in params.keys():
            if key not in default_keys:
                raise ValueError("The params key {0} is not in default params".format(key))

    # Do default params <= params if they have the same key
    params_keys = set(params.keys())
    for k, v in default_params.items():
        if k in params_keys:
            if isinstance(v, type(params[k])):
                if isinstance(v, dict):
                    # To parse a sub-dict.
                    sub_params = assign_params_dict(v, params[k], force_check, support_unknow)
                    default_params[k] = sub_params
                else:
                    default_params[k] = params[k]
            elif isinstance(v, float) and isinstance(params[k], int):
                default_params[k] = params[k] * 1.0
            elif v is None or params[k] is None:
                default_params[k] = params[k]
            else:
                raise ValueError(
                    "The value type of default params [{0}] is "
                    "not equal to [{1}] of params for k={2}".format(
                        type(default_params[k]), type(params[k]), k
                    )
                )

    # Support unknow keys
    if not force_check and support_unknow:
        for key in params.keys():
            if key not in default_keys:
                default_params[key] = params[key]

    return default_params


def split_params(params: dict):
    params_split = {"public": {}}
    params_split_keys = params_split.keys()
    for k, v in params.items():
        if len(k.split(".")) == 2:
            name, param = k.split(".")
            if name in params_split_keys:
                params_split[name][param] = v
            else:
                params_split[name] = {param: v}
        elif len(k.split(".")) == 1:
            params_split["public"][k] = v
        else:
            raise ValueError("Expected only one . in key, but got {0}".format(k))

    return params_split


def auto_str(value, auto=True):
    if isinstance(value, str) and auto:
        return "'{0}'".format(value)
    else:
        return str(value)


def iterator_to_params_str(iterator, sep=",", auto=True):
    return sep.join(auto_str(x, auto) for x in iterator)


def dict_to_params_str(dict, auto=True, connect="=", sep=","):
    params_list = []
    for k, v in dict.items():
        if "\033[" in k:
            prefix = k.split("m")[0] + "m"
            suffix = "\033[0m"
            k = k.split("m", 1)[1]
        else:
            prefix = ""
            suffix = ""
        params_list.append(f"{prefix}{k + connect + auto_str(v, auto)}{suffix}")
    return iterator_to_params_str(params_list, sep, False)


def read_log_csv(csv_path: str):
    dataframe = pd.read_csv(csv_path).drop_duplicates(["epoch", "iter"], keep="last", inplace=True)
    return dataframe


# Multi-GPU training [Two solutions: Horovod or DDP]
def init_multi_gpu_training(gpu_id="", solution="ddp", port=29500):
    num_gpu = len(parse_gpu_id_option(gpu_id))
    if num_gpu > 1:
        # The DistributedDataParallel (DDP) solution is suggested.
        if solution == "ddp":
            init_ddp(port)
            if is_main_training():
                logger.info("DDP has been initialized.")
        elif solution == "horovod":
            init_horovod()
            if is_main_training():
                logger.info("Horovod has been initialized.")
        elif solution == "dp":
            # logger.info("DP has been initialized.")
            pass
        else:
            raise TypeError("Do not support {} solution for multi-GPU training.".format(solution))


def convert_synchronized_batchnorm(model):
    if use_ddp():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def is_main_training():
    if use_horovod():
        import horovod.torch as hvd

        # Set rank=0 to main training process. See trainer.init_training().
        if hvd.rank() == 0:
            return True
        else:
            return False
    elif use_ddp():
        if dist.get_rank() == 0:
            return True
        else:
            return False
    return True


def auto_scale_lr(lr):
    if use_horovod():
        import horovod.torch as hvd

        return lr * hvd.size()
    elif use_ddp():
        return lr * dist.get_world_size()
    else:
        return lr


# Horovod


def init_horovod():
    os.environ["USE_HOROVOD"] = "true"
    import horovod.torch as hvd

    hvd.init()


def use_horovod():
    return os.getenv("USE_HOROVOD") == "true"


# DDP


def init_ddp(port=29500):
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("NCCL is not available.")

    # Just plan to support NCCL for GPU-Training with single machine, but it is easy to extend by yourself.
    # Init_method is defaulted to 'env://' (environment) and The IP is 127.0.0.1 (localhost).
    # Based on this init_method, world_size and rank will be set automatically with DDP,
    # so do not give these two params to init_process_group.
    # The port will be always defaulted to 29500 by torch that will result in init_process_group failed
    # when number of training task > 1. So, use subtools/pytorch/launcher/multi_gpu/get_free_port.py to get a
    # free port firstly, then give this port to launcher by --port. All of these have been auto-set by runLauncher.sh.

    # os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(backend="nccl")


def use_ddp():
    return torch.distributed.is_initialized()


def cleanup_ddp():
    torch.distributed.destroy_process_group()


def get_free_port(ip="127.0.0.1"):
    import socket

    # Use contextlib to close socket after return the free port.
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # Set port as 0, socket will auto-select a free port. And then fetch this port.
        s.bind((ip, 0))
        return s.getsockname()[1]


def convert_dict_to_yaml(nested_dict: {}, indent=4):
    yaml_string = ""

    def _convert_dict(_nested_dict, _yaml_string, nested_layer, indent=4):
        for key, value in _nested_dict.items():
            if type(value) == dict:
                _yaml_string += "\n{}{}:".format(" " * indent * nested_layer, key)
                _yaml_string = _convert_dict(value, _yaml_string, nested_layer + 1, indent=indent)
            else:
                _yaml_string += "\n{}{}: {}".format(" " * indent * nested_layer, key, value)
        return _yaml_string

    yaml_string = _convert_dict(nested_dict, yaml_string, 0, indent=indent)

    return yaml_string.strip()


def convert_to_yaml(overrides, indent=4):
    """Convert args to yaml for overrides. Handle '--arg=val' and '--arg val' type args
       indent: yaml indent
    """
    # Construct nested dict
    yaml_dict = {}
    for arg in overrides:
        arg = arg.replace("--", "")
        if "=" in arg:
            keys, value = arg.split("=")
        elif " " in arg:
            keys, value = arg.split(" ")
        else:
            raise ValueError
        keys = keys.split(".")
        _yaml_dict = yaml_dict
        key = keys.pop(0)
        while keys:
            _yaml_dict = _yaml_dict.setdefault(key, {})
            key = keys.pop(0)
        _yaml_dict[key] = value

    # Construct yaml string
    yaml_string = convert_dict_to_yaml(yaml_dict, indent=indent)

    return yaml_string


def save_pkl(obj, file):
    """Save an object in pkl format. (speechbrain)

    Arguments
    ---------
    obj : object
        Object to save in pkl format
    file : str
        Path to the output file
    sampling_rate : int
        Sampling rate of the audio file, TODO: this is not used?

    Example
    -------
    >>> tmpfile = os.path.join(getfixture('tmpdir'), "example.pkl")
    >>> save_pkl([1, 2, 3, 4, 5], tmpfile)
    >>> load_pkl(tmpfile)
    [1, 2, 3, 4, 5]
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    """Loads a pkl file. (speechbrain)

    For an example, see `save_pkl`.

    Arguments
    ---------
    file : str
        Path to the input pkl file.

    Returns
    -------
    The loaded object.
    """

    # Deals with the situation where two processes are trying
    # to access the same label dictionary by creating a lock
    count = 100
    while count > 0:
        if os.path.isfile(file + ".lock"):
            time.sleep(1)
            count -= 1
        else:
            break

    try:
        open(file + ".lock", "w").close()
        with open(file, "rb") as f:
            return pickle.load(f)
    finally:
        if os.path.isfile(file + ".lock"):
            os.remove(file + ".lock")


def format_order_of_magnitude(number, abbreviate=True):
    """Formats number to the appropriate order of magnitude for printing.

    Arguments
    ---------
    number : int, float
        The number to format.
    abbreviate : bool
        Whether to use abbreviations (k,M,G) or words (Thousand, Million,
        Billion). Numbers will be either like: "123.5k" or "123.5 Thousand".

    Returns
    -------
    str
        The formatted number. Note that the order of magnitude token is part
        of the string.

    Example
    -------
    >>> print(format_order_of_magnitude(123456))
    123.5k
    >>> print(format_order_of_magnitude(0.00000123, abbreviate=False))
    1.2 millionths
    >>> print(format_order_of_magnitude(5, abbreviate=False))
    5
    """

    orders_abbrev = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "µ",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
    }

    # Short scale
    # Negative powers of ten in lowercase, positive in uppercase
    orders_words = {
        -24: "septillionths",
        -21: "sextillionths",
        -18: "quintillionths",
        -15: "quadrillionths",
        -12: "trillionths",
        -9: "billionths",
        -6: "millionths",
        -3: "thousandths",
        0: "",
        3: "Thousand",
        6: "Million",
        9: "Billion",
        12: "Trillion",
        15: "Quadrillion",
        18: "Quintillion",
        21: "Sextillion",
        24: "Septillion",
    }
    style = orders_abbrev if abbreviate else orders_words
    precision = "{num:3.1f}"
    order = 3 * math.floor(math.log(math.fabs(number), 1000))
    # Fallback for very large numbers:
    while order not in style and order != 0:
        order = order - math.copysign(3, order)  # Bring 3 units towards 0
    order_token = style[order]
    if order != 0:
        formatted_number = precision.format(num=number / 10 ** order)
    else:
        if isinstance(number, int):
            formatted_number = str(number)
        else:
            formatted_number = precision.format(num=number)
    if abbreviate or not order_token:
        return formatted_number + order_token
    else:
        return formatted_number + " " + order_token


def load_data_csv(
    csv_path, replacements={}, repl_field="wav_path", id_field="utt_id", delimiter=","
):
    """Loads CSV and formats string values.

    Uses the CSV data format, where the CSV must have an
    'utt_id' field.
    If there is a field called duration, it is interpreted as a float.
    The rest of the fields are left as they are (legacy _format and _opts fields
    are not used to load the data in any special way).

    String replacements are supported.

    Arguments
    ----------
    csv_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), e.g., {"/data/corpus": "/home/corpus"}
        This is used to format the value of repl_field.
    repl_field : str
    id_field: str

    Returns
    -------
    dict
        CSV data with replacements applied.

    Example
    -------
    >>> csv_spec = '''utt_id,wav_path,duration
    ... utt1,/data/corpus/utt1.wav,1.45
    ... utt2,/data/corpus/utt2.wav,2.0
    ... '''
    >>> tmpfile = getfixture("tmpdir") / "test.csv"
    >>> with open(tmpfile, "w") as fo:
    ...     _ = fo.write(csv_spec)
    >>> data = load_data_csv(tmpfile, {"/data/corpus": "/home/corpus"}, repl_field="wav_path")
    >>> data["utt1"]["wav_path"]
    '/home/corpus/utt1.wav'
    """

    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True, delimiter=delimiter)
        # variable_finder = re.compile(r"\$([\w.]+)")
        for row in reader:
            # ID:
            try:
                data_id = row[f"{id_field}"]
                del row[f"{id_field}"]  # This is used as a key in result, instead.
            except KeyError:
                raise KeyError(
                    "CSV has to have an 'utt_id' field, with unique ids" " for all data points"
                )
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            for key, value in row.items():
                try:
                    # row[key] = variable_finder.sub(
                    #     lambda match: str(replacements[match[1]]), value
                    # )
                    if key == repl_field:
                        for repl, target in replacements.items():
                            if repl in value:
                                row[key] = value.replace(repl, target)
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements " "which were not supplied."
                    )
            # Duration:
            if "duration" in row:
                row["duration"] = float(row["duration"])
            result[data_id] = row
    return result


def backup_code(args, hparams, model_dir, overrides, description=""):
    if is_main_training() and not args.debug:
        if "aug" in hparams["dataset_params"] and hparams["dataset_params"]["aug"]:
            overrides.append(f"dataset_params.aug_conf={model_dir}/config/augmentation.yaml")
            shutil.copy(
                hparams["dataset_params"]["aug_conf"], f"{model_dir}/config/augmentation.yaml",
            )
        if "source_aug" in hparams["dataset_params"] and hparams["dataset_params"]["source_aug"]:
            overrides.append(
                f"dataset_params.source_aug_conf={model_dir}/config/source_augmentation.yaml"
            )
            shutil.copy(
                hparams["dataset_params"]["source_aug_conf"],
                f"{model_dir}/config/source_augmentation.yaml",
            )
        if "target_aug" in hparams["dataset_params"] and hparams["dataset_params"]["target_aug"]:
            overrides.append(
                f"dataset_params.target_aug_conf={model_dir}/config/target_augmentation.yaml"
            )
            shutil.copy(
                hparams["dataset_params"]["target_aug_conf"],
                f"{model_dir}/config/target_augmentation.yaml",
            )
        overrides.append(f"bunch={model_dir}/backup/{hparams['bunch']}")
        overrides.append(f"encoder={model_dir}/backup/{hparams['encoder']}")
        overrides.append(f"trainer={model_dir}/backup/{hparams['trainer']}")

        shutil.copytree("local", os.path.join(model_dir, "backup/local"))
        os.makedirs(os.path.join(model_dir, "backup/scripts"))
        for file in os.listdir("./"):
            if os.path.isfile(file):
                shutil.copy(file, os.path.join(model_dir, "backup/scripts"))

        # Copy executing file to output directory
        module = inspect.getmodule(inspect.currentframe().f_back)
        if module is not None:
            callingfile = os.path.realpath(module.__file__)
            shutil.copy(callingfile, os.path.join(model_dir, "backup"))

        # Copy the relevant core files to backuo directory
        shutil.copytree(f"{os.getenv('speakernet')}", os.path.join(model_dir, "backup/speakernet"))

        # Write the parameters file
        hyperparams_filename = os.path.join(model_dir, "config/hyperparams.yaml")
        overrides_yaml = convert_to_yaml(overrides)
        with open(args.hparams_file) as f:
            resolved_yaml = resolve_references(f, overrides_yaml, indent=4)
        with open(hyperparams_filename, "w") as w:
            print("# %s" % os.path.abspath(args.hparams_file), file=w)
            print("# yamllint disable\n", file=w)
            shutil.copyfileobj(resolved_yaml, w)

            # Command line arguments
            print(
                "\n# -------------------------------------"
                "------------------------------------------------- #",
                file=w,
            )
            print("# Command line arguments", file=w)
            args_yaml = convert_dict_to_yaml(args.__dict__)
            print(args_yaml, file=w)

        # write the description
        if description == "":
            exps = os.listdir(model_dir.rsplit("/", 1)[0])
            if exps != []:
                sorted_exps = sorted(
                    exps, key=lambda x: time.mktime(time.strptime(x, "%Y-%m-%d_%H:%M:%S")),
                )
                if len(sorted_exps) > 1:
                    last_readme = os.path.join(
                        model_dir.rsplit("/", 1)[0], sorted_exps[-2], "README"
                    )
                    if os.path.exists(last_readme):
                        with open(last_readme, "r") as r:
                            description = r.readline().strip()

        with open(os.path.join(model_dir, "README"), "w") as w:
            print(f"{description}", file=w)


def alarm_handler(signum, frame):
    raise TimeoutError


def input_with_timeout(prompt="", timeout=60):
    # set signal handler
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)  # produce SIGALRM in `timeout` seconds

    try:
        return input(prompt)
    except TimeoutError:
        print("Time up! Input is none")
        return ""
    finally:
        signal.alarm(0)  # cancel alarm


def load_wavs(
    wav_paths: str,
    start_position: Optional[str] = None,
    end_position: Optional[str] = None,
    return_size=False,
):
    """
    Load multiple wavs
    """
    wav_paths = wav_paths.split(" ")
    if len(wav_paths) == 1:
        if start_position is None and end_position is None:
            sig, fs = torchaudio.load(wav_paths[0])
            wav_size = sig.size(1)
        elif "_" not in start_position and "_" not in end_position:
            num_frames = int(end_position) - int(start_position)
            sig, fs = torchaudio.load(
                wav_paths[0], num_frames=num_frames, frame_offset=int(start_position)
            )
            wav_size = sig.size(1)
        else:
            raise ValueError

        if not return_size:
            return sig, fs
        else:
            return sig, fs, wav_size
    else:
        if start_position is None and end_position is None:
            wav_size = []
            sig, fs = torchaudio.load(wav_paths[0])
            wav_size.append(sig.size(1))
            for wav_path in wav_paths[1:]:
                _sig, _fs = torchaudio.load(wav_path)
                assert fs == _fs
                wav_size.append(_sig.size(1))
                sig = torch.cat((sig, _sig), dim=1)
            assert sum(wav_size) == sig.size(1)

            if not return_size:
                return sig, fs
            else:
                return sig, fs, wav_size

        elif "_" in start_position and "_" in end_position:
            start_idx, start_position = [int(x) for x in start_position.split("_")]
            end_idx, end_position = [int(x) for x in end_position.split("_")]
            assert end_idx >= start_idx
            if end_idx == start_idx:
                sig, fs = torchaudio.load(
                    wav_paths[start_idx],
                    num_frames=end_position - start_position,
                    frame_offset=start_position,
                )
            else:
                sig, fs = torchaudio.load(
                    wav_paths[start_idx], num_frames=-1, frame_offset=start_position
                )
                for idx in range(start_idx + 1, end_idx):
                    _sig, _fs = torchaudio.load(wav_paths[idx], num_frames=-1, frame_offset=0)
                    assert fs == _fs
                    sig = torch.cat((sig, _sig), dim=1)
                _sig, _fs = torchaudio.load(
                    wav_paths[end_idx], num_frames=end_position, frame_offset=0
                )
                assert fs == _fs
                sig = torch.cat((sig, _sig), dim=1)
            return sig, fs
        else:
            raise ValueError


def read_trials(trials):
    scores = []
    with open(trials, "r") as f:
        lines = f.readlines()
        for line in lines:
            scores.append(line.strip().split())
    return scores


def read_scp(scp_file, value_type="str", multi_value=False):
    _dict = {}
    with open(scp_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if not multi_value:
                _dict[line[0]] = eval(f"{value_type}(line[1])")
            else:
                _dict[line[0]] = [eval(f"{value_type}(x)") for x in line[1:]]
    return _dict


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None


if __name__ == "__main__":
    write_nnet_config(
        "11.............111111111111111111",
        "222222222222222222.212412",
        "/home/lijianchen/workspace/sre/voxceleb/tmp/nnet.config",
    )
