from datetime import datetime
import dis
import hashlib
import inspect
import os
import re
import pandas as pd
import json
import pickle
from typing import Callable, Literal, Tuple, Union

_PYCHECKPOINT_DIRNAME_TEMPLATE = "{identifier}_[{date}]_{hash}_pycheckpoint"
_PYCHECKPOINT_FILENAME_TEMPLATE = "{arg_repr}_[{date}]_{arg_hash}_pycheckpoint.{ext}"
_PYCHECKPOINT_TMP_FILENAME_TEMPLATE = "{arg_repr}_[{date}]_{arg_hash}_pycheckpoint.incomplete.{ext}"
_PYCHECKPOINT_DATETIME_FORMAT = "%m-%d-%Y-%H-%M-%S"
_PYCHECKPOINT_DEFAULT_LOGGING_MESSAGE = "[{cur_time}] Pycheckpoint: Loading checkpoint created at {date}. Function: {func}, Args: {args}"


def _pycheckpoint_validify_filename(filename: str) -> str:
    return re.sub(r'[^\w_\-\.\(\)\[\]\{\}\+=,~]', '', filename)

def _pycheckpoint_validify_function(func: Callable) -> Tuple[bool, str]:
    if len(func.__code__.co_freevars) > 0:
        return False, "Function has free variables"
    global_mem_instr = {
        "LOAD_GLOBAL",
        "STORE_GLOBAL",
        "LOAD_NAME",
        "STORE_NAME",
        "LOAD_FROM_DICT_OR_GLOBALS",
        "DELETE_GLOBAL",
        "DELETE_NAME",
    }
    for instr in dis.Bytecode(func):
        if instr.opname in global_mem_instr:
            return False, f"Function may use global memory: {instr.opname}. {instr}"

    return True, ""

def _pycheckpoint_fingerprint_function(func: Callable) -> str:
    consts = func.__code__.co_consts
    names = func.__code__.co_names
    bytecodes = func.__code__.co_code
    return hashlib.sha256(pickle.dumps((consts, names, bytecodes))).hexdigest()


def pycheckpoint(
        checkpoint_path: Union[str, os.PathLike]=".pycheckpoint",
        serialization: Union[Literal["pickle", "pandas.csv", "pandas.parquet", "json"], Tuple[Callable, Callable, str]]="pickle",
        serialize_function_kwargs: Tuple[dict, dict]=({}, {}),
        canonical_args: bool=True,
        log_message_template: str=_PYCHECKPOINT_DEFAULT_LOGGING_MESSAGE,
    )-> Callable:
    """
    Decorator to checkpoint a function. The function will be checkpointed based on the function name, the function bytecode, and the arguments. 

    The checkpointed results are stored in the `checkpoint_path` directory. 
    The checkpointed results are stored in a directory named
    `function_name_[date]_hash_pycheckpoint` where `function_name` is the name of the function, 
    `date` is the date of the checkpoint, and `hash` is the sha256 hash of the function bytecode. 
    The checkpointed results are stored in files named `args_repr_[date]_arg_hash_pycheckpoint.extension` 
    where `args_repr` is a string representation of the arguments, `date` is the date of the checkpoint,
    `arg_hash` is the sha256 hash of the arguments, and `extension` is the file extension. 
    The file extension is determined by the `serialization` argument.
    A copy of the function source code is stored in a file named `function_name_source.py` in the checkpoint directory.

    Args:
        checkpoint_path (Union[str, os.PathLike], optional): Path to store checkpoints. Defaults to ".pycheckpoint".
        serialization (Union[Literal["pickle", "pandas.csv", "pandas.parquet", "json"], Tuple[Callable, Callable, str], optional): 
            Serialization method. Defaults to "pickle". If Tuple, the first element is the serialization function, 
            the second element is the deserialization function, and the third element is the file extension. 
            The serialization function should have the signature `def serialize_function(obj, path, **kwargs)`.
            The deserialization function should have the signature `def deserialize_function(path, **kwargs)`.
        serialize_function_kwargs (Tuple[dict, dict], optional): Keyword arguments for serialization and deserialization functions. Defaults to ({}, {}).
        canonical_args (bool, optional): If True, the function will be checkpointed based on the canonical arguments. If true, all positional arguments 
            and kwargs are represented in a unified form. Defaults to True.
        log_message_template (str, optional): Log message template. Defaults to _PYCHECKPOINT_DEFAULT_LOGGING_MESSAGE. Supported placeholders are:
            - cur_time: Current time
            - file_path: File path
            - filename: Filename
            - date: Date
                - year: Year
                - month: Month
                - day: Day
                - hour: Hour
                - minute: Minute
                - second: Second
                - weekday: Weekday
            - func: Function name
            - func_hash: Function hash
            - args: Arguments
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            if serialization == "pickle":
                def serialize_function(obj, path):
                    with open(path, "wb") as f:
                        pickle.dump(obj, f, **serialize_function_kwargs[0])
                def deserialize_function(path):
                    with open(path, "rb") as f:
                        return pickle.load(f, **serialize_function_kwargs[1])
                filename_extension = "pkl"
            elif serialization == "pandas.csv":
                def serialize_function(obj, path):
                    obj.to_csv(path, **serialize_function_kwargs[0])
                def deserialize_function(path):
                    return pd.read_csv(path, **serialize_function_kwargs[1])
                filename_extension = "csv"
            elif serialization == "pandas.parquet":
                def serialize_function(obj, path):
                    obj.to_parquet(path, **serialize_function_kwargs[0])
                def deserialize_function(path):
                    return pd.read_parquet(path, **serialize_function_kwargs[1])
                filename_extension = "parquet"
            elif serialization == "json":
                def serialize_function(obj, path):
                    with open(path, "w") as f:
                        json.dump(obj, f, **serialize_function_kwargs[0])
                def deserialize_function(path):
                    with open(path, "r") as f:
                        return json.load(f, **serialize_function_kwargs[1])
                filename_extension = "json"
            else:
                def serialize_function(obj, path):
                    serialization[0](obj, path, **serialize_function_kwargs[0])
                def deserialize_function(path):
                    return serialization[1](path, **serialize_function_kwargs[1])
                filename_extension = _pycheckpoint_validify_filename(serialization[2])

            identifier = _pycheckpoint_validify_filename(func.__qualname__)

            valid, reason = _pycheckpoint_validify_function(func)
            if not valid:
                raise ValueError(f"Function cannot be checkpointed: {reason}")
            
            func_hash = _pycheckpoint_fingerprint_function(func)
            
            # stringify args and kwargs
            if canonical_args:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                canonical_kwargs = bound_args.arguments
                all_args_repr = ",".join([f"{key}={repr(value)}" for key, value in canonical_kwargs.items()])[:160]
                all_args_repr = _pycheckpoint_validify_filename(all_args_repr)
                arg_hash_obj = hashlib.sha256(pickle.dumps(canonical_kwargs))
                arg_hash = arg_hash_obj.hexdigest()
            else:
                args_repr = ",".join([str(arg) for arg in args])[:80]
                kwargs_repr = ",".join(f"{key}-{repr(value)}" for key, value in kwargs.items())[:80]
                all_args_repr = f"{args_repr}_{kwargs_repr}"
                all_args_repr = _pycheckpoint_validify_filename(all_args_repr)

                args_tuple = tuple(args)
                kwargs_tuple = tuple(sorted(kwargs.items()))
                arg_hash_obj = hashlib.sha256(pickle.dumps((args_tuple, kwargs_tuple)))
                arg_hash = arg_hash_obj.hexdigest()

            date = datetime.now().strftime(_PYCHECKPOINT_DATETIME_FORMAT)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            
            checkpoint_dir = None
            for dir in os.listdir(checkpoint_path):
                if dir.startswith(f"{identifier}_") and dir.endswith(f"_{func_hash}_pycheckpoint"):
                    checkpoint_dir = dir
                    break
            
            if not checkpoint_dir:
                checkpoint_dir = _PYCHECKPOINT_DIRNAME_TEMPLATE.format(
                    identifier=identifier,
                    hash=func_hash,
                    date=date,
                )
                os.makedirs(os.path.join(checkpoint_path, checkpoint_dir))
                src = inspect.getsource(func)
                with open(os.path.join(checkpoint_path, checkpoint_dir, f"{identifier}_source.py"), "w") as f:
                    f.write(src)

            checkpoint_dir_path = os.path.join(checkpoint_path, checkpoint_dir)


            for file in os.listdir(checkpoint_dir_path):
                prefix, suffix = f"{all_args_repr}_", f"_{arg_hash}_pycheckpoint.{filename_extension}"
                if file.endswith(suffix):
                    checkpoint_date = file[len(prefix):-(len(suffix))]
                    assert checkpoint_date.startswith("[") and checkpoint_date.endswith("]"), f"Invalid checkpoint date: {checkpoint_date} in file {os.path.join(checkpoint_dir_path, file)}"
                    checkpoint_date = checkpoint_date[1:-1]
                    try: 
                        checkpoint_date = datetime.strptime(checkpoint_date, _PYCHECKPOINT_DATETIME_FORMAT)
                    except ValueError:
                        raise ValueError(f"Invalid checkpoint date format: {checkpoint_date} in file {os.path.join(checkpoint_dir_path, file)}")
                    
                    print(log_message_template.format(
                        cur_time = datetime.now(),
                        file_path=os.path.join(checkpoint_dir_path, file),
                        filename = file,
                        date = checkpoint_date,
                        year = checkpoint_date.year,
                        month = checkpoint_date.month,
                        day = checkpoint_date.day,
                        hour = checkpoint_date.hour,
                        minute = checkpoint_date.minute,
                        second = checkpoint_date.second,
                        weekday = checkpoint_date.weekday(),
                        func = identifier,
                        func_hash = func_hash,
                        args = all_args_repr,
                    ))
                    return deserialize_function(os.path.join(checkpoint_dir_path, file))

            checkpoint_filename = _PYCHECKPOINT_FILENAME_TEMPLATE.format(
                arg_repr=all_args_repr,
                arg_hash=arg_hash,
                date=date,
                ext=filename_extension,
            )
            checkpoint_file_path = os.path.join(checkpoint_dir_path, checkpoint_filename)
            tmp_checkpoint_filename = _PYCHECKPOINT_TMP_FILENAME_TEMPLATE.format(
                arg_repr=all_args_repr,
                arg_hash=arg_hash,
                date=date,
                ext=filename_extension,
            )
            tmp_checkpoint_file_path = os.path.join(checkpoint_dir_path, tmp_checkpoint_filename)
            result = func(*args, **kwargs)


            serialize_function(result, tmp_checkpoint_file_path)
            
            os.replace(tmp_checkpoint_file_path, checkpoint_file_path)

            return result
        return wrapper
    return decorator
