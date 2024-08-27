# pycheckpoint

```
def pycheckpoint(
        checkpoint_path: Union[str, os.PathLike]=".pycheckpoint",
        serialization: Union[Literal["pickle", "pandas.csv", "pandas.parquet", "json"], Tuple[Callable, Callable, str]]="pickle",
        serialize_function_kwargs: Tuple[dict, dict]=({}, {}),
        canonical_args: bool=True,
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

    """
```
