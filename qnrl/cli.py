from dataclasses import dataclass
import pathlib
from typing import Union
import json
import yaml
import json
import enum

import tyro
from mashumaro.mixins.dict import DataClassDictMixin
import rich
import rich.pretty

from . import tools
from .optimizer import *
from .gym import *
from .checkpoint import *
from .algorithms import *


import jax
jax.config.update("jax_enable_x64", True) # Enable double-precision for quantum models (default is single-precision).


# MARK: CLI

class DumpFormat(enum.StrEnum):
    JSON = enum.auto()
    YAML = enum.auto()
    PRETTY = enum.auto()


@dataclass(kw_only=True)
class Dump(DataClassDictMixin):
    """Dump the given configuration to various supported formats."""
    format: DumpFormat


@dataclass(kw_only=True)
class Run(DataClassDictMixin):
    """Run training algorithm using the given config."""
    pass


@dataclass(kw_only=True)
class Args(DataClassDictMixin):
    """Training CLI."""
    config: pathlib.Path # Path of config file in format [yaml, json, python].
    mode: Union[
        Dump,
        Run,
    ]


if __name__ == "__main__":
    
    args = tyro.cli(Args, config=(tyro.conf.EnumChoicesFromValues, tyro.conf.OmitSubcommandPrefixes,))
    
    suffix = args.config.suffix
    match suffix:
        case ".yaml" | ".yml":
            with open(args.config) as f:
                data = yaml.safe_load(f) or {}
            # Load the configuration object.
            config = algorithm.Algorithm.from_dict(data)
        # Load from JSON.
        case ".json":
            with open(args.config) as f:
                data = json.load(f) or {}
            # Load the configuration object.
            config = algorithm.Algorithm.from_dict(data)
        # Load from Python.
        case ".py":
            mod = tools.load_python_module_from_filepath(args.config)
            try:
                config = mod.config # The configuration must be saved in a variable called `config`.
            except:
                raise ValueError(f"could not load config from {args.config}")
        case _:
            raise ValueError(f"{suffix} is an unsupported file extension")
    
    # Dump configuration to various formats.
    if isinstance(args.mode, Dump):
        match args.mode.format:
            case DumpFormat.YAML:
                data = config.to_dict()
                yaml_string = yaml.dump(data, sort_keys=False, default_flow_style=False, indent=2)
                print(yaml_string)
            case DumpFormat.JSON:
                data = config.to_dict()
                json_string = json.dumps(data, indent=2, cls=tools.make_combined_json_encoder(tools.NumpyJSONEncoder, tools.JAXJSONEncoder))
                rich.print_json(json_string)
            case DumpFormat.PRETTY:
                data = config.to_dict()
                rich.pretty.pprint(data, expand_all=True)

    # Run training algorithm.
    elif isinstance(args.mode, Run):
        config.run()