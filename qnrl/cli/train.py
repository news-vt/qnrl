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

from .. import tools
from ..config.optimizer import *
from ..config.gym import *
from ..config.checkpoint import *
from ..config.algorithm import *



# MARK: CLI

class DumpFormat(enum.StrEnum):
    JSON = enum.auto()
    YAML = enum.auto()
    PRETTY = enum.auto()

@dataclass(kw_only=True)
class Args(DataClassDictMixin):
    """Training CLI."""
    config: pathlib.Path # Path of config file in format [yaml, json, python].
    dump: DumpFormat | None = None


if __name__ == "__main__":
    
    args = tyro.cli(Args, config=(tyro.conf.EnumChoicesFromValues,))
    
    suffix = args.config.suffix
    match suffix:
        case ".yaml" | ".yml":
            with open(args.config) as f:
                data = yaml.safe_load(f) or {}
            # Load the configuration object.
            config = AlgorithmConfig.from_dict(data)
        # Load from JSON.
        case ".json":
            with open(args.config) as f:
                data = json.load(f) or {}
            # Load the configuration object.
            config = AlgorithmConfig.from_dict(data)
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
    if args.dump is not None:
        match args.dump:
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
    else:
        # Run training algorithm.
        config.run()
    
    ################
    
    # # Load config from YAML file.
    # default_config = None
    # argv = sys.argv[1:]
    # # Load from YAML.
    # if argv and argv[0].endswith((".yaml", ".yml",)):
    #     with open(argv[0]) as f:
    #         config = yaml.safe_load(f) or {}
    #     # Load the configuration object.
    #     default_config = Args.from_dict(config)
    #     argv = argv[1:]
    # # Load from JSON.
    # elif argv and argv[0].endswith((".json",)):
    #     with open(argv[0]) as f:
    #         config = json.load(f) or {}
    #     # Load the configuration object.
    #     default_config = Args.from_dict(config)
    #     argv = argv[1:]
    
    # # Instantiate the CLI with defaults loaded from YAML file.
    # # args = tyro.cli(Args, config=(tyro.conf.OmitSubcommandPrefixes, tyro.conf.EnumChoicesFromValues,), args=argv, default=default_config)
    # args = tyro.cli(Args, config=(tyro.conf.EnumChoicesFromValues,), args=argv, default=default_config)
    # # args = tyro.cli(Args, config=(tyro.conf.EnumChoicesFromValues,tyro.conf.ConsolidateSubcommandArgs,), args=argv, default=default_config)
    
    # # Dump configuration to various formats.
    # if args.dump is not None:
    #     match args.dump:
    #         case DumpFormat.YAML:
    #             data = args.to_dict()
    #             yaml_string = yaml.dump(data, sort_keys=False, default_flow_style=False, indent=2)
    #             print(yaml_string)
    #         case DumpFormat.JSON:
    #             data = args.to_dict()
    #             rich.print_json(data=data, indent=2)
    #         case DumpFormat.PRETTY:
    #             data = args.to_dict()
    #             rich.pretty.pprint(data, expand_all=True)
    # else:
    #     # Run training algorithm.
    #     args.config.run()