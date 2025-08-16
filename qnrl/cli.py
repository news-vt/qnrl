from dataclasses import dataclass
import pathlib
from typing import Annotated, Literal, Union
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
class DumpAlgorithm(DataClassDictMixin):
    """Dump the given configuration to various supported formats."""
    format: DumpFormat
    
    def main(self, config: algorithm.Algorithm):
        match self.format:
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


@dataclass(kw_only=True)
class DumpMetrics(DataClassDictMixin):
    """Dump the given configuration to various supported formats."""
    format: DumpFormat
    type: Literal['list', 'dict'] = 'dict' # Dump as either time-sorted history list, or metrics dictionary.
    
    def main(self, config: algorithm.Algorithm):
        if config.metric_manager is not None:
            # Load metrics.
            metric_manager = config.metric_manager
            metric_manager.load()

            match self.type:
                case 'list':
                    data = metric_manager.get_history()
                case 'dict':
                    data = metric_manager.get_metrics()
                case _:
                    raise ValueError(f"{self.type} is not a supported metric dump type")

            match self.format:
                case DumpFormat.YAML:
                    yaml_string = yaml.dump(data, sort_keys=False, default_flow_style=False, indent=2)
                    print(yaml_string)
                case DumpFormat.JSON:
                    json_string = json.dumps(data, indent=2, cls=tools.make_combined_json_encoder(tools.NumpyJSONEncoder, tools.JAXJSONEncoder))
                    rich.print_json(json_string)
                case DumpFormat.PRETTY:
                    rich.pretty.pprint(data, expand_all=True)
        else:
            raise ValueError("no metric manager provided")


@dataclass(kw_only=True)
class Run(DataClassDictMixin):
    """Run training algorithm using the given config."""
    
    def main(self, config: algorithm.Algorithm):
        config.run() # Run training algorithm.


@dataclass(kw_only=True)
class Plot(DataClassDictMixin):
    def main(self, config: algorithm.Algorithm):
        if config.metric_manager is not None:
            # Load metrics.
            metric_manager = config.metric_manager
            metric_manager.load()
            
            import plotext as plt
            metrics = metric_manager.get_metrics()
            
            # plt.figure()
            plt.plot(metrics['loss_step'], metrics['loss'])
            plt.xlabel('loss_step')
            plt.ylabel('loss')
            plt.show()

            # plt.figure()
            plt.clf()
            plt.plot(metrics['episode_step'], metrics['episode_reward'])
            plt.xlabel('episode_step')
            plt.ylabel('episode_reward')
            plt.show()
        else:
            raise ValueError("no metric manager provided")


@dataclass(kw_only=True)
class Metrics(DataClassDictMixin):
    mode: Union[
        Annotated[DumpMetrics, tyro.conf.subcommand(name='dump')],
        Plot,
    ]
    
    def main(self, config: algorithm.Algorithm):
        # Dump metrics to standard output.
        if isinstance(self.mode, DumpMetrics):
            self.mode.main(config)
        
        # Plot metric results.
        elif isinstance(self.mode, Plot):
            self.mode.main(config)


@dataclass(kw_only=True)
class Args(DataClassDictMixin):
    """Training CLI."""
    config: pathlib.Path # Path of config file in format [yaml, json, python].
    mode: Union[
        Annotated[DumpAlgorithm, tyro.conf.subcommand(name='dump')],
        Run,
        Metrics,
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
    
    # Run program mode.
    args.mode.main(config)