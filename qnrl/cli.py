from dataclasses import dataclass, fields
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
import numpy as np
import jax.numpy as jnp

from . import tools
from .optimizer import *
from .gym import *
from .checkpoint import *
from .algorithms import *
from .modules import SerializableModule


import jax
jax.config.update("jax_enable_x64", True) # Enable double-precision for quantum models (default is single-precision).


# MARK: CLI

class DumpFormat(enum.StrEnum):
    JSON = enum.auto()
    YAML = enum.auto()
    PRETTY = enum.auto()


@dataclass(kw_only=True)
class InfoModel(DataClassDictMixin):
    """Dump the model parameter information specified in the given algorithm.
    
    Extract model names and total number of parameters:
    >>> cli --config <file> info-model --format json | jq 'map({name: .name, total: .parameters.total})'
    
    Extract model names and raw parameters in PyTree dictionary format:
    >>> cli --config <file> info-model --include-raw-parameters --format json | jq 'map({name: .name, param: .parameters.param})'
    """
    format: DumpFormat
    include_raw_parameters: tyro.conf.FlagCreatePairsOff[bool] = False
    """Output model parameters, but WARNING this can be very slow and verbose."""
    
    def main(self, config: algorithm.Algorithm):
        data = []
        for field in fields(config):
            if field.type is SerializableModule:
                # Get the model object.
                model = getattr(config, field.name)
                # Inspect the model.
                d = tools.inspect_flax_nnx_model(model=model, include_raw_parameters=self.include_raw_parameters)
                # Add model name as field.
                d['name'] = field.name
                # Add inspection object to summary report.
                data.append(d)
        
        match self.format:
            case DumpFormat.YAML:
                yaml_string = yaml.dump(data, sort_keys=False, default_flow_style=False, indent=2)
                print(yaml_string)
            case DumpFormat.JSON:
                json_string = json.dumps(data, indent=2, cls=tools.make_combined_json_encoder(tools.NumpyJSONEncoder, tools.JAXJSONEncoder))
                rich.print_json(json_string)
            case DumpFormat.PRETTY:
                rich.pretty.pprint(data, expand_all=True)


@dataclass(kw_only=True)
class DumpAlgorithm(DataClassDictMixin):
    """Dump the given configuration to various supported formats.
    
    Pretty-print configuration:
    >>> cli --config <file> dump --format pretty
    
    Dump configuration to JSON file:
    >>> cli --config <file> dump --format json | <outfile>.json
    
    Dump configuration to YAML file:
    >>> cli --config <file> dump --format yaml | <outfile>.yaml
    
    Extract metric file path:
    >>> cli --config <file> dump --format json | jq '.metric_manager.path'
    """
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
    """Dump the given configuration to various supported formats.
    
    Extract all metric keys:
    >>> cli --config <file> metrics dump --type dict --format json | jq 'keys'
    """
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

# TODO add live update https://github.com/piccolomo/plotext/blob/master/readme/environments.md#rich
@dataclass(kw_only=True)
class Plot(DataClassDictMixin):
    """Plot metric data using X/Y data keys.
    
    If you are unsure of what keys exist, see `InfoMetrics`.
    """
    type: Literal['terminal', 'matplotlib'] = 'terminal'
    x: str # X-axis data key.
    y: str # Y-axis data key.
    x_min: float | None = None # X-axis range min.
    x_max: float | None = None # X-axis range max.
    y_min: float | None = None # Y-axis range min.
    y_max: float | None = None # Y-axis range max.
    clip_x_min: float | None = None # Clip min X-axis data.
    clip_x_max: float | None = None # Clip max X-axis data.
    clip_y_min: float | None = None # Clip min Y-axis data.
    clip_y_max: float | None = None # Clip max Y-axis data.
    
    def main(self, config: algorithm.Algorithm):
        if config.metric_manager is not None:
            # Load metrics.
            metric_manager = config.metric_manager
            metric_manager.load()
            metrics = metric_manager.get_metrics()
            
            x_data = np.array(metrics[self.x])
            y_data = np.array(metrics[self.y])
            
            # Clip X-axis data.
            if self.clip_x_min is not None:
                idx = x_data >= self.clip_x_min
                x_data = x_data[idx]
                y_data = y_data[idx]
            if self.clip_x_max is not None:
                idx = x_data <= self.clip_x_max
                x_data = x_data[idx]
                y_data = y_data[idx]
            
            # Clip Y-axis data.
            if self.clip_y_min is not None:
                idx = y_data >= self.clip_y_min
                x_data = x_data[idx]
                y_data = y_data[idx]
            if self.clip_y_max is not None:
                idx = y_data <= self.clip_y_max
                x_data = x_data[idx]
                y_data = y_data[idx]
            
            match self.type:
                case 'terminal':
                    import plotext as plt
                    
                    plt.plot(x_data, y_data)
                    plt.xlabel(self.x)
                    plt.ylabel(self.y)
                    
                    plt.xlim(left=self.x_min, right=self.x_max)
                    plt.ylim(lower=self.y_min, upper=self.y_max)
                    
                    plt.show()

                case 'matplotlib':
                    import matplotlib.pyplot as plt
                    
                    plt.plot(x_data, y_data)
                    plt.xlabel(self.x)
                    plt.ylabel(self.y)
                    
                    plt.xlim(left=self.x_min, right=self.x_max)
                    plt.ylim(bottom=self.y_min, top=self.y_max)
                    
                    plt.show()
        else:
            raise ValueError("no metric manager provided")


@dataclass(kw_only=True)
class InfoMetrics(DataClassDictMixin):
    """Get metric information.
    
    This is different than dumping the entire metrics file. See `DumpMetrics` for dumping entire metric file contents.
    
    Extract all metric keys:
    >>> cli --config <file> metrics info --format json | jq '.[].key'
    
    Extract keys `{key, min, argmin}` from metrics:
    >>> cli --config <file> metrics info --format json | jq 'map({key, min, argmin})'
    """
    format: DumpFormat = DumpFormat.PRETTY

    def main(self, config: algorithm.Algorithm):
        if config.metric_manager is not None:
            # Load metrics.
            metric_manager = config.metric_manager
            metric_manager.load()
            metrics = metric_manager.get_metrics()
            
            # Collect metric information.
            data = [dict(
                key=key,
                shape=list(jnp.array(val).shape),
                dtype=str(jnp.array(val).dtype),
                min=jnp.array(val).min().item(),
                max=jnp.array(val).max().item(),
                argmin=jnp.array(val).argmin().item(),
                argmax=jnp.array(val).argmax().item(),
                mean=jnp.array(val).mean().item(),
                std=jnp.array(val).std().item(),
            ) for key, val in metrics.items()]
            
            # Dump.
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
class Metrics(DataClassDictMixin):
    """Interact with metrics file."""
    mode: Union[
        Annotated[DumpMetrics, tyro.conf.subcommand(name='dump')],
        Annotated[InfoMetrics, tyro.conf.subcommand(name='info')],
        Plot,
    ]
    
    def main(self, config: algorithm.Algorithm):
        self.mode.main(config)


@dataclass(kw_only=True)
class Args(DataClassDictMixin):
    """Algorithm CLI."""
    config: pathlib.Path # Path of config file in format [yaml, json, python].
    mode: Union[
        Annotated[DumpAlgorithm, tyro.conf.subcommand(name='dump')],
        Annotated[InfoModel, tyro.conf.subcommand(name='info-model')],
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