from dataclasses import dataclass, fields
import pathlib
from typing import Annotated, Literal, Union
import json
import yaml
import json
import enum
import os
import time
import sys

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


def load_algo_config(path: pathlib.Path) -> algorithm.Algorithm:
    """Loads algorithm config from path, or throws error."""
    # Verify config file exists.
    if not path.exists():
        raise ValueError(f"config file does not exist")
    
    suffix = path.suffix
    match suffix:
        case ".yaml" | ".yml":
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            # Load the configuration object.
            return algorithm.Algorithm.from_dict(data)
        # Load from JSON.
        case ".json":
            with open(path) as f:
                data = json.load(f) or {}
            # Load the configuration object.
            return algorithm.Algorithm.from_dict(data)
        # Load from Python.
        case ".py":
            try:
                mod = tools.load_python_module_from_filepath(path)
                return mod.config # The configuration must be saved in a variable called `config`.
            except Exception as e:
                raise ValueError(f"could not load config from {path}: {e}")
        case _:
            raise ValueError(f"{suffix} is an unsupported file extension")



# MARK: CLI

class DumpFormat(enum.StrEnum):
    JSON = enum.auto()
    YAML = enum.auto()
    PRETTY = enum.auto()


@dataclass(kw_only=True)
class InfoModel(DataClassDictMixin):
    """Dump the model parameter information specified in the given algorithm.
    
    Extract model names and total number of parameters:
    >>> cli info-model --config <file> --format json | jq 'map({name: .name, total: .parameters.total})'
    
    Extract model names and raw parameters in PyTree dictionary format:
    >>> cli info-model --config <file> --include-raw-parameters --format json | jq 'map({name: .name, param: .parameters.param})'
    """
    config: pathlib.Path # Path of config file in format [yaml, json, python].
    format: DumpFormat
    include_raw_parameters: tyro.conf.FlagCreatePairsOff[bool] = False
    """Output model parameters, but WARNING this can be very slow and verbose."""
    
    def main(self):
        # Load config.
        config = load_algo_config(path=self.config)
        
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
    >>> cli dump --config <file> --format pretty
    
    Dump configuration to JSON file:
    >>> cli dump --config <file> --format json | <outfile>.json
    
    Dump configuration to YAML file:
    >>> cli dump --config <file> --format yaml | <outfile>.yaml
    
    Extract metric file path:
    >>> cli dump --config <file> --format json | jq '.metric_manager.path'
    """
    config: pathlib.Path # Path of config file in format [yaml, json, python].
    format: DumpFormat
    
    def main(self):
        # Load config.
        config = load_algo_config(path=self.config)
        
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
class Train(DataClassDictMixin):
    """Run training algorithm using the given config."""
    config: pathlib.Path # Path of config file in format [yaml, json, python].
    
    def main(self):
        # Load config.
        config = load_algo_config(path=self.config)
        
        # Run training algorithm.
        config.train()


@dataclass(kw_only=True)
class Eval(DataClassDictMixin):
    """Run evaluation algorithm using the given config."""
    config: pathlib.Path # Path of config file in format [yaml, json, python].
    format: DumpFormat = DumpFormat.PRETTY
    kwargs: dict[str, bool | int | float | str] = field(default_factory=dict)
    
    def main(self):
        # Load config.
        config = load_algo_config(path=self.config)
        
        history_list = config.eval(**self.kwargs) # Run training algorithm.
        # Convert to dictionary of lists.
        data = tools.listofdict_to_dictoflist(history_list)
        
        match self.format:
            case DumpFormat.YAML:
                yaml_string = yaml.dump(data, sort_keys=False, default_flow_style=False, indent=2)
                print(yaml_string)
            case DumpFormat.JSON:
                json_string = json.dumps(data, indent=2, cls=tools.make_combined_json_encoder(tools.NumpyJSONEncoder, tools.JAXJSONEncoder))
                rich.print_json(json_string)
            case DumpFormat.PRETTY:
                rich.pretty.pprint(data, expand_all=True)


# TODO add live update https://github.com/piccolomo/plotext/blob/master/readme/environments.md#rich
@dataclass(kw_only=True)
class Plot(DataClassDictMixin):
    """Plot metric data using X/Y data keys.
    
    If you are unsure of what keys exist, see `InfoMetrics`.
    
    Example plot X-axis "loss_step" and Y-axis "loss" metric keys using "clear" style in terminal:
    >>> cli plot --config <file> --x loss_step --y loss --type terminal --theme clear
    """
    config: list[pathlib.Path] # Path of config file in format [yaml, json, python].
    type: Literal['terminal', 'matplotlib'] = 'terminal'
    x: str # X-axis data key.
    y: str # Y-axis data key.
    x_min: float | None = None # X-axis range min.
    x_max: float | None = None # X-axis range max.
    y_min: float | None = None # Y-axis range min.
    y_max: float | None = None # Y-axis range max.
    show_last_n: int | None = None # Show the last `N` data points
    theme: str | None = None # Plotext "theme", or Matplotlib "style".
    color: list[str] | None = None # Plot line color.
    sync: bool = False # Synchronize X and Y axis data ranges to what is available across all configs.
    
    def main(self):
        
        # Load config.
        configs = {cfg_path.stem: load_algo_config(path=cfg_path) for cfg_path in self.config}
        colors = {key: color for color, key in zip(self.color or [None]*len(self.config), configs.keys())}
        
        # Ensure all configs have metrics.
        if any(cfg.metric_manager is None for _,cfg in configs.items()):
            raise ValueError("no metric manager provided for one of the metrics")
        
        # X/Y data.
        data_dict = {
            'x': {},
            'y': {},
        }
        
        for key, cfg in configs.items():
            # Load metrics.
            metric_manager = cfg.metric_manager
            metric_manager.load()
            metrics = metric_manager.get_metrics()
            
            if self.x not in metrics:
                print(f"ERROR:X-axis key {self.x} not found in metrics file", file=sys.stderr)
                return
            if self.y not in metrics:
                print(f"ERROR:Y-axis key {self.y} not found in metrics file", file=sys.stderr)
                return
            
            x_data = np.array(metrics[self.x])
            y_data = np.array(metrics[self.y])
            
            # Clip X-axis data.
            if self.x_min is not None:
                idx = x_data >= self.x_min
                x_data = x_data[idx]
                y_data = y_data[idx]
            if self.x_max is not None:
                idx = x_data <= self.x_max
                x_data = x_data[idx]
                y_data = y_data[idx]
            
            # Clip Y-axis data.
            if self.y_min is not None:
                idx = y_data >= self.y_min
                x_data = x_data[idx]
                y_data = y_data[idx]
            if self.y_max is not None:
                idx = y_data <= self.y_max
                x_data = x_data[idx]
                y_data = y_data[idx]
            
            # Select the last `N` data points.
            if self.show_last_n is not None:
                x_data = x_data[-self.show_last_n:]
                y_data = y_data[-self.show_last_n:]
                
            # Preserve data.
            data_dict['x'][key] = x_data
            data_dict['y'][key] = y_data
        
        # Synchronize X/Y axis data display to data available across all configs.
        if self.sync:
            self.x_min = max(x_data.min() for _, x_data in data_dict['x'].items())
            self.x_max = min(x_data.max() for _, x_data in data_dict['x'].items())
            # Clip X-axis data for all configs.
            for key in configs.keys():
                # Clip X min.
                idx = data_dict['x'][key] >= self.x_min
                data_dict['x'][key] = data_dict['x'][key][idx]
                data_dict['y'][key] = data_dict['y'][key][idx]
                # Clip X max.
                idx = data_dict['x'][key] <= self.x_max
                data_dict['x'][key] = data_dict['x'][key][idx]
                data_dict['y'][key] = data_dict['y'][key][idx]
        
        # Plot.
        match self.type:
            case 'terminal':
                import plotext as plt
                
                plt.theme(theme=self.theme or "pro")
                
                # Plot data in each file.
                for key in configs.keys():
                    plt.plot(data_dict['x'][key], data_dict['y'][key], color=colors[key], label=key)
                
                # Set axis labels and title.
                plt.xlabel(self.x)
                plt.ylabel(self.y)
                plt.title(f"{self.y} vs. {self.x}")
                
                # Set axis limits.
                plt.xlim(left=self.x_min, right=self.x_max)
                plt.ylim(lower=self.y_min, upper=self.y_max)
                
                # Show the plot.
                plt.show()

            case 'matplotlib':
                import matplotlib.pyplot as plt
                
                if self.theme is not None:
                    plt.style.use(style=self.theme)
                
                # Plot data in each file.
                for key in configs.keys():
                    plt.plot(data_dict['x'][key], data_dict['y'][key], color=colors[key], label=key)
                
                # Set axis labels and title.
                plt.xlabel(self.x)
                plt.ylabel(self.y)
                plt.title(f"{self.y} vs. {self.x}")
                
                # Set axis limits.
                plt.xlim(left=self.x_min, right=self.x_max)
                plt.ylim(bottom=self.y_min, top=self.y_max)
                
                # Show the plot.
                plt.show()


@dataclass(kw_only=True)
class PlotLive(DataClassDictMixin):
    """Live plot metric data using X/Y data keys.
    
    If you are unsure of what keys exist, see `InfoMetrics`.
    
    Example live plot X-axis "loss_step" and Y-axis "loss" metric keys using "clear" style in terminal with an update inveral of "1" seconds and only show the last 200 data points:
    >>> cli plot-live --config <file> --x loss_step --y loss --show-last-n 200 --sleep 1 --theme clear
    """
    config: pathlib.Path # Path of config file in format [yaml, json, python].
    x: str # X-axis data key.
    y: str # Y-axis data key.
    sleep: float = 1.0 # Sleep interval in seconds.
    x_min: float | None = None # X-axis range min.
    x_max: float | None = None # X-axis range max.
    y_min: float | None = None # Y-axis range min.
    y_max: float | None = None # Y-axis range max.
    show_last_n: int | None = None # Show the last `N` data points.
    theme: str | None = "pro" # Plotext "theme"
    color: str | None = None # Plot line color.
    
    def main(self):
        # Load config.
        config = load_algo_config(path=self.config)
        
        if config.metric_manager is not None:
            metric_manager = config.metric_manager
            
            import plotext as plt
            
            plt.themes
            
            plt.theme(theme=self.theme)
            
            try:
                while True:
                    
                    # Load new data.
                    metric_manager.load()
                    metrics = metric_manager.get_metrics()
                    
                    if self.x not in metrics:
                        print(f"ERROR:X-axis key {self.x} not found in metrics file", file=sys.stderr)
                        return
                    if self.y not in metrics:
                        print(f"ERROR:Y-axis key {self.y} not found in metrics file", file=sys.stderr)
                        return
                    
                    x_data = np.array(metrics[self.x])
                    y_data = np.array(metrics[self.y])
                    
                    # Clip X-axis data.
                    if self.x_min is not None:
                        idx = x_data >= self.x_min
                        x_data = x_data[idx]
                        y_data = y_data[idx]
                    if self.x_max is not None:
                        idx = x_data <= self.x_max
                        x_data = x_data[idx]
                        y_data = y_data[idx]
                    
                    # Clip Y-axis data.
                    if self.y_min is not None:
                        idx = y_data >= self.y_min
                        x_data = x_data[idx]
                        y_data = y_data[idx]
                    if self.y_max is not None:
                        idx = y_data <= self.y_max
                        x_data = x_data[idx]
                        y_data = y_data[idx]
                    
                    # Select the last `N` data points.
                    if self.show_last_n is not None:
                        x_data = x_data[-self.show_last_n:]
                        y_data = y_data[-self.show_last_n:]
                    
                    plt.clt() # Clear terminal.
                    plt.cld() # Clear data only.
                    w, h = plt.terminal_size()
                    plt.plot_size(w, h-1) # Automatically resize (use -1 to include space for title).
                    plt.plot(x_data, y_data, color=self.color)
                    plt.xlabel(self.x)
                    plt.ylabel(self.y)
                    plt.title(config.metric_manager.metric_dir.stem)
                    plt.show()
                    
                    # Blocking wait for next file change.
                    if not tools.wait_for_file_change(config.metric_manager.metric_dir, interval=self.sleep):
                        return
                    
            except KeyboardInterrupt:
                return
        else:
            raise ValueError("no metric manager provided")


@dataclass(kw_only=True)
class DumpMetrics(DataClassDictMixin):
    """Dump the given configuration to various supported formats.
    
    Extract all metric keys:
    >>> cli metrics dump --config <file> --type dict --format json | jq 'keys'
    """
    config: pathlib.Path # Path of config file in format [yaml, json, python].
    format: DumpFormat
    type: Literal['list', 'dict'] = 'dict' # Dump as either time-sorted history list, or metrics dictionary.
    
    def main(self):
        # Load config.
        config = load_algo_config(path=self.config)
        
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
class InfoMetrics(DataClassDictMixin):
    """Get metric information.
    
    This is different than dumping the entire metrics file. See `DumpMetrics` for dumping entire metric file contents.
    
    Extract all metric keys:
    >>> cli metrics info --config <file> --format json | jq '.[].key'
    
    Extract keys `{key, min, argmin}` from metrics:
    >>> cli metrics info --config <file> --format json | jq 'map({key, min, argmin})'
    
    Extract "loss" and "episode_reward" keys from metrics:
    >>> cli metrics info --config <file> --format json | jq '.[] | select(.key=="loss" or .key=="episode_reward")'
    """
    config: list[pathlib.Path] # Path of config file in format [yaml, json, python].
    format: DumpFormat = DumpFormat.PRETTY

    def main(self):
        # Load config.
        configs = {cfg_path.stem: load_algo_config(path=cfg_path) for cfg_path in self.config}
        
        # Ensure all configs have metrics.
        if any(cfg.metric_manager is None for _,cfg in configs.items()):
            raise ValueError("no metric manager provided for one of the metrics")
        
        # Collect metrics for all given files.
        data = []
        for cfg_key, cfg in configs.items():
            # Load metrics.
            metric_manager = cfg.metric_manager
            metric_manager.load()
            metrics = metric_manager.get_metrics()
            
            # Collect metric information.
            d = {
                'key': cfg_key,
                'metrics': [m.to_dict() for m in tools.compute_train_result_statistics(metrics)], # Serialize.
            }
            data.append(d)
            
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


@dataclass(kw_only=True)
class Metrics(DataClassDictMixin):
    """Interact with metrics file."""
    mode: Union[
        Annotated[DumpMetrics, tyro.conf.subcommand(name='dump')],
        Annotated[InfoMetrics, tyro.conf.subcommand(name='info')],
    ]
    
    def main(self):
        self.mode.main()


@dataclass(kw_only=True)
class Args(DataClassDictMixin):
    """Algorithm CLI."""
    # config: pathlib.Path # Path of config file in format [yaml, json, python].
    mode: Union[
        Annotated[DumpAlgorithm, tyro.conf.subcommand(name='dump')],
        Annotated[InfoModel, tyro.conf.subcommand(name='info-model')],
        Train,
        Eval,
        Metrics,
        Plot,
        PlotLive,
    ]


if __name__ == "__main__":
    
    args = tyro.cli(Args, config=(tyro.conf.EnumChoicesFromValues, tyro.conf.OmitSubcommandPrefixes,))
    
    # Run program mode.
    args.mode.main()