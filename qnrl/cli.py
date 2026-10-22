from dataclasses import dataclass, fields
import pathlib
from typing import Annotated, Literal, Union
import json
import yaml
import enum
import os
import time
import sys
import datetime
import tempfile

import tyro
from mashumaro.mixins.dict import DataClassDictMixin
import rich
import rich.pretty
import numpy as np
import jax.numpy as jnp
from matplotlib.figure import Figure as MatplotlibFigure
from matplotlib.animation import FuncAnimation as MatplotlibFuncAnimation
from plotext._figure import default_figure_class as PlotextFigure
import wandb
import pandas as pd
import moviepy

from . import tools
from .store import HashStore
from .optimizer import *
from .environment import *
from .checkpoint import *
from .algorithms import *
from .modules import SerializableModule
from .metric import MetricManager, LocalFileMetricManager, WandBMetricManager
from .serialization import hash_serializable_object

# Register gymnasium environments.
import gymnasium as gym
# Register ALE environments.
import ale_py
gym.register_envs(ale_py)


import jax
jax.config.update("jax_enable_x64", True) # Enable double-precision for quantum models (default is single-precision).


def load_algo_config(
    key: str | None = None,
    store: HashStore | None = None,
    path: pathlib.Path | None = None,
    overrides: dict[str, Any] | None = None,
    ) -> algorithm.Algorithm:
    """Loads algorithm config from path or key + store, or throws error.
    
    Python configurations allow for the most flexibility. There must be a variable called `config` at global scope in the module that is a configuration object. There can also be a `factory()` function with keyword-only arguments that builds the configuration object dynamically from a set of parameters. To load from a factory function the user must provide arguments via the `overrides` dictionary.
    """
    
    # Load from hash store.
    if key is not None and store is not None:
        try:
            data = store.decode(store[key])
            config = algorithm.Algorithm.from_dict(data)
        except Exception as e:
            raise ValueError(f"could not load config from {store}: {e}")
    
    elif path is not None:
        # Verify config file exists.
        if not path.exists():
            raise ValueError(f"config file does not exist")
        
        suffix = path.suffix
        match suffix:
            case ".yaml" | ".yml":
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                # Load the configuration object.
                config = algorithm.Algorithm.from_dict(data)
            # Load from JSON.
            case ".json":
                with open(path) as f:
                    data = json.load(f) or {}
                # Load the configuration object.
                config = algorithm.Algorithm.from_dict(data)
            # Load from Python.
            case ".py":
                try:
                    # Load configuration module.
                    mod = tools.load_python_module_from_filepath(path)
                    
                    # Load config dynamically from factory if one is provided.
                    if overrides is not None and hasattr(mod, 'factory'):
                        config = mod.factory(**overrides)
                    # Load config object directly.
                    else:
                        config = mod.config # The configuration must be saved in a variable called `config`.
                except Exception as e:
                    raise ValueError(f"could not load config from {path}: {e}")
            case _:
                raise ValueError(f"{suffix} is an unsupported file extension")
    else:
        raise ValueError("must provide either path or key + store.")
    
    # Perform overrides on the configuration object.
    if overrides is not None:
        for k, v in overrides.items():
            if hasattr(config, k):
                config.__setattr__(k, v)

    # Return the configuration object.
    return config



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


# TODO add live update https://github.com/piccolomo/plotext/blob/master/readme/environments.md#rich
@dataclass(kw_only=True)
class Plot(DataClassDictMixin):
    """Plot metric data using X/Y data keys.
    
    If you are unsure of what keys exist, see `InfoMetrics`.
    
    Example plot X-axis "loss_step" and Y-axis "loss" metric keys using "clear" style in terminal:
    >>> cli plot --config <file> --x loss_step --y loss --type terminal --theme clear
    """
    config: list[pathlib.Path] | None = None # Path of config file in format [yaml, json, python].
    history: list[pathlib.Path] | None = None # Direct path to history file of metrics to plot.
    # type: Literal['terminal', 'matplotlib'] = 'terminal'
    x: str # X-axis data key.
    y: str # Y-axis data key.
    x_min: float | None = None # X-axis range min.
    x_max: float | None = None # X-axis range max.
    y_min: float | None = None # Y-axis range min.
    y_max: float | None = None # Y-axis range max.
    show_last_n: int | None = None # Show the last `N` data points
    sync: bool = False # Synchronize X and Y axis data ranges to what is available across all configs.
    stats: bool = False # Report min/max/mean statistics in plot label.
    plot_type: Literal['line', 'scatter'] = 'line' # Plot type as line or scatter plot.
    
    def __call__(self):
        return self.main()
    
    def __post_init__(self):
        # Ensure that either config or history file list was given.
        assert self.config or self.history, "must provide either list of configurations or direct list of history files"
        
        ### Create a map from unique keys to data source.
        self.source_map: dict[str, pathlib.Path | algorithm.Algorithm | MetricManager] = {}
        #
        # Data source is a config object.
        if self.config:
            cfgs = {cfg_path.stem: load_algo_config(path=cfg_path) for cfg_path in self.config}
            
            # Ensure all configs have metrics.
            if any(cfg.metric_manager is None for _,cfg in cfgs.items()):
                raise ValueError("no metric manager provided for one of the metrics")
            
            self.source_map.update(cfgs)
        # Data source is a direct history file.
        if self.history:
            for path in self.history:
                if path.is_dir():
                    metric_manager = LocalFileMetricManager(
                        metric_dir=path.resolve(),
                        metric_file_type="json", # TODO assume JSON for now, but this should be inferred from the directory somehow.
                        )
                    self.source_map.update({path.stem: metric_manager})
                elif path.is_file():
                    self.source_map.update({path.stem: path})
                else:
                    raise ValueError(f"Unsupported history source {path} (must be either directory, regular file, or algorithm config)")
    
    
    def main(self):
        """Used for CLI."""
        pass
    
    def prepare_data(self, source_map: dict[str, pathlib.Path | algorithm.Algorithm]):
        
        # X/Y data.
        data_dict = {
            'x': {},
            'y': {},
        }
        
        for key, src in source_map.items():
            # Load metrics from config object.
            if isinstance(src, algorithm.Algorithm):
                metric_manager = src.metric_manager # It cannot be None here because we check in main.
                metric_manager.load()
                metrics = metric_manager.get_metrics()
            # Load metrics directly from metric manager.
            elif isinstance(src, MetricManager):
                metric_manager = src
                metric_manager.load()
                metrics = metric_manager.get_metrics()
            # Load metrics from history file.
            elif isinstance(src, pathlib.Path):
                history_list = tools.load_train_results(src)
                metrics = tools.listofdict_to_dictoflist(history_list)
            else:
                raise ValueError(f"Unsupported history source {src} (must be either directory, regular file, or algorithm config)")
            
            # Ensure desired keys are present.
            if self.x not in metrics:
                raise ValueError(f"ERROR:X-axis key {self.x} not found in metrics file")
            if self.y not in metrics:
                raise ValueError(f"ERROR:Y-axis key {self.y} not found in metrics file")
            
            # Preserve data.
            data_dict['x'][key] = np.array(metrics[self.x])
            data_dict['y'][key] = np.array(metrics[self.y])
        
        # Synchronize X/Y axis data display to data available across all configs.
        if self.sync:
            self.x_min = max(x_data.min() for _, x_data in data_dict['x'].items())
            self.x_max = min(x_data.max() for _, x_data in data_dict['x'].items())
            # Clip X-axis data for all configs.
            for key in source_map.keys():
                # Clip X min.
                idx = data_dict['x'][key] >= self.x_min
                data_dict['x'][key] = data_dict['x'][key][idx]
                data_dict['y'][key] = data_dict['y'][key][idx]
                # Clip X max.
                idx = data_dict['x'][key] <= self.x_max
                data_dict['x'][key] = data_dict['x'][key][idx]
                data_dict['y'][key] = data_dict['y'][key][idx]
        
        # Apply any clipping on data after sync.
        for key in source_map.keys():
            # Clip X-axis data.
            if self.x_min is not None:
                idx = data_dict['x'][key] >= self.x_min
                data_dict['x'][key] = data_dict['x'][key][idx]
                data_dict['y'][key] = data_dict['y'][key][idx]
            if self.x_max is not None:
                idx = data_dict['x'][key] <= self.x_max
                data_dict['x'][key] = data_dict['x'][key][idx]
                data_dict['y'][key] = data_dict['y'][key][idx]
            
            # Clip Y-axis data.
            if self.y_min is not None:
                idx = data_dict['y'][key] >= self.y_min
                data_dict['x'][key] = data_dict['x'][key][idx]
                data_dict['y'][key] = data_dict['y'][key][idx]
            if self.y_max is not None:
                idx = data_dict['y'][key] <= self.y_max
                data_dict['x'][key] = data_dict['x'][key][idx]
                data_dict['y'][key] = data_dict['y'][key][idx]
            
            # Select the last `N` data points.
            if self.show_last_n is not None:
                data_dict['x'][key] = data_dict['x'][key][-self.show_last_n:]
                data_dict['y'][key] = data_dict['y'][key][-self.show_last_n:]

        return data_dict

@dataclass(kw_only=True)
class PlotMatplotlib(Plot):
    figure_kwargs: dict[str, int | float | list[int] | list[float] | str | list[str]] = field(default_factory=dict) # Figure keyword arguments for `type='matplotlib'`.
    legend_kwargs: dict[str, int | float | list[int] | list[float] | str | list[str]] = field(default_factory=dict) # Legend keyword arguments for `type='matplotlib'`.
    style: str | None = None # Matplotlib "style".
    color: list[str] | None = None # Plot line color.
    show: bool = True # Show the figure.
    zorder: dict[str, int] | None = None # Plot line Z-axis ordering.
    alpha: float | None = None # Plot line opacity (1 is solid, 0 is transparent).
    
    def __call__(self):
        return self.main()

    def main(self) -> MatplotlibFigure:
        
        # Get data.
        data_dict = self.prepare_data(source_map=self.source_map)
        
        colors = {key: color for color, key in zip(self.color or [None]*len(self.source_map), self.source_map.keys())}
        
        if self.zorder is None:
            self.zorder = {key: i for i, key in enumerate(self.source_map)}
        if len(self.zorder) != len(self.source_map):
            raise ValueError(f"Length of zorder must match number of configurations.")
        
        # Plot.
        import matplotlib.pyplot as plt
        
        if self.style is not None:
            plt.style.use(style=self.style)
        
        # Set plot function.
        match self.plot_type:
            case 'line':
                plot_fn = plt.plot
            case 'scatter':
                plot_fn = plt.scatter
            case _:
                raise ValueError(f"Unsupported plot type {self.plot_type}")
        
        # Create figure.
        fig = plt.figure(**self.figure_kwargs)
        
        # Plot data in each file.
        for i, key in enumerate(self.source_map.keys()):
            if self.stats:
                label = f"{key} (min {data_dict['y'][key].min():.4f}, max {data_dict['y'][key].max():.4f}, mean {data_dict['y'][key].mean():.4f})"
            else:
                label = key
            plot_fn(data_dict['x'][key], data_dict['y'][key], color=colors[key], label=label, zorder=self.zorder[key], alpha=self.alpha)
        
        # Set axis labels and title.
        plt.xlabel(self.x)
        plt.ylabel(self.y)
        plt.title(f"{self.y} vs. {self.x}")
        plt.legend(**self.legend_kwargs)
        
        # Set axis limits.
        plt.xlim(left=self.x_min, right=self.x_max)
        plt.ylim(bottom=self.y_min, top=self.y_max)
        
        # Show the plot.
        if self.show:
            plt.show()

        return fig

@dataclass(kw_only=True)
class PlotPlotext(Plot):
    theme: str | None = None # Plotext "theme".
    color: list[str] | None = None # Plot line color.

    def main(self):
        
        # Get data.
        data_dict = self.prepare_data(source_map=self.source_map)
        
        colors = {key: color for color, key in zip(self.color or [None]*len(self.source_map), self.source_map.keys())}
        
        import plotext as plt

        plt.theme(theme=self.theme or "pro")
        
        # Set plot function.
        match self.plot_type:
            case 'line':
                plot_fn = plt.plot
            case 'scatter':
                plot_fn = plt.scatter
            case _:
                raise ValueError(f"Unsupported plot type {self.plot_type}")

        # Plot data in each file.
        for key in self.source_map.keys():
            if self.stats:
                label = f"{key} (min {data_dict['y'][key].min():.4f}, max {data_dict['y'][key].max():.4f}, mean {data_dict['y'][key].mean():.4f})"
            else:
                label = key
            plot_fn(data_dict['x'][key], data_dict['y'][key], color=colors[key], label=label)

        # Set axis labels and title.
        plt.xlabel(self.x)
        plt.ylabel(self.y)
        plt.title(f"{self.y} vs. {self.x}")

        # Set axis limits.
        plt.xlim(left=self.x_min, right=self.x_max)
        plt.ylim(lower=self.y_min, upper=self.y_max)

        # Show the plot.
        plt.show()


@dataclass(kw_only=True)
class PlotLive(DataClassDictMixin):
    """Live plot metric data using X/Y data keys.
    
    If you are unsure of what keys exist, see `InfoMetrics`.
    
    Example live plot X-axis "loss_step" and Y-axis "loss" metric keys using "clear" style in terminal with an update inveral of "1" seconds and only show the last 200 data points:
    >>> cli plot-live --config <file> --x loss_step --y loss --show-last-n 200 --sleep 1 --theme clear
    """
    config: list[pathlib.Path] | None = None # Path of config file in format [yaml, json, python].
    history: list[pathlib.Path] | None = None # Direct path to history file of metrics to plot.
    x: str # X-axis data key.
    y: str # Y-axis data key.
    sleep: float = 1.0 # Sleep interval in seconds.
    x_min: float | None = None # X-axis range min.
    x_max: float | None = None # X-axis range max.
    y_min: float | None = None # Y-axis range min.
    y_max: float | None = None # Y-axis range max.
    show_last_n: int | None = None # Show the last `N` data points.
    theme: str | None = "pro" # Plotext "theme"
    color: list[str] | None = None # Plot line color.
    sync: bool = False # Synchronize X and Y axis data ranges to what is available across all configs.
    stats: bool = False # Report min/max/mean statistics in plot label.
    plot_type: Literal['line', 'scatter'] = 'line' # Plot type as line or scatter plot.
    
    def __post_init__(self):
        # Ensure that either config or history file list was given.
        assert self.config or self.history, "must provide either list of configurations or direct list of history files"
        
        ### Create a map from unique keys to data source.
        self.source_map: dict[str, pathlib.Path | algorithm.Algorithm | MetricManager] = {}
        #
        # Data source is a config object.
        if self.config:
            cfgs = {cfg_path.stem: load_algo_config(path=cfg_path) for cfg_path in self.config}
            
            # Ensure all configs have metrics.
            if any(cfg.metric_manager is None for _,cfg in cfgs.items()):
                raise ValueError("no metric manager provided for one of the metrics")
            
            self.source_map.update(cfgs)
        # Data source is a direct history file.
        if self.history:
            for path in self.history:
                if path.is_dir():
                    metric_manager = LocalFileMetricManager(
                        metric_dir=path.resolve(),
                        metric_file_type="json", # TODO assume JSON for now, but this should be inferred from the directory somehow.
                        )
                    self.source_map.update({path.stem: metric_manager})
                elif path.is_file():
                    self.source_map.update({path.stem: path})
                else:
                    raise ValueError(f"Unsupported history source {path} (must be either directory, regular file, or algorithm config)")

                
            # self.source_map.update({path.stem: path for path in self.history})
    
    def main(self):
        
        colors = {key: color for color, key in zip(self.color or [None]*len(self.source_map), self.source_map.keys())}
        
        # X/Y data.
        data_dict = {
            'x': {},
            'y': {},
        }
        
        import plotext as plt
        plt.theme(theme=self.theme)
        
        # Build list of file-based metric sources to watch for updates.
        metric_sources = []
        for _, src in self.source_map.items():
            if isinstance(src, algorithm.Algorithm):
                metric_sources.append(src.metric_manager.metric_dir)
            elif isinstance(src, MetricManager):
                metric_sources.append(src.metric_dir)
            elif isinstance(src, pathlib.Path):
                metric_sources.append(src)
            else:
                raise ValueError(f"Unsupported history source {src} (must be either directory, regular file, or algorithm config)")
        
        # Set plot function.
        match self.plot_type:
            case 'line':
                plot_fn = plt.plot
            case 'scatter':
                plot_fn = plt.scatter
            case _:
                raise ValueError(f"Unsupported plot type {self.plot_type}")
        
        try:
            while True:
                
                plt.clt() # Clear terminal.
                plt.cld() # Clear data only.
                w, h = plt.terminal_size()
                plt.plot_size(w, h-1) # Automatically resize (use -1 to include space for title).
                
                for key, src in self.source_map.items():
                    # Load metrics from config object.
                    if isinstance(src, algorithm.Algorithm):
                        metric_manager = src.metric_manager # It cannot be None here because we check in main.
                        metric_manager.load()
                        metrics = metric_manager.get_metrics()
                    # Load metrics directly from metric manager.
                    elif isinstance(src, MetricManager):
                        metric_manager = src
                        metric_manager.load()
                        metrics = metric_manager.get_metrics()
                    # Load metrics from history file.
                    elif isinstance(src, pathlib.Path):
                        history_list = tools.load_train_results(src)
                        metrics = tools.listofdict_to_dictoflist(history_list)
                    else:
                        raise ValueError(f"Unsupported history source {src} (must be either directory, regular file, or algorithm config)")
                    
                    if self.x not in metrics:
                        print(f"ERROR:X-axis key {self.x} not found in metrics file", file=sys.stderr)
                        return
                    if self.y not in metrics:
                        print(f"ERROR:Y-axis key {self.y} not found in metrics file", file=sys.stderr)
                        return
                    
                    # Preserve data as an array.
                    data_dict['x'][key] = np.array(metrics[self.x])
                    data_dict['y'][key] = np.array(metrics[self.y])
                
                # Synchronize X/Y axis data display to data available across all configs.
                if self.sync:
                    x_min = max(x_data.min() for _, x_data in data_dict['x'].items())
                    x_max = min(x_data.max() for _, x_data in data_dict['x'].items())
                    # Clip X-axis data for all configs.
                    for key in self.source_map.keys():
                        # Clip X min.
                        idx = data_dict['x'][key] >= x_min
                        data_dict['x'][key] = data_dict['x'][key][idx]
                        data_dict['y'][key] = data_dict['y'][key][idx]
                        # Clip X max.
                        idx = data_dict['x'][key] <= x_max
                        data_dict['x'][key] = data_dict['x'][key][idx]
                        data_dict['y'][key] = data_dict['y'][key][idx]

                # Apply any clipping on data after sync.
                for key in self.source_map.keys():
                    # Clip X-axis data.
                    if self.x_min is not None:
                        idx = data_dict['x'][key] >= self.x_min
                        data_dict['x'][key] = data_dict['x'][key][idx]
                        data_dict['y'][key] = data_dict['y'][key][idx]
                    if self.x_max is not None:
                        idx = data_dict['x'][key] <= self.x_max
                        data_dict['x'][key] = data_dict['x'][key][idx]
                        data_dict['y'][key] = data_dict['y'][key][idx]
                    
                    # Clip Y-axis data.
                    if self.y_min is not None:
                        idx = data_dict['y'][key] >= self.y_min
                        data_dict['x'][key] = data_dict['x'][key][idx]
                        data_dict['y'][key] = data_dict['y'][key][idx]
                    if self.y_max is not None:
                        idx = data_dict['y'][key] <= self.y_max
                        data_dict['x'][key] = data_dict['x'][key][idx]
                        data_dict['y'][key] = data_dict['y'][key][idx]
                    
                    # Select the last `N` data points.
                    if self.show_last_n is not None:
                        data_dict['x'][key] = data_dict['x'][key][-self.show_last_n:]
                        data_dict['y'][key] = data_dict['y'][key][-self.show_last_n:]
                
                # Plot data in each file.
                for key in self.source_map.keys():
                    if self.stats:
                        label = f"{key} (min {data_dict['y'][key].min():.4f}, max {data_dict['y'][key].max():.4f}, mean {data_dict['y'][key].mean():.4f}, first {data_dict['y'][key][0]:.4f}, last {data_dict['y'][key][-1]:.4f})"
                    else:
                        label = key
                    plot_fn(data_dict['x'][key], data_dict['y'][key], color=colors[key], label=label)
                
                # Set axis labels and title.
                plt.xlabel(self.x)
                plt.ylabel(self.y)
                plt.title(f"Live plot: {datetime.datetime.now()}")
                
                # Show the plot.
                plt.show()
                
                # Blocking wait for next file change.
                if not tools.wait_for_list_of_file_change(metric_sources, interval=self.sleep):
                    return
                
        except KeyboardInterrupt:
            return


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
            with config.metric_manager as metric_manager:
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
            # Load metrics into memory.
            with cfg.metric_manager as metric_manager:
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


from qnrl.serialization import SerialType, hash_serializable_object
@dataclass(kw_only=True)
class Train(DataClassDictMixin):
    """Run training algorithm using the given config."""
    config: dict[str, int | float | str | bool | list | tuple | dict | None]
    num_eval: int = 10 # Number of evaluation runs after training.
    eval_seed: int | None = None # Override evaluation seed. Defaults to using configuration seed.
    video_fps: int | None = None # Override video FPS. Defaults to using FPS set by algorithm report.
    eval_total_timesteps: int | None = None # Override total number of timesteps for evaluation (this helps limit length of video generation).
    
    def main(self):
        
        # Deserialize algorithm config.
        algo_config = qnrl.Algorithm.from_dict(self.config)
        
        # Generate hash for configuration object.
        cfg_hash, cfg_str = hash_serializable_object(self.config)
        print(f"Config dictionary: {self.config}")
        print(f"Config string: {cfg_str}")
        print(f"Hash: {cfg_hash}")
        
        # Create a manager for interfacing with wandb.
        metric_manager = WandBMetricManager(
            login=True, # Already logged in at beginning of notebook.
            name=f"{cfg_hash}",
            terminate_on_wait_until_finished=False,
            define_metrics=[
                {"name": "*", "summary": "min"},
                {"name": "*", "summary": "max"},
                {"name": "*", "summary": "mean"},
                {"name": "*", "summary": "first"},
                {"name": "*", "summary": "last"},
            ],
        )
        
        # Log the original configuration string.
        metric_manager.run.config.update({"config_hash": cfg_hash, "config_string": cfg_str})
        
        # Generate hash for config WITHOUT seed key.
        cfg_without_seed = self.config.copy()
        cfg_without_seed.pop("seed", None)
        cfg_hash_without_seed, cfg_str_without_seed = hash_serializable_object(cfg_without_seed)
        metric_manager.run.config.update({"seed_group_hash": cfg_hash_without_seed, "seed_group_str": cfg_str_without_seed})
        print(f"Seed Group Hash: {cfg_hash_without_seed}")
        
        # Train and evaluate within a temporary directory.
        with tempfile.TemporaryDirectory(delete=True) as tmp_dir:
            root_dir = pathlib.Path(tmp_dir)
            print(f"Root path: {root_dir}")
            
            # Run training algorithm.
            print("Train:")
            train_dir = root_dir / 'train' # Root directory for training files.
            train_dir.mkdir(parents=True, exist_ok=True)

            # Train the model(s).
            _, train_report = algo_config.train(metric_manager=metric_manager)
            
            # Add training report to run summary.
            metric_manager.run.summary.update(train_report)
            
            # Save model(s).
            models_save_root = train_dir / 'models'
            models_save_root.mkdir(parents=True, exist_ok=True)
            _ = algo_config.save(models_save_root)
            artifact = wandb.Artifact(name=f"run-{metric_manager.run.id}-models", type="model")
            artifact.add_dir(str(models_save_root))
            metric_manager.run.log_artifact(artifact).wait() # Ensure models are logged.
            
            # Evaluate multiple times and preserve results.
            print("Eval:")
            eval_dir = root_dir / 'eval' # Root directory for training files.
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_reports = []
            eval_seed = self.eval_seed if self.eval_seed is not None else algo_config.seed
            key = jax.random.PRNGKey(seed=eval_seed)
            for eval_idx in range(self.num_eval):
                
                # Generate a new random key.
                key, subkey = jax.random.split(key=key)
                
                # Evaluate the model using a different random seed.
                seed_eval = int(subkey[0])
                eval_kwargs = {}
                if self.eval_total_timesteps is not None:
                    eval_kwargs['total_timesteps'] = self.eval_total_timesteps
                _, eval_report = algo_config.eval(seed=seed_eval, **eval_kwargs)
                for i, (k,v) in enumerate(eval_report.items()):
                    # Generate videos compatible with WandB.
                    if isinstance(v, moviepy.VideoClip):
                        unique_video_key = f"{cfg_hash}_eval={eval_idx}_seed={seed_eval}_item={i}"
                        video_filepath = eval_dir / f"{unique_video_key}.mp4"
                        v.write_videofile(video_filepath, codec="libx264")
                        fps = self.video_fps if self.video_fps is not None else v.fps
                        eval_report[k] = wandb.Video(video_filepath, fps=fps, format="mp4", caption=unique_video_key)
                    elif isinstance(v, MatplotlibFuncAnimation):
                        unique_video_key = f"{cfg_hash}_eval={eval_idx}_seed={seed_eval}_item={i}"
                        video_filepath = eval_dir / f"{unique_video_key}.mp4"
                        v.save(video_filepath, extra_args=['-vcodec', 'libx264'])
                        eval_report[k] = wandb.Video(video_filepath, fps=self.video_fps, format="mp4", caption=unique_video_key)
                
                # Add an index counter for the evaluation report.
                eval_report['eval/index'] = eval_idx
                
                # Preserve the evaluation report.
                eval_reports.append(eval_report)
                
                # Log evaluation results.
                metric_manager.run.log(eval_report)

            # Finalize the logging session.
            metric_manager.run.finish()


@dataclass(kw_only=True)
class ModelSize(DataClassDictMixin):
    """Report model size(s) for training configuration."""
    config: dict[str, int | float | str | bool | list | tuple | dict | None]
    model_key: str = 'model'
    
    def main(self):
        
        # Deserialize algorithm config.
        algo_config = qnrl.Algorithm.from_dict(self.config)
        
        model = algo_config.__getattribute__(self.model_key)
        print(qnrl.get_model_total_parameter_count(model))


@dataclass(kw_only=True)
class SweepRunCount(DataClassDictMixin):
    """Report the number of run configurations within a WandB grid sweep.
    
    >>> cli sweep-run-count --config ./sweeps/qnrl_CartPole-v1.yaml # 96
    """
    config: pathlib.Path
    """Path to sweep configuration YAML file."""
    
    def main(self):
        
        assert self.config.exists() and self.config.is_file(), "Configuration file does not exist or is not a file"
        
        # Load sweep YAML file.
        with open(str(self.config), 'r') as f:
            sweep_config = yaml.safe_load(f)
        
        # Extract the run configurations within the sweep grid.
        run_metas = tools.expand_wandb_sweep_grid(sweep_config)
        
        # Report the number of run configurations.
        print(len(run_metas))



@dataclass(kw_only=True)
class Args(DataClassDictMixin):
    """Algorithm CLI."""
    # config: pathlib.Path # Path of config file in format [yaml, json, python].
    mode: Union[
        Annotated[DumpAlgorithm, tyro.conf.subcommand(name='dump')],
        Annotated[InfoModel, tyro.conf.subcommand(name='info-model')],
        Train,
        Metrics,
        PlotMatplotlib,
        PlotPlotext,
        PlotLive,
        ModelSize,
        SweepRunCount,
    ]


if __name__ == "__main__":
    
    args = tyro.cli(Args, config=(
        tyro.conf.EnumChoicesFromValues,
        tyro.conf.OmitSubcommandPrefixes,
        tyro.conf.UsePythonSyntaxForLiteralCollections,
        tyro.conf.FlagConversionOff,
        ))
    
    # Run program mode.
    args.mode.main()