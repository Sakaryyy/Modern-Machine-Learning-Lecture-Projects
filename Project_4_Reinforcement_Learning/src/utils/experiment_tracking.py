"""Experiment tracking utilities for reinforcement learning workflows."""

from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from Project_4_Reinforcement_Learning.src.environment import StepMetrics

__all__ = [
    "EpisodeRecorder",
    "RunArtifactManager",
    "RunMetadata",
]


@dataclass(slots=True)
class RunMetadata:
    """Metadata for a single experiment run.

    Parameters
    ----------
    run_id:
        Unique identifier for the run folder.
    created_at:
        ISO-like timestamp string for the run creation time.
    root_dir:
        Root directory where all run artifacts are stored.
    notes:
        Optional notes describing the run context.
    """

    run_id: str
    created_at: str
    root_dir: Path
    notes: Optional[str] = None


class RunArtifactManager:
    """Manage experiment artifact storage in a structured directory tree.

    Notes
    -----
    The manager creates a run-specific folder containing subfolders for
    configurations, data exports, plots, saved models, and environment state
    snapshots. Each helper method is designed to be reusable across different
    experiments and policies.
    """

    def __init__(self, output_root: Path, run_name: str | None = None) -> None:
        self._output_root = output_root
        self._run_name = run_name
        self._metadata: RunMetadata | None = None
        self._run_dir: Path | None = None
        self._dirs: Dict[str, Path] = {}

    @property
    def metadata(self) -> RunMetadata:
        """Return metadata for the active run.

        Raises
        ------
        RuntimeError
            If the run directory has not been initialized yet.
        """

        if self._metadata is None:
            raise RuntimeError("RunArtifactManager has not been initialized.")
        return self._metadata

    def initialize(self) -> RunMetadata:
        """Create the directory tree for a new run.

        Returns
        -------
        RunMetadata
            Metadata describing the newly created run.
        """

        # Ensure the root exists before carving out a run-specific folder.
        self._output_root.mkdir(parents=True, exist_ok=True)
        run_id = self._generate_run_id()
        run_dir = self._output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Standardized subdirectories keep artifacts consistent across runs.
        directories = {
            "configs": run_dir / "configs",
            "data": run_dir / "data",
            "plots": run_dir / "plots",
            "models": run_dir / "models",
            "environment_state": run_dir / "environment_state",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._metadata = RunMetadata(
            run_id=run_id,
            created_at=created_at,
            root_dir=run_dir,
        )
        self._run_dir = run_dir
        self._dirs = directories
        return self._metadata

    def save_json(self, payload: Mapping[str, Any], filename: str, subdir: str = "configs") -> Path:
        """Save JSON data into the requested subdirectory.

        Parameters
        ----------
        payload:
            Mapping with JSON-serializable values.
        filename:
            Name of the JSON file.
        subdir:
            Subdirectory key (e.g., ``"configs"`` or ``"data"``).

        Returns
        -------
        pathlib.Path
            Path to the saved JSON file.
        """

        directory = self._resolve_subdir(subdir)
        path = directory / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return path

    def save_config(self, config: Any, filename: str = "config.json") -> Path:
        """Persist a configuration object to disk.

        Parameters
        ----------
        config:
            Dataclass or dictionary containing configuration values.
        filename:
            Name of the output JSON file.

        Returns
        -------
        pathlib.Path
            Path to the saved file.
        """

        if is_dataclass(config):
            payload: Mapping[str, Any] = asdict(config)
        elif isinstance(config, Mapping):
            payload = config
        else:
            raise TypeError("Config must be a dataclass or mapping.")
        return self.save_json(payload, filename=filename, subdir="configs")

    def save_dataframe(
            self,
            frame: pd.DataFrame,
            filename: str,
            subdir: str = "data",
            include_excel: bool = True,
    ) -> List[Path]:
        """Save a dataframe to CSV (and optionally Excel) format.

        Parameters
        ----------
        frame:
            DataFrame containing experiment data.
        filename:
            Base filename without extension.
        subdir:
            Subdirectory key for the output file.
        include_excel:
            If ``True`` also save the dataframe as an Excel workbook.

        Returns
        -------
        list[pathlib.Path]
            Paths to the saved files.
        """

        directory = self._resolve_subdir(subdir)
        csv_path = directory / f"{filename}.csv"
        frame.to_csv(csv_path, index=False)
        paths = [csv_path]

        if include_excel:
            excel_path = directory / f"{filename}.xlsx"
            frame.to_excel(excel_path, index=False)
            paths.append(excel_path)

        return paths

    def save_policy(self, policy: Any, filename: str = "policy.pkl") -> Path:
        """Serialize and save a policy artifact.

        Parameters
        ----------
        policy:
            Policy object to serialize.
        filename:
            Name of the saved artifact.

        Returns
        -------
        pathlib.Path
            Path to the saved policy file.
        """

        directory = self._resolve_subdir("models")
        path = directory / filename

        # Prefer native save hooks when policies expose them; otherwise pickle.
        save_method = getattr(policy, "save", None)
        if callable(save_method):
            save_method(path)
        else:
            with path.open("wb") as handle:
                pickle.dump(policy, handle)

        return path

    def save_text(self, content: str, filename: str, subdir: str = "data") -> Path:
        """Save raw text data to disk.

        Parameters
        ----------
        content:
            String content to write.
        filename:
            Name of the output text file.
        subdir:
            Subdirectory key for the output file.

        Returns
        -------
        pathlib.Path
            Path to the saved text file.
        """

        directory = self._resolve_subdir(subdir)
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        return path

    def _resolve_subdir(self, name: str) -> Path:
        """Return the path for a configured subdirectory."""

        if not self._dirs:
            raise RuntimeError("RunArtifactManager has not been initialized.")
        if name not in self._dirs:
            raise KeyError(f"Unknown subdir '{name}'.")
        return self._dirs[name]

    def _generate_run_id(self) -> str:
        """Create a filesystem-friendly run identifier."""

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if not self._run_name:
            return f"run_{timestamp}"
        sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", self._run_name.strip())
        return f"run_{timestamp}_{sanitized}"


class EpisodeRecorder:
    """Collect step-level data from environment rollouts.

    Notes
    -----
    The recorder stores observations, actions, rewards, and detailed environment
    metrics. The resulting dataframe can be exported directly to CSV/Excel or
    passed into plotting utilities.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._cumulative_reward = 0.0

    def record_step(
            self,
            observation: np.ndarray,
            action: np.ndarray,
            reward: float,
            terminated: bool,
            truncated: bool,
            metrics: StepMetrics,
    ) -> None:
        """Store a single step transition.

        Parameters
        ----------
        observation:
            Observation vector at the current step.
        action:
            Action array chosen by the policy.
        reward:
            Reward received after the action.
        terminated:
            Episode termination flag.
        truncated:
            Episode truncation flag.
        metrics:
            Environment metrics emitted by the environment.
        """

        self._cumulative_reward += float(reward)

        # Store primitive values for serialization-friendly exports.
        observation_values = observation.astype(float).tolist()
        action_values = action.astype(float).tolist()

        record = {
            "time_step": metrics.time_step,
            "time_of_day": metrics.time_of_day,
            "reward": float(reward),
            "cumulative_reward": self._cumulative_reward,
            "terminated": terminated,
            "truncated": truncated,
            "observation_time_of_day": observation_values[0],
            "observation_day_of_year": observation_values[1],
            "observation_buying_price": observation_values[2],
            "observation_battery_energy": observation_values[3],
            "observation_battery_capacity": observation_values[4],
            "observation_battery_health": observation_values[5],
            "action_solar_to_demand": action_values[0],
            "action_solar_to_battery": action_values[1],
            "action_battery_to_demand": action_values[2],
            "action_battery_to_grid": action_values[3],
            "action_grid_to_battery": action_values[4],
        }
        for idx, value in enumerate(metrics.forecast_solar_intensity, start=1):
            record[f"forecast_solar_intensity_{idx}"] = float(value)
        for idx, value in enumerate(metrics.forecast_market_price, start=1):
            record[f"forecast_market_price_{idx}"] = float(value)
        for idx, value in enumerate(metrics.forecast_demand, start=1):
            record[f"forecast_demand_{idx}"] = float(value)
        record.update(asdict(metrics))
        self._records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the recorded steps as a dataframe.

        Returns
        -------
        pandas.DataFrame
            Tabular representation of all recorded steps.
        """

        frame = pd.DataFrame(self._records)
        if not frame.empty:
            frame = frame.sort_values("time_step").reset_index(drop=True)
        return frame

    def summary(self) -> Dict[str, float | int]:
        """Return summary statistics for the episode.

        Returns
        -------
        dict
            Dictionary containing total reward and episode length.
        """

        return {
            "total_reward": float(self._cumulative_reward),
            "num_steps": len(self._records),
        }
