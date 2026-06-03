from __future__ import annotations

import json
from collections.abc import Mapping


def _serialize_value(value):
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_serialize_value(item) for item in value)
    return value


def flatten_mapping(mapping: Mapping, prefix: str = "context") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in mapping.items():
        normalized_key = f"{prefix}_{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_mapping(value, normalized_key))
        elif isinstance(value, (list, tuple, set)):
            flat[normalized_key] = json.dumps(_serialize_value(value), sort_keys=True)
        else:
            flat[normalized_key] = value
    return flat


def build_run_context(
    *,
    config: dict,
    models_config: dict,
    dataset_name: str,
    split_name: str,
    seed: int,
    family: str,
    model_name: str | None = None,
    scenario: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    dataset_key = dataset_name.lower()
    context: dict[str, object] = {
        "dataset": {
            "name": dataset_name.upper(),
            "config": _serialize_value(config["datasets"][dataset_key]),
        },
        "preprocessing": _serialize_value(config.get("preprocessing", {})),
        "noise": _serialize_value(config.get("noise", {})),
        "project": {
            "device": config.get("project", {}).get("device", "cpu"),
            "random_seeds": _serialize_value(config.get("project", {}).get("random_seeds", [])),
        },
        "training": _serialize_value(models_config.get("training", {})),
        "automata": _serialize_value(models_config.get("automata", {})),
        "run": {
            "family": family,
            "seed": int(seed),
            "split": split_name,
        },
    }
    if model_name is not None:
        context["model"] = {
            "name": model_name.upper(),
            "config": _serialize_value(models_config.get("deep_learning", {}).get(model_name.lower(), {})),
        }
    if scenario is not None:
        context["run"]["scenario"] = scenario
    if extra:
        context["extra"] = _serialize_value(extra)
    return context


def attach_context_to_record(record: dict[str, object], context: dict[str, object]) -> dict[str, object]:
    enriched = dict(record)
    enriched["experiment_context"] = json.dumps(_serialize_value(context), sort_keys=True)
    enriched.update(flatten_mapping(context))
    return enriched
