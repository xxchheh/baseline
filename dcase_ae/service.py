from pathlib import Path
from typing import Any

from dcase_ae.health_library import HealthLibraryConfig, analyze_json_file_with_runtime
from dcase_ae.runtime import RuntimeConfig


def analyze_file(
    input_path: str | Path,
    machine_type: str,
    *,
    library_name: str | None = None,
    runtime_config: RuntimeConfig | None = None,
    build_config: HealthLibraryConfig | None = None,
) -> dict[str, Any]:
    return analyze_json_file_with_runtime(
        input_path,
        machine_type,
        runtime_config=runtime_config,
        library_name=library_name,
        build_config=build_config,
    )
