import json
import os
from typing import Any, List


def read_json_list(path: str) -> List[Any]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


def write_json_list(path: str, data: List[Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_json_item(path: str, item: Any) -> None:
    data = read_json_list(path)
    data.append(item)
    write_json_list(path, data)
