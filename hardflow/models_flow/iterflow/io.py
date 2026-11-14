"""Strict, atomic JSON output for experiment records."""
import json
from pathlib import Path

def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)

write = write_json
