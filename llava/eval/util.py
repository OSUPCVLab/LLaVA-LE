import json


def load_file_jsonl(path):
    """Load JSONL file where each line is a JSON object"""
    with open(path) as f:
        return [json.loads(row) for row in f]


def load_file_json(path):
    """Load standard JSON file (array of objects)"""
    with open(path) as f:
        return json.load(f)


def load_file_auto(path):
    """Automatically detect JSON vs JSONL"""
    if path.endswith(".jsonl"):
        return load_file_jsonl(path)
    elif path.endswith(".json"):
        return load_file_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def get_avg(x):
    """Calculate average of a list"""
    return sum([float(y) for y in x]) / len(x)