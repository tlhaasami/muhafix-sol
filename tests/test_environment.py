import os
import yaml
import pytest

def test_config_file_exists_and_loads():
    assert os.path.exists("config.yaml"), "config.yaml does not exist"
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg is not None
    assert cfg["project"]["name"] == "muhafiz-sol"
    assert cfg["project"]["random_seed"] == 42
    assert "paths" in cfg
    assert "dataset" in cfg

def test_directories_exist():
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/gold",
        "data/features",
        "src",
        "models",
        "tests",
        "docs",
        "reports"
    ]
    for d in required_dirs:
        assert os.path.exists(d), f"Required directory {d} is missing"

def test_requirements_file_exists():
    assert os.path.exists("requirements.txt")
    with open("requirements.txt", "r", encoding="utf-8") as f:
        content = f.read()
    assert "pandas" in content
    assert "scikit-learn" in content
    assert "datasets" in content
