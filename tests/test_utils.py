import numpy as np
import os
from src.utils import save_object, load_object


def test_save_and_load_object(tmp_path):
    test_obj = {"a": 1, "b": 2}
    file_path = tmp_path / "test.pkl"
    save_object(str(file_path), test_obj)
    loaded_obj = load_object(str(file_path))

    assert test_obj == loaded_obj