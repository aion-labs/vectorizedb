import os
import tempfile

import numpy as np

from vectorizedb import Database


def test_overall_functionality():
    """
    Test the overall functionality of the Database class.

    This test function creates a temporary directory, initializes a Database object,
    and performs various assertions to verify the functionality of the database.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db")
        db = Database(db_path, dim=5)
        db["test"] = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        assert "test" in db
        assert db["test"] is not None
        assert db["test"][0] is not None
        assert db["test"][1] is None

        db["test2"] = (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), {"foo": "bar"})
        assert "test2" in db
        assert db["test2"] is not None
        assert db["test2"][0] is not None
        assert db["test2"][1] is not None
        assert db["test2"][1]["foo"] == "bar"
        assert len(db) == 2

        del db["test"]
        assert "test" not in db
        assert db["test2"] is not None
        assert "test3" not in db
        assert len(db) == 1
