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

        for key, vector, metadata in db:
            assert [key, vector] is not None
            assert key in ["test", "test2"]
            if key == "test2":
                assert metadata is not None
                assert metadata["foo"] == "bar"

        result = db.search(np.array([1.1, 1.0, 1.0, 1.0, 1.0]), k=1)
        assert result is not None
        key, vector, distance, metadata = result.__next__()
        assert [key, vector, distance, metadata] is not None
        assert key == "test2"
        assert np.sum(vector) == 5.0
        assert distance < 0.1
        assert metadata is not None
        assert metadata["foo"] == "bar"

        del db["test"]
        assert "test" not in db
        assert db["test2"] is not None
        assert "test3" not in db
        assert len(db) == 1
