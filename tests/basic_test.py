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
        db["test"] = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        assert "test" in db
        assert db["test"] is not None
        assert db["test"][0] is not None
        assert db["test"][0].shape == (5,)
        assert np.sum(db["test"][0]) == 10.0
        assert db["test"][1] is None

        db["test2"] = (
            np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            {"foo": "bar"},
        )
        assert "test2" in db
        assert db["test2"] is not None
        assert db["test2"][0].shape == (5,)
        assert np.sum(db["test2"][0]) == 5.0
        assert db["test2"][1] is not None
        assert db["test2"][1]["foo"] == "bar"
        assert len(db) == 2

        for key, vector, metadata in db:
            assert [key, vector] is not None
            assert key in ["test", "test2"]
            if key == "test2":
                assert metadata is not None
                assert metadata["foo"] == "bar"

        assert "test" in db
        db.update_vector("test", np.array([2.0, 2.0, 2.0, 3.0, 2.0], dtype=np.float32))
        assert "test" in db
        assert db["test"][0].shape == (5,)
        assert np.sum(db["test"][0]) == 11.0
        assert db.index.get_current_count() == 2
        db.update_metadata("test", {"foo": "baz"})
        assert "test" in db
        assert db["test"][1]["foo"] == "baz"
        db.update_metadata("test", {"foo": "booz", "bar": "baz"})
        assert "test" in db
        assert db["test"][1]["foo"] == "booz"
        assert db["test"][1]["bar"] == "baz"

        result = db.search(np.array([1.1, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), k=1)
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

        db.close()

        try:
            db["test2"]
        except RuntimeError as e:
            assert "closed" in str(e)

        try:
            db["test3"] = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        except RuntimeError as e:
            assert "closed" in str(e)

        try:
            del db["test2"]
        except RuntimeError as e:
            assert "closed" in str(e)

        try:
            "test2" in db
        except RuntimeError as e:
            assert "closed" in str(e)


def test_resizing():
    db_size = 180
    resize_buffer_size = 5

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db")
        db = Database(db_path, dim=5, resize_buffer_size=resize_buffer_size)
        for i in range(db_size):
            db[f"test{i}"] = np.float32(np.random.rand(5))
        assert len(db) == db_size


def test_reloading():
    db_size = 180
    resize_buffer_size = 5
    dim = 384

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db")
        db = Database(db_path, dim=dim, resize_buffer_size=resize_buffer_size)
        for i in range(db_size - 1):
            db[f"test{i}"] = np.float32(np.random.rand(dim))

        db[f"test{db_size - 1}"] = np.float32(np.random.rand(dim)), {"foo": "bar"}
        vec, metadata = db[f"test{db_size - 1}"]

        assert len(db) == db_size
        assert vec is not None
        assert vec.shape == (dim,)
        assert metadata["foo"] == "bar"

        db.close()

        db = Database(db_path, dim=dim, resize_buffer_size=resize_buffer_size)

        assert db[f"test{db_size - 1}"] is not None
        assert len(db) == db_size
        assert vec is not None
        assert vec.shape == (dim,)
        assert metadata["foo"] == "bar"

        db["foo"] = np.float32(np.random.rand(dim)), {"bar": "baz"}
        vec, metadata = db["foo"]
        assert len(db) == db_size + 1
        assert vec is not None
        assert vec.shape == (dim,)
        assert metadata["bar"] == "baz"


def test_autorepair_when_db_is_inconsistent():
    db_size = 180
    resize_buffer_size = 200
    dim = 384

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db")
        db = Database(db_path, dim=dim, resize_buffer_size=resize_buffer_size)

        for i in range(db_size - 1):
            db[f"test{i}"] = np.float32(np.random.rand(dim))

        db[f"test{db_size - 1}"] = np.float32(np.random.rand(dim)), {"foo": "bar"}
        vec, metadata = db[f"test{db_size - 1}"]

        assert len(db) == db_size
        assert vec is not None
        assert vec.shape == (dim,)
        assert metadata["foo"] == "bar"

        # Close the database with unsaved changes
        db.mapping.close()
        db.metadata.close()

        db = Database(db_path, dim=dim, resize_buffer_size=resize_buffer_size)
        assert len(db) == db_size
        db["test-test"] = np.float32(np.random.rand(dim)), {"foo": "baz"}
        vec, metadata = db["test-test"]
        assert len(db) == db_size + 1
        assert vec is not None
        assert vec.shape == (dim,)
        assert metadata["foo"] == "baz"

        num_results = 3
        for key, vector, distance, metadata in db.search(
            np.random.rand(dim), k=num_results
        ):
            assert "test" in key
            assert vector.shape == (dim,)
            assert type(distance) == np.float32
            assert metadata == {} or metadata["foo"] in ["bar", "baz"]
            num_results -= 1
        assert num_results == 0
