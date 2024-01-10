from typing import Optional, Union, List, Tuple, Dict
import os
import pathlib
import hnswlib
import lmdb
import numpy as np
import msgpack


class Database:
    def __init__(
        self,
        path: os.PathLike,
        dim: int,
        readonly: bool = False,
        similarity: str = "cosine",
        resize_buffer_size: int = 10000,
    ):
        """
        Initialize a Database object.

        Args:
            path (os.PathLike): The path to the directory where the VectorizeDB data will be stored.
            dim (int): The dimensionality of the vectors to be stored.
            readonly (bool, optional): Whether to open the database in read-only mode. Defaults to False.
            similarity (str, optional): The similarity metric to use for indexing. Defaults to "cosine".
            resize_buffer_size (int, optional): The number of elements to add to the index when resizing. Defaults to 10000.
        """

        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        metadata_path = bytes(self.path / "metadata")
        mapping_path = bytes(self.path / "mapping")
        index_path = self.path / "index.tree"

        self.metadata = lmdb.open(metadata_path, readonly=readonly, map_size=int(1e12))
        self.mapping = lmdb.open(mapping_path, readonly=readonly, map_size=int(1e12))

        self.max_elements = (
            self.mapping.stat()["entries"] + resize_buffer_size
        ) or resize_buffer_size
        self.resize_buffer_size = resize_buffer_size

        self.index = hnswlib.Index(space=similarity, dim=dim)
        if index_path.exists():
            self.index.load_index(str(index_path), max_elements=self.max_elements)
            self.sync()
        else:
            self.index.init_index(
                max_elements=self.max_elements, ef_construction=200, M=16
            )

    def add(self, key: str, vector: np.array, metadata: Optional[Dict] = None):
        """
        Add a vector to the database with the specified key.

        Args:
            key (str): The key associated with the vector, used for lookups.
            vector (np.array): The vector to be added.
            metadata (Optional[Dict]): Optional metadata associated with the vector.

        Returns:
            None
        """
        with self.mapping.begin(write=True) as mapping_txn, self.metadata.begin(
            write=True
        ) as metadata_txn:
            idx_id = self.index.get_current_count().to_bytes(4, "big")
            try:
                self.index.add_items([vector])
            except RuntimeError as e:
                if "exceeds the specified limit" in str(e):
                    self._resize()
                    self.index.add_items([vector])
                else:
                    raise e

            mapping_txn.put(idx_id, key.encode("utf-8"))
            metadata = metadata or {}
            metadata = {**metadata, "__vec__": vector.tobytes(), "__idx__": idx_id}
            metadata_txn.put(key.encode("utf-8"), msgpack.packb(metadata))

    def sync(self):
        """
        Sync the database to disk.

        Returns:
            None
        """
        len_metadata = self.metadata.stat()["entries"]
        len_mapping = self.mapping.stat()["entries"]
        len_index = self.index.get_current_count()

        if len_metadata != len_mapping:
            raise RuntimeError(
                f"Metadata and mapping databases are out of sync. {len_metadata} != {len_mapping}"
            )

        if len_metadata != len_index:
            self.index.resize_index(len_metadata + self.resize_buffer_size)
            with self.mapping.begin() as mapping_txn, self.metadata.begin() as metadata_txn:
                with mapping_txn.cursor() as cursor:
                    for idx_id, key in cursor:
                        metadata = metadata_txn.get(key)
                        metadata = msgpack.unpackb(metadata)
                        idx_id = metadata["__idx__"]
                        vec = np.frombuffer(metadata["__vec__"])
                        self.index.add_items([vec], [int.from_bytes(idx_id, "big")])

        self.mapping.sync()
        self.metadata.sync()
        self.index.save_index(str(self.path / "index.tree"))

    def _resize(self):
        self.sync()
        self.max_elements *= 2
        self.index.resize_index(self.max_elements)

    def search(
        self, vector: np.array, k: int = 10
    ) -> List[Tuple[str, np.array, float, Dict]]:
        """
        Search the database for the nearest neighbors of the specified vector.

        Args:
            vector (np.array): The vector to search for.
            k (int, optional): The number of nearest neighbors to return. Defaults to 10.

        Returns:
            List[Tuple[str, np.array, float, Dict]]: A list of tuples containing the key, vector, distance, and metadata of the nearest neighbors.
        """
        idx_ids, distances = self.index.knn_query(vector, k=k)
        with self.mapping.begin() as mapping_txn, self.metadata.begin() as metadata_txn:
            for idx_id, distance in zip(idx_ids[0], distances[0]):
                idx_id = int(idx_id).to_bytes(4, "big")
                key = mapping_txn.get(idx_id)
                metadata = metadata_txn.get(key)
                metadata = msgpack.unpackb(metadata)
                vector = np.frombuffer(metadata["__vec__"])
                del metadata["__vec__"]
                del metadata["__idx__"]
                yield key.decode("utf-8"), vector, distance, metadata

    def __setitem__(self, key: str, val: Union[List, Tuple]):
        if isinstance(val, tuple):
            vector, metadata = val
        else:
            vector, metadata = val, None
        self.add(key, vector, metadata)

    def __getitem__(self, key: str) -> List:
        with self.metadata.begin() as metadata_txn:
            metadata = metadata_txn.get(key.encode("utf-8"))
            if not metadata:
                raise KeyError(key)
            metadata = msgpack.unpackb(metadata)
            vector = np.frombuffer(metadata["__vec__"])
            del metadata["__vec__"]
            del metadata["__idx__"]
            if metadata == {}:
                metadata = None
            return vector, metadata

    def __contains__(self, key: str) -> bool:
        with self.metadata.begin() as metadata_txn:
            return bool(metadata_txn.get(key.encode("utf-8")))

    def __iter__(self):
        with self.mapping.begin() as mapping_txn, self.metadata.begin() as metadata_txn:
            with mapping_txn.cursor() as cursor:
                for _, key in cursor:
                    metadata = metadata_txn.get(key)
                    metadata = msgpack.unpackb(metadata)
                    vector = np.frombuffer(metadata["__vec__"])
                    del metadata["__vec__"]
                    del metadata["__idx__"]
                    if metadata == {}:
                        metadata = None
                    yield key.decode("utf-8"), vector, metadata

    def __delitem__(self, key: str):
        with self.mapping.begin(write=True) as mapping_txn, self.metadata.begin(
            write=True
        ) as metadata_txn:
            metadata = metadata_txn.get(key.encode("utf-8"))
            if not metadata:
                raise KeyError(key)
            metadata = msgpack.unpackb(metadata)
            idx_id = metadata["__idx__"]

            mapping_txn.delete(idx_id)
            self.index.mark_deleted(int.from_bytes(idx_id, "big"))
            metadata_txn.delete(key.encode("utf-8"))

    def __len__(self) -> int:
        with self.mapping.begin() as mapping_txn:
            return mapping_txn.stat()["entries"]

    def close(self):
        self.sync()
        self.mapping.close()
        self.metadata.close()
