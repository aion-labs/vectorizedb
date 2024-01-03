VectorizeDB
===========

Overview
--------

VectorizeDB is a Python package designed for the efficient storage and retrieval of high-dimensional vectors. It's particularly useful in applications like machine learning and information retrieval. The package utilizes hnswlib for fast approximate nearest neighbor searches and LMDB for scalable and reliable storage.

Installation
------------

To install VectorizeDB, ensure you have Python 3.10 or higher. It can be installed via pip:

.. code-block:: bash

    pip install vectorizedb

Usage
-----

**Initialization**

.. code-block:: python

    from vectorizedb import Database

    # Initialize a new database
    db = Database(path="path/to/db", dim=128, readonly=False, similarity="cosine", max_elements=1000000)

**Adding Data**

.. code-block:: python

    import numpy as np

    # Add a vector with an associated key
    db.add(key="sample_key", vector=np.random.rand(128))

    # Add a vector with metadata
    db.add(key="another_key", vector=np.random.rand(128), metadata={"info": "sample metadata"})

    # Another way to add data
    db["yet_another_key"] = (np.random.rand(128), {"info": "sample metadata"})

**Retrieving Data**

.. code-block:: python

    # Retrieve vector and metadata by key
    vector, metadata = db["sample_key"]

    # Check if a key exists in the database
    exists = "sample_key" in db


**Iterating Through Data**

.. code-block:: python

    # Iterate through all keys, vectors and metadata in the database
    for key, vector, metadata in db:
        print(key, metadata)

**Deleting Data**

.. code-block:: python

    # Delete a vector from the database by key
    del db["sample_key"]

**Searching**

.. code-block:: python

    # Search for nearest neighbors of a vector
    results = db.search(vector=np.random.rand(128), k=5)
    for key, vector, distance, metadata in results:
        print(key, distance, metadata)

**Database Length**

.. code-block:: python

    # Get the number of entries in the database
    length = len(db)

License
-------

VectorizeDB is released under the Apache License. For more details, see the LICENSE file included in the package.
