"""
Dataset and datasplit management side of the logger.
This repo experiments with different modelling strategies for drug response prediction
    via adaptive screen and learning with small amounts of incremental data,
    therefore being able to cleanly track which data has been used/seen
    by the models would be useful, hence this logging design.
"""

import math


from nbutils.batchie_mono.logger._storage import digest, encode


class DatasetRecords:
    """    
    Manage datasets and their splits, ensuring unique responses and consistent metadata.
    Provides methods to register datasets, resolve response IDs, and create data splits.
    """
    ##TODO maybe this is not the smartest way to keep track of dataset
    ##(maybe dataset should be a frozen artifact and logger references parts of it)
        
    def register_dataset(
        self, 
        records: list[dict], 
        *, 
        description: dict | None = None, 
        cells: dict | None = None, 
        drugs: dict | None = None
    ) -> str:
        """
        Register a new dataset with its responses and associated metadata to db.

        :param records: List of response records, each containing 'cell_id', 'drug_id', 'dose', and 'value'.
        :param description: Optional dataset description.
        :param cells: Optional metadata for cell lines.
        :param drugs: Optional metadata for drugs.
        :return: The unique dataset ID.
        """
        normalized = []
        for row in records:
            if not isinstance(row["cell_id"], str) or not isinstance(
                row["drug_id"], str
            ):
                raise ValueError("cell_id and drug_id must be strings")
            dose, value = float(row["dose"]), float(row["value"])
            if not math.isfinite(dose) or not math.isfinite(value):
                raise ValueError("Only finite, available responses may be registered")
            normalized.append(
                dict(
                    cell_id=row["cell_id"],
                    drug_id=row["drug_id"],
                    dose=dose,
                    value=value,
                )
            )
        normalized.sort(key=lambda r: (r["cell_id"], r["drug_id"], r["dose"]))
        keys = [(r["cell_id"], r["drug_id"], r["dose"]) for r in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate response keys")
        cells, drugs = dict(cells or {}), dict(drugs or {})
        for row in normalized:
            cells.setdefault(row["cell_id"], {})
            drugs.setdefault(row["drug_id"], {})
        if any(not isinstance(k, str) for k in [*cells, *drugs]):
            raise ValueError("Axis identifiers must be strings")
        description = description or {}
        dataset_id = digest([normalized, description, cells, drugs])
        with self.transaction():
            if not self.connection.execute(
                "SELECT 1 FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone():
                self.insert(
                    "datasets",
                    dataset_id=dataset_id,
                    description=encode(description),
                    response_count=len(normalized),
                )
                for axis, items in [("cell", cells), ("drug", drugs)]:
                    self.connection.executemany(
                        "INSERT INTO entities VALUES (?,?,?,?)",
                        [(dataset_id, axis, k, encode(v)) for k, v in items.items()],
                    )
                self.connection.executemany(
                    "INSERT INTO responses VALUES (?,?,?,?,?,?)",
                    [
                        (digest([dataset_id, *key]), dataset_id, *key, row["value"])
                        for key, row in zip(keys, normalized)
                    ],
                )
        return dataset_id

    def response_ids(self, dataset_id: str, keys: list[tuple[str, str, float]]) -> list[str]:
        """
        Resolve (cell, drug, dose) keys, preserving caller order.
        """
        ids = []
        for key in keys:
            key = tuple(key)
            # Retain Python key matching: SQLite may coerce query parameters.
            mapping = {
                (r["cell_id"], r["drug_id"], r["dose"]): r["response_id"]
                for r in self.rows(
                    "SELECT response_id, cell_id, drug_id, dose FROM responses "
                    "WHERE dataset_id=? AND cell_id=? AND drug_id=? AND dose=?",
                    (dataset_id, *key),
                )
            }
            ids.append(mapping[key])
        return ids

    def create_split(
        self, dataset_id: str, *, initial: list[str], pool: list[str], validation: list[str]=(), test: list[str]=(), definition=None
    ) -> str:
        """Tells the db how the master dataset gets split for an experiment"""
        roles = {
            "initial": list(initial),
            "pool": list(pool),
            "validation": list(validation),
            "test": list(test),
        }
        members = [(r, role) for role, ids in roles.items() for r in ids]
        ids = [r for r, _ in members]
        if len(set(ids)) != len(ids):
            raise ValueError("Split roles must be disjoint and contain no duplicates")
        available = {
            r["response_id"]
            for r in self.rows(
                "SELECT response_id FROM responses WHERE dataset_id=?", (dataset_id,)
            )
        }
        if not set(ids) <= available:
            raise ValueError(
                "Split references an unavailable or foreign-dataset response"
            )
        definition = definition or {}
        split_id = digest([dataset_id, sorted(members), definition])
        with self.transaction():
            self.connection.execute(
                "INSERT OR IGNORE INTO splits VALUES (?,?,?)",
                (split_id, dataset_id, encode(definition)),
            )
            self.connection.executemany(
                "INSERT OR IGNORE INTO split_memberships VALUES (?,?,?)",
                [(split_id, r, role) for r, role in members],
            )
        return split_id
