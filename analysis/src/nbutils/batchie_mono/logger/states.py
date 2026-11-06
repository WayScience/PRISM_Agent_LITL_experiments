"""
Experiment lifecycle, state ancestry, action reservation/completion
"""

from nbutils.batchie_mono.logger._storage import encode, now, uid


class StateRecords:
    def start_experiment(
        self, name, split_id, *, model_family, config=None, experiment_id=None
    ):
        """Create an initial data state even for zero-round experiments.

        Replicates share dataset/split IDs but always have distinct experiment IDs.
        Initial model_state_id is NULL, explicitly meaning no saved model yet.
        """
        if not name or not model_family:
            raise ValueError("Experiment name and model family are required")
        eid, did, sid = experiment_id or uid(), uid(), uid()
        with self.transaction():
            self.insert(
                "experiments",
                experiment_id=eid,
                name=name,
                split_id=split_id,
                model_family=model_family,
                config=encode(config or {}),
                status="running",
                created_at=now(),
            )
            self.insert(
                "data_states", data_state_id=did, experiment_id=eid, parent_id=None
            )
            self.connection.execute(
                "INSERT INTO reveals SELECT ?,response_id FROM split_memberships WHERE split_id=? AND role='initial'",
                (did, split_id),
            )
            self.insert(
                "experiment_states",
                state_id=sid,
                experiment_id=eid,
                sequence=0,
                model_state_id=None,
                data_state_id=did,
            )
            self.connection.execute(
                "UPDATE experiments SET current_state_id=? WHERE experiment_id=?",
                (sid, eid),
            )
        return eid

    def current_state(self, experiment_id):
        return self.one(
            "SELECT s.* FROM experiments e JOIN experiment_states s ON s.state_id=e.current_state_id WHERE e.experiment_id=?",
            (experiment_id,),
        )

    def revealed_ids(self, data_state_id):
        """Reconstruct the reveal set from ancestry; no dense mask copies."""
        self.one(
            "SELECT data_state_id FROM data_states WHERE data_state_id=?",
            (data_state_id,),
        )
        return {
            r["response_id"]
            for r in self.rows(
                """
            WITH RECURSIVE ancestry(id,parent_id) AS (
                SELECT data_state_id,parent_id FROM data_states WHERE data_state_id=?
                UNION ALL SELECT d.data_state_id,d.parent_id FROM data_states d JOIN ancestry a ON d.data_state_id=a.parent_id
            ) SELECT response_id FROM reveals JOIN ancestry ON data_state_id=id""",
                (data_state_id,),
            )
        }

    def candidate_ids(self, experiment_id, *, state_id=None):
        state = (
            self.current_state(experiment_id)
            if state_id is None
            else self.one(
                "SELECT * FROM experiment_states WHERE state_id=? AND experiment_id=?",
                (state_id, experiment_id),
            )
        )
        pool = {
            r["response_id"]
            for r in self.rows(
                """SELECT m.response_id FROM split_memberships m
            JOIN experiments e USING(split_id) WHERE e.experiment_id=? AND m.role='pool' """,
                (experiment_id,),
            )
        }
        return pool - self.revealed_ids(state["data_state_id"])

    def start_action(self, experiment_id, kind, *, config=None, key=None):
        """Reserve the current state. An explicit key makes registration retryable.

        Reusing a key requires identical kind/config. A completed action can be
        inspected/reused; failed actions require a new key for a new attempt.
        """
        if not kind:
            raise ValueError("Action kind is required")
        key, payload = key or uid(), encode(config or {})
        with self.transaction():
            prior = self.connection.execute(
                "SELECT * FROM actions WHERE experiment_id=? AND idempotency_key=?",
                (experiment_id, key),
            ).fetchone()
            if prior is not None:
                if prior["kind"] != kind or prior["config"] != payload:
                    raise ValueError(
                        "Idempotency key reused with different action contents"
                    )
                return prior["action_id"]
            exp = self.one(
                "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
            )
            if exp["status"] != "running":
                raise ValueError("Experiment is not running")
            aid = uid()
            self.insert(
                "actions",
                action_id=aid,
                experiment_id=experiment_id,
                kind=kind,
                input_state_id=exp["current_state_id"],
                status="running",
                config=payload,
                idempotency_key=key,
                started_at=now(),
            )
        return aid

    def _active(self, action_id, kind=None):
        action = self.one("SELECT * FROM actions WHERE action_id=?", (action_id,))
        state = self.current_state(action["experiment_id"])
        if (
            action["status"] != "running"
            or action["input_state_id"] != state["state_id"]
        ):
            raise ValueError("Action is not active on the current state")
        if kind is not None and action["kind"] != kind:
            raise ValueError("Wrong action kind")
        return action, state

    def _advance(self, action, state, *, model_state_id, data_state_id):
        sid = uid()
        self.insert(
            "experiment_states",
            state_id=sid,
            experiment_id=action["experiment_id"],
            sequence=state["sequence"] + 1,
            model_state_id=model_state_id,
            data_state_id=data_state_id,
        )
        self.connection.execute(
            "UPDATE actions SET status='completed',output_state_id=?,finished_at=? WHERE action_id=?",
            (sid, now(), action["action_id"]),
        )
        self.connection.execute(
            "UPDATE experiments SET current_state_id=? WHERE experiment_id=?",
            (sid, action["experiment_id"]),
        )
        return sid

    def fail_action(self, action_id, error, *, interrupted=False):
        """Record failure without publishing partial output states."""
        with self.transaction():
            self.connection.execute(
                "UPDATE actions SET status=?,error=?,finished_at=? WHERE action_id=? AND status='running'",
                (
                    "interrupted" if interrupted else "failed",
                    str(error),
                    now(),
                    action_id,
                ),
            )

    def finish_experiment(self, experiment_id, *, error=None):
        """Finish a controller invocation; failures also close its in-flight action."""
        with self.transaction():
            self.one(
                "SELECT experiment_id FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            )
            active = self.connection.execute(
                "SELECT 1 FROM actions WHERE experiment_id=? AND status='running'",
                (experiment_id,),
            ).fetchone()
            if active and error is None:
                raise ValueError("Cannot finish with an active action")
            if error is not None:
                self.connection.execute(
                    "UPDATE actions SET status='failed',error=?,finished_at=? WHERE experiment_id=? AND status='running'",
                    (str(error), now(), experiment_id),
                )
            self.connection.execute(
                "UPDATE experiments SET status=?,error=?,finished_at=? WHERE experiment_id=?",
                (
                    "completed" if error is None else "failed",
                    None if error is None else str(error),
                    now(),
                    experiment_id,
                ),
            )
