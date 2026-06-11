"""Visual-symbolic pattern engine — schema extraction and gap filling."""

from __future__ import annotations

from mind.types import Schema, VisualShape


class PatternEngine:
    def __init__(self, schemas: list[Schema] | None = None) -> None:
        self._schemas = list(schemas or [])

    def add_schema(self, schema: Schema) -> None:
        self._schemas.append(schema)

    def infer_schema(self, examples: list[VisualShape], schema_id: str = "inferred") -> Schema | None:
        """From examples, infer inner invariant if consistent."""
        with_inner = [e for e in examples if e.inner is not None]
        if len(with_inner) < 2:
            return None
        inners = {e.inner for e in with_inner}
        if len(inners) != 1:
            return None
        inner = next(iter(inners))
        confidence = min(1.0, len(with_inner) / max(len(examples), 1))
        schema = Schema(
            id=schema_id,
            inner_invariant=inner,
            examples=with_inner,
            confidence=confidence,
        )
        self._schemas.append(schema)
        return schema

    def complete_sequence(self, shapes: list[VisualShape]) -> str | None:
        """
        Given a sequence where the last shape may lack inner symbol,
        return what should fill the gap (e.g. circle).
        """
        if not shapes:
            return None

        # try known schemas first
        for schema in self._schemas:
            fill = self._fill_with_schema(shapes, schema)
            if fill:
                return fill

        # infer on the fly from prior items in sequence
        prior = shapes[:-1]
        target = shapes[-1]
        if target.inner is not None:
            return None
        schema = self.infer_schema(prior, schema_id="seq_inferred")
        if schema is None:
            return None
        return schema.predict_inner(target.outer)

    def _fill_with_schema(self, shapes: list[VisualShape], schema: Schema) -> str | None:
        if len(shapes) < 2:
            return None
        target = shapes[-1]
        if target.inner is not None:
            return None
        # prior items should match schema pattern
        prior = shapes[:-1]
        matched = 0
        for p in prior:
            pred = schema.predict_inner(p.outer)
            if pred and p.inner == pred:
                matched += 1
        if matched < max(1, len(prior) - 1):
            return None
        return schema.predict_inner(target.outer)

    def detect_gap(self, shapes: list[VisualShape]) -> bool:
        if not shapes:
            return False
        return shapes[-1].inner is None and any(s.inner is not None for s in shapes[:-1])
