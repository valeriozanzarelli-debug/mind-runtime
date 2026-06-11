"""Fragmented memory graph with cue-dependent retrieval."""

from __future__ import annotations

from mind.types import Cue, CueKind, Fragment, HiddenDetail


class MemoryGraph:
    def __init__(self, fragments: list[Fragment] | None = None) -> None:
        self._fragments: dict[str, Fragment] = {}
        for f in fragments or []:
            self.add(f)

    def add(self, fragment: Fragment) -> None:
        self._fragments[fragment.id] = fragment

    def get(self, fragment_id: str) -> Fragment | None:
        return self._fragments.get(fragment_id)

    def all_fragments(self) -> list[Fragment]:
        return list(self._fragments.values())

    def retrieve(self, cue: Cue, *, limit: int = 8, spread: int = 1) -> list[Fragment]:
        """Spreading activation from cue tokens; hidden details gate extra links."""
        tokens = _tokenize_cue(cue)
        if not tokens:
            return []

        scores: dict[str, float] = {}
        for fid, frag in self._fragments.items():
            s = max((frag.matches_hook(t) for t in tokens), default=0.0)
            if s > 0:
                scores[fid] = s * (0.5 + 0.5 * frag.weight)

        # one hop spread via links (only if parent activated)
        for _ in range(spread):
            extra: dict[str, float] = {}
            for fid, base in list(scores.items()):
                frag = self._fragments[fid]
                for link_id in frag.links:
                    linked = self._fragments.get(link_id)
                    if linked is None:
                        continue
                    if not _hidden_allows(cue, linked):
                        continue
                    extra[link_id] = max(extra.get(link_id, 0.0), base * 0.55 * linked.weight)
            for k, v in extra.items():
                scores[k] = max(scores.get(k, 0.0), v)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self._fragments[fid] for fid, sc in ranked[:limit] if sc > 0.15]

    def compress_title(self, function: str) -> Fragment | None:
        """Best compressed title for a function (highest weight)."""
        matches = [f for f in self._fragments.values() if function in f.functions]
        if not matches:
            return None
        return max(matches, key=lambda f: f.weight)


def _tokenize_cue(cue: Cue) -> list[str]:
    if cue.kind == CueKind.VISUAL:
        return [cue.value, *cue.value.replace("+", " ").split()]
    if cue.kind == CueKind.FUNCTION:
        return [cue.value]
    raw = cue.value.lower().replace(",", " ").replace("?", " ")
    return [t for t in raw.split() if len(t) > 2]


def _hidden_allows(cue: Cue, fragment: Fragment) -> bool:
    """Fragments with hidden_details only fully open from matching cues."""
    if not fragment.hidden_details:
        return True
    text = cue.value.lower()
    for hd in fragment.hidden_details:
        if any(tok in text for tok in hd.retrieve_only_if_cue_contains):
            return True
    # still allow if directly hooked (not only via hidden link)
    return False


def merge_episodes_into_title(
    episode_ids: list[str],
    graph: MemoryGraph,
    *,
    new_title: str,
    function: str,
    weight: float,
    sensation_id: str,
    hooks: list[str],
) -> Fragment:
    """Simulate 30 bulb changes → one compressed title fragment."""
    merged = Fragment(
        id=f"compressed_{function}",
        title=new_title,
        weight=weight,
        sensation_id=sensation_id,
        hooks=hooks,
        functions=[function],
        links=episode_ids,
    )
    graph.add(merged)
    return merged
