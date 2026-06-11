"""Persistenza stato Baby — continua ad imparare tra sessioni e riavvii."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from organism.learning.store import build_organism_payload, restore_organism_payload


def _finalize_state_file(path: Path) -> None:
    """Nursery runs as www-data — keep state readable after root-owned training writes."""
    try:
        os.chmod(path, 0o664)
        import grp

        www = grp.getgrnam("www-data")
        uid = path.stat().st_uid
        os.chown(path, uid, www.gr_gid)
    except (KeyError, OSError, ImportError):
        try:
            os.chmod(path, 0o664)
        except OSError:
            pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp.replace(path)
        _finalize_state_file(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def baby_state_path() -> Path:
    if os.environ.get("ORGANISM_BABY_STATE"):
        return Path(os.environ["ORGANISM_BABY_STATE"])
    server_default = Path("/opt/mind-runtime/data/baby_state.json")
    if server_default.parent.parent.exists():
        return server_default
    return Path.home() / ".organism" / "baby_state.json"


class BabyStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else baby_state_path()

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, agent: Any) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        org = agent.org
        organism_blob: dict[str, Any] = {}
        if org is not None:
            organism_blob = build_organism_payload(
                org.brain, org.memory, org.learner, meta=org.stats
            )

        payload = {
            "version": 1,
            "seed": agent.seed,
            "born": agent._born,
            "dormant": agent._dormant,
            "researched": agent._researched,
            "synapses_at_birth": agent._synapses_at_birth,
            "last_vision_hash": agent._last_vision_hash,
            "last_audio_hash": agent._last_audio_hash,
            "teacher": agent.teacher.to_dict(),
            "dialogue": agent.dialogue.to_dict(),
            "code_tokens": agent.code.tokens.to_dict(),
            "neural_lexicon": agent.composer.lexicon.to_dict(),
            "word_weights": agent.composer.lexicon.to_dict(),
            "speech_loop": agent.speech_loop.stats(),
            "workspace": agent.workspace.stats(),
            "affect": agent.affect.stats(),
            "amygdala": agent.amygdala.stats(),
            "presence": {
                **agent.presence.to_dict(),
                "familiar_scenes": list(agent.presence._familiar_scenes),
                "last_spoke_t": agent.presence._last_spoke_t,
                "caregiver_until": agent.presence._caregiver_until,
                "state": agent.presence.state.to_dict(),
            },
            "corrections": agent.corrections.to_dict(),
            "visual_binder": agent.visual_binder.to_dict(),
            "face_binder": agent.face_binder.to_dict(),
            "self_learner": agent.self_learner.to_dict(),
            "self_model": agent.self_model.stats(),
            "dream": agent.dream_engine.stats(),
            "waves": agent.waves.stats(),
            "tasks": agent.tasks.to_dict(),
            "brain_growth": agent.brain_growth.stats(),
            "working_memory": agent.working_memory.to_dict(),
            "photo_memory": agent.photo_memory.to_dict(),
            "episodic_memory": agent.episodic_memory.to_dict(),
            "narrator": agent.narrator.to_dict(),
            "goal_stack": agent.goal_stack.to_dict(),
            "hypothesis_engine": agent.hypothesis_engine.to_dict(),
            "thought_generator": agent.thought_generator.stats(),
            "consciousness_log": agent._consciousness_log[-120:],
            "consciousness_seq": agent._consciousness_seq,
            "dialogue_log": agent._dialogue_log[-80:],
            "visual_features": agent._last_visual_features,
            "phonemes": agent.speech.phonemes.to_dict() if agent.speech else {},
            "curiosity": _curiosity_to_dict(agent.curiosity),
            "journal": agent.journal.recent(40),
            "birth_record": agent.journal.birth_record,
            "organism": organism_blob,
        }
        _atomic_write_json(self.path, payload)
        return self.path

    def load(self, agent: Any) -> dict[str, Any]:
        if not self.path.exists():
            return {"loaded": False}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not payload.get("born"):
            return {"loaded": False, "reason": "never born"}

        from organism.motor.emergent_speech import EmergentSpeechMotor
        from organism.runtime import OrganismRuntime

        agent.seed = int(payload.get("seed", agent.seed))
        agent.org = OrganismRuntime.baby(seed=agent.seed)
        if not agent._synapses_at_birth:
            agent._synapses_at_birth = agent.org.brain.synapse_count
        org_blob = payload.get("organism", {})
        if org_blob:
            restore_organism_payload(org_blob, agent.org.brain, agent.org.memory, agent.org.learner)

        agent.speech = EmergentSpeechMotor(agent.org.brain, seed=agent.seed)
        agent.composer.bind(agent.org.brain, agent.speech)
        agent.teacher.load_dict(payload.get("teacher", {}))
        agent.dialogue.load_dict(payload.get("dialogue", {}))
        agent.code.tokens.load_dict(payload.get("code_tokens", {}))
        for pair in payload.get("dialogue", {}).get("learned", []):
            if pair.get("kind") == "code":
                agent.code.learn_snippet(pair["when"], pair["say"])
        agent.speech.phonemes.load_dict(payload.get("phonemes", {}))
        agent.composer.lexicon.load_dict(
            payload.get("neural_lexicon") or payload.get("word_weights", {})
        )
        agent.composer.lexicon.squash_overexposed()
        agent.working_memory.load_dict(payload.get("working_memory", {}))
        agent.photo_memory.load_dict(payload.get("photo_memory", {}))
        agent.episodic_memory.load_dict(payload.get("episodic_memory", {}))
        if payload.get("narrator"):
            agent.narrator.load_dict(payload["narrator"])
        if payload.get("goal_stack"):
            agent.goal_stack.load_dict(payload["goal_stack"])
        if payload.get("hypothesis_engine"):
            agent.hypothesis_engine.load_dict(payload["hypothesis_engine"])
        _curiosity_from_dict(agent.curiosity, payload.get("curiosity", {}))
        sl = payload.get("speech_loop", {})
        agent.speech_loop._babble_cycles = int(sl.get("babble_cycles", 0))
        agent.speech_loop._corrections = int(sl.get("corrections", 0))
        agent.speech_loop._last_spoke_text = str(sl.get("last_spoke", ""))
        ws = payload.get("workspace", {})
        agent.workspace._ignitions = int(ws.get("ignitions", 0))
        agent.workspace._last_focus = str(ws.get("last_focus", ""))
        agent._last_vision_hash = str(payload.get("last_vision_hash", ""))
        agent._last_visual_features = dict(payload.get("visual_features", {}))
        agent.affect.load_dict(payload.get("affect", {}))
        agent.amygdala.load_dict(payload.get("amygdala", {}))
        pres = payload.get("presence", {})
        if pres:
            agent.presence.load_dict(pres)
            agent.presence._familiar_scenes = set(pres.get("familiar_scenes", []))
        agent.corrections.load_dict(payload.get("corrections", {}))
        agent.visual_binder.load_dict(payload.get("visual_binder", {}))
        agent.face_binder.load_dict(payload.get("face_binder", {}))
        agent.self_learner.load_dict(payload.get("self_learner", {}))
        agent.self_model.load_dict(payload.get("self_model", {}))
        agent.dream_engine.load_dict(payload.get("dream", {}))
        ws = payload.get("waves", {})
        agent.waves._tick = int(ws.get("tick", 0))
        agent.waves._cycles = int(ws.get("cycles", 0))
        agent.waves._idx = int(ws.get("idx", 0))
        agent.tasks.load_dict(payload.get("tasks", {}))
        agent.brain_growth.load_dict(payload.get("brain_growth", {}))
        agent._consciousness_log = list(payload.get("consciousness_log", []))
        agent._consciousness_seq = int(payload.get("consciousness_seq", len(agent._consciousness_log)))
        agent._dialogue_log = list(payload.get("dialogue_log", []))

        agent._born = True
        agent._dormant = bool(payload.get("dormant", False))
        agent._apply_cognition_config()
        if agent._dormant:
            agent.thought_generator.stop()
        else:
            agent._start_thought_loop()
        agent._researched = bool(payload.get("researched", False))
        agent._synapses_at_birth = int(payload.get("synapses_at_birth", 0))
        if agent.org and agent._synapses_at_birth > agent.org.brain.synapse_count:
            agent._synapses_at_birth = agent.org.brain.synapse_count
        agent._last_vision_hash = str(payload.get("last_vision_hash", ""))
        agent._last_audio_hash = str(payload.get("last_audio_hash", ""))
        agent.journal.birth_record = payload.get("birth_record")
        agent.journal.entries.clear()
        for row in payload.get("journal", []):
            agent.journal.entries.append(_journal_entry_from_dict(row))

        return {
            "loaded": True,
            "syllables": agent.speech.phonemes.count,
            "phrases": len(agent.teacher.all_learned()),
            "stimuli_seen": len(agent.curiosity.state.seen_stimuli),
        }


def _curiosity_to_dict(drive: Any) -> dict[str, Any]:
    s = drive.state
    return {
        "level": s.level,
        "novelty": s.novelty,
        "uncertainty": s.uncertainty,
        "boredom": s.boredom,
        "last_stimulus_key": s.last_stimulus_key,
        "seen_stimuli": list(s.seen_stimuli),
        "last_activity_t": s.last_activity_t,
    }


def _curiosity_from_dict(drive: Any, data: dict[str, Any]) -> None:
    s = drive.state
    s.level = float(data.get("level", s.level))
    s.novelty = float(data.get("novelty", s.novelty))
    s.uncertainty = float(data.get("uncertainty", s.uncertainty))
    s.boredom = float(data.get("boredom", s.boredom))
    s.last_stimulus_key = str(data.get("last_stimulus_key", ""))
    s.seen_stimuli = set(data.get("seen_stimuli", []))
    s.last_activity_t = float(data.get("last_activity_t", s.last_activity_t))


def _journal_entry_from_dict(d: dict[str, Any]) -> Any:
    from organism.nursery.journal import ThoughtEntry

    return ThoughtEntry(
        cycle=int(d.get("cycle", 0)),
        phase=str(d.get("phase", "")),
        timestamp=float(d.get("timestamp", 0)),
        input_summary=str(d.get("input", "")),
        thoughts=list(d.get("thoughts", [])),
        mind_action=d.get("action"),
        mind_fragments=list(d.get("fragments", [])),
        expression=str(d.get("expression", "")),
        learning=d.get("learning"),
        brain=dict(d.get("brain", {})),
        new_fragment=d.get("new_fragment"),
    )
