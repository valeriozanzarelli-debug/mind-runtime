"""Insegnamento e curriculum — metodi del baby agent."""
from __future__ import annotations

import time
from typing import Any

from organism.autonomous.baby_types import BabyMoment, _tokens_from_heard, normalize_dialogue_key
from organism.brain.oscillation import inject_wave
from organism.brain.growth import active_ids, wire_coactive
from organism.runtime import OrganismRuntime
from organism.cognition.pathway import wire_dialogue_pathway
from organism.cognition.visual_pathway import wire_visual_association
from organism.drives.curiosity import stimulus_key_from_sensory, stimulus_key_visual_context
from organism.sensory.web_media import vision_hash
from organism.sensory.reading import ReadingChannel
from organism.cognition.tasks import TaskKind
from organism.research.senses import research_human_senses


class BabyTeachingMixin:
    def correct_speech(self, *, heard: str, correct: str) -> dict[str, Any]:
        """Correzione caregiver — penalizza mismatch fonetico."""
        org = self._ensure()
        if not self.speech:
            return {"ok": False}
        from organism.teaching.phonemes import split_italian_syllables

        wrong_syl = split_italian_syllables(heard)
        right_syl = split_italian_syllables(correct)
        penalized = self.speech.phonemes.penalize_mismatch(wrong_syl, right_syl, penalty=0.03)
        reinforced = self.speech.phonemes.reinforce_sequence(right_syl, boost=0.4)
        self.speech.hear(correct, boost=1.0)
        self.composer.absorb(correct, boost=0.8)
        r = self.dialogue.teach(heard.strip()[:80], correct.strip())
        org.perceive({"text": correct})
        org.brain.propagate(steps=1)
        self._persist()
        return {
            "ok": True,
            "penalized": penalized,
            "reinforced": reinforced,
            "dialogue": r,
        }

    def reflect(self, *, prompt: str = "") -> dict[str, Any]:
        """Riflessione autonoma — coscienza decide se pensare o parlare."""
        org = self._ensure()
        self._last_sense_t = time.time()
        heard = prompt.strip() or None
        if heard:
            org.perceive({"text": heard})
            org.brain.propagate(steps=2)
            self.speech.hear(heard, boost=0.7)
            self.composer.absorb(heard, boost=0.5)
        else:
            inject_wave(org.brain, "reflect", tick=int(org.brain.tick), amplitude=0.8)
        impulse = "vocalize"
        spoke, code_out, thought_d, ws_d, mood, _, from_thought, plan, stream = self._act(
            org,
            heard=heard,
            impulse=impulse,
            long_form=bool(heard),
            skey="reflect",
        )
        moment = BabyMoment(
            impulse=impulse,
            spoke=spoke,
            code=code_out or "",
            understood=from_thought or bool(spoke),
            from_thought=from_thought,
            self_heard=False,
            speech_error={},
            consciousness=ws_d,
            consciousness_stream=stream,
            self_state=self.self_model.state.to_dict(),
            wave=self.waves.last.to_dict(),
            dream=self.dream_engine.state.to_dict(),
            emotion=self.affect.state.to_dict(),
            social_tone={},
            presence=self._last_presence,
            thought=thought_d,
            stimulus_key="reflect",
            curiosity=self.curiosity.state.to_dict(),
            learned=True,
            brain=mood.to_dict(),
            wanted_to_speak=False,
            symbols=thought_d.get("symbols", []),
        )
        self._last_moment = moment
        self._persist()
        return {"moment": moment.to_dict(), "stats": org.stats}

    def _is_stuck_repeat(self, candidate: str) -> bool:
        c = candidate.lower().strip()
        if not c or len(c) < 4:
            return False
        hits = sum(1 for s in self._recent_spokes if s and (s == c or c in s or s in c))
        return hits >= 2

    def _anti_repeat_speech(
        self,
        spoke: str,
        *,
        org: OrganismRuntime,
        heard: str | None,
        ask_mode: bool,
    ) -> str:
        if not spoke or ask_mode:
            return spoke
        if not self._is_stuck_repeat(spoke):
            return spoke
        if self._vision_fresh and self.speech:
            alt = self._visual_utterance(org)
            if alt and not self._is_stuck_repeat(alt):
                return alt
        if heard and not ask_mode:
            uw = self.self_learner.detect_unknown(heard, has_pathway=False)
            if uw and self.speech:
                from organism.cognition.thought import Thought

                t = Thought(
                    symbols=["IMPULSE:ask", f"UNKNOWN:{uw[0]}"],
                    themes=["non", "so", uw[0]],
                    pressure=0.55,
                )
                q = self.composer._from_curiosity_question(t, self.speech)
                if q:
                    return q
        return spoke

    def teach_from_web_image(
        self,
        url: str,
        *,
        name: str,
        phrase: str | None = None,
        kind: str = "object",
        image_prefetch: dict[str, Any] | None = None,
        hd_size: int = 256,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Insegna da immagine internet — oggetto, volto o emozione (HD)."""
        from organism.teaching.web_fetch import fetch_image_rgba

        name = name.strip().lower()
        if not url or not name:
            return {"ok": False, "reason": "missing_url_or_name"}
        try:
            img = image_prefetch or fetch_image_rgba(url, size=hd_size)
        except Exception as e:
            return {"ok": False, "reason": str(e)[:120], "url": url[:80]}

        face_data = {}
        grid, sig = self._decode_vision(
            image_gray=img.get("image_gray"),
            image_b64=None,
            image_rgba=img.get("image_rgba"),
            image_w=int(img.get("image_w", hd_size)),
            image_h=int(img.get("image_h", hd_size)),
            color_rgb=img.get("color_rgb"),
        )
        face_data = self._last_visual_features.get("face") or {}
        self.photo_memory.capture(
            object_sig=str(self._last_visual_features.get("object_sig", "")),
            features=self._last_visual_features,
            labels=[name],
            thumb_gray=(img.get("image_gray") or [])[:256],
        )

        say = phrase or f"vedo {name}"
        result: dict[str, Any] = {"ok": True, "name": name, "kind": kind, "url": url[:120]}

        if kind == "emotion":
            emo_r = self.face_binder.teach_emotion(name, face_data)
            say = phrase or f"questa espressione è {name}"
            self.composer.absorb(f"{name} emozione volto {say}", boost=1.1)
            self.dialogue.teach(f"che emozione è", f"è {name}")
            result.update(emo_r)
        elif kind == "face":
            face_r = self.face_binder.teach_face(name, face_data)
            say = phrase or f"questo è un {name}"
            self.composer.absorb(f"volto {name} {say}", boost=1.0)
            result.update(face_r)
        else:
            obj_r = self.teach_object(
                name,
                image_gray=img.get("image_gray"),
                image_rgba=img.get("image_rgba"),
                image_w=int(img.get("image_w", hd_size)),
                image_h=int(img.get("image_h", hd_size)),
                color_rgb=img.get("color_rgb"),
                phrase=say,
            )
            result.update(obj_r)

        if kind in ("face", "emotion") and self.speech:
            org = self._ensure()
            self.speech.hear(say, boost=0.9)
            org.perceive({"text": say})
            org.brain.propagate(steps=1)
            self._wire_from_input(org, had_text=True, had_vision=True, teach=True)

        if persist:
            self._persist()
        result["phrase"] = say
        result["face_detected"] = bool(face_data.get("detected"))
        return result

    def run_web_curriculum(
        self,
        *,
        objects: bool = True,
        emotions: bool = True,
        faces: bool = True,
    ) -> dict[str, Any]:
        """Curriculum automatico da Wikimedia — oggetti, volti, emozioni."""
        from organism.teaching.web_curriculum import run_web_curriculum

        return run_web_curriculum(
            self.teach_from_web_image,
            objects=objects,
            emotions=emotions,
            faces=faces,
        )

    def run_mega_curriculum(self, *, limit: int = 1000, pause_s: float = 0.35) -> dict[str, Any]:
        """Curriculum 1000 oggetti — visione HD."""
        from organism.teaching.mega_curriculum import run_mega_curriculum

        def _teach_batch(**kwargs: Any) -> dict[str, Any]:
            return self.teach_from_web_image(persist=False, **kwargs)

        result = run_mega_curriculum(_teach_batch, limit=limit, pause_s=pause_s)
        self._persist()
        return result

    def run_code_curriculum(self) -> dict[str, Any]:
        """Insegna programmazione — snippet e programmi interi."""
        from organism.teaching.code_curriculum import run_code_curriculum

        return run_code_curriculum(self.teach_dialogue)

    def train_syntax_curriculum(self, *, repeats: int = 3, absorb_limit: int = 250) -> dict[str, Any]:
        """Fase 2 — rinforzo transizioni sintattiche Layer 2."""
        from organism.teaching.syntax_curriculum import curriculum_sentences

        org = self._ensure()
        stats = self.narrator.bootstrap_curriculum(repeats=repeats)
        absorbed = 0
        for sent in curriculum_sentences()[:absorb_limit]:
            self.composer.absorb(sent, boost=0.35)
            if self.speech:
                self.speech.hear(sent, boost=0.25)
            absorbed += 1
        org.brain.propagate(steps=1)
        if org.brain.plasticity:
            org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
        self._persist()
        return {
            "ok": True,
            "syntax": stats,
            "absorbed": absorbed,
            "narrator": self.narrator.stats(),
        }

    def run_fluency_benchmark(self, *, limit: int = 100, pause_s: float = 0.12) -> dict[str, Any]:
        """Benchmark 100 prompt — fluenza, coerenza, SVO."""
        from organism.teaching.fluency_benchmark import run_benchmark

        def _sense(prompt: str) -> dict[str, Any]:
            return self.sense(text=prompt)

        result = run_benchmark(_sense, limit=limit, pause_s=pause_s)
        self._persist()
        return result

    def run_dog_curriculum(self, *, limit: int = 50, pause_s: float = 0.3) -> dict[str, Any]:
        """Training visivo cane — pose e contesti diversi."""
        from organism.teaching.dog_curriculum import run_dog_curriculum

        def _teach(**kwargs: Any) -> dict[str, Any]:
            return self.teach_from_web_image(persist=False, **kwargs)

        result = run_dog_curriculum(_teach, limit=limit, pause_s=pause_s)
        self._persist()
        return result

    def train_integrated_cycle(self, cycle: int) -> dict[str, Any]:
        """Un passo training Fase 3 — storie, dialoghi, motor loop, oggetti."""
        import random

        from organism.teaching.corpus import REASONING, PHILOSOPHY, STORIES_EXTENDED
        from organism.teaching.story_curriculum import dialogue_chains, long_story_chunks

        org = self._ensure()
        rng = random.Random(self.seed + cycle)
        log: dict[str, Any] = {"cycle": cycle, "actions": []}

        if cycle == 1 or cycle % 500 == 0:
            r = self.train_syntax_curriculum(repeats=3, absorb_limit=200)
            log["actions"].append({"syntax": r.get("syntax", {})})

        if cycle % 10 == 0:
            chunks = long_story_chunks()
            chunk = chunks[cycle % len(chunks)]
            self.read(chunk)
            self.self_hear(text=chunk[:120])
            log["actions"].append({"story_chunk": chunk[:60]})

        if cycle % 25 == 0:
            pairs = STORIES_EXTENDED + REASONING + PHILOSOPHY
            when, say = pairs[cycle % len(pairs)]
            self.teach_dialogue(when, say)
            log["actions"].append({"dialogue": when[:40]})

        if cycle % 40 == 0:
            when, say = dialogue_chains()[cycle % len(dialogue_chains())]
            self.teach_dialogue(when, say)
            for _ in range(2):
                m = self.sense(text=when).get("moment") or {}
                spoke = str(m.get("spoke", ""))
                if spoke:
                    self.self_hear(text=spoke)
            log["actions"].append({"dialogue_chain": when})

        if cycle % 100 == 0:
            dog = self.run_dog_curriculum(limit=20, pause_s=0.2)
            log["actions"].append({"dog": dog.get("taught", 0)})

        if cycle % 200 == 0:
            mega = self.run_mega_curriculum(limit=30, pause_s=0.2)
            log["actions"].append({"mega": mega.get("taught", 0)})

        if cycle % 300 == 0:
            self.sleep_cycle()
            log["actions"].append({"sleep": True})

        # Ogni ciclo: stimolo casuale + motor loop chiuso
        probes = ["cosa pensi", "cosa vedi", "chi sei", "perché impari", "cosa provi"]
        q = rng.choice(probes)
        m = self.sense(text=q).get("moment") or {}
        spoke = str(m.get("spoke", ""))
        if spoke and len(spoke) > 8:
            self.self_hear(text=spoke)
        log["probe"] = q
        log["spoke_words"] = len(spoke.split())

        org.brain.propagate(steps=1)
        if cycle % 50 == 0:
            self._persist()
        return log

    def browse_web(self, url: str) -> dict[str, Any]:
        """Legge una pagina web e assorbe conoscenza."""
        from organism.cognition.browser_agent import fetch_page_text

        org = self._ensure()
        page = fetch_page_text(url)
        if not page.get("ok"):
            return page
        text = str(page.get("text", ""))[:4000]
        self.read(text)
        for kw in page.get("keywords", [])[:20]:
            self.composer.absorb(kw, boost=0.35)
        self.episodic_memory.record(
            heard=f"browse {url[:80]}",
            spoke="",
            themes=page.get("keywords", [])[:8],
            objects=[],
            emotion="",
        )
        org.brain.propagate(steps=1)
        self._persist()
        return {
            **page,
            "absorbed_words": len(page.get("keywords", [])),
            "title": page.get("title", ""),
        }

    def teach_object(
        self,
        name: str,
        *,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
        phrase: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Insegna nome oggetto con vista — forma+colore associati all'immagine."""
        org = self._ensure()
        name = name.strip().lower()
        if not name:
            return {"learned": False}
        grid, sig = self._decode_vision(
            image_gray=image_gray,
            image_b64=image_b64,
            image_w=image_w,
            image_h=image_h,
            color_rgb=color_rgb,
            image_rgba=image_rgba,
        )
        if not grid:
            return {"learned": False, "reason": "no_image"}
        self._last_vision_hash = sig
        self._vision_fresh = True
        org.perceive({"image": grid, "width": image_w, "height": image_h})
        org.brain.propagate(steps=2)
        color = str(self._last_visual_features.get("color", ""))
        if phrase:
            say = phrase
        elif color and color not in ("colore", "grigio"):
            say = f"vedo un {name} {color}"
        else:
            say = f"vedo un {name}"
        obj_sig = str(self._last_visual_features.get("object_sig", sig))
        self.visual_binder.bind(
            sig,
            say,
            boost=1.4,
            features=self._last_visual_features,
            object_sig=obj_sig,
            object_name=name,
        )
        self.photo_memory.capture(
            object_sig=obj_sig,
            features=self._last_visual_features,
            labels=[name, color] if color else [name],
            thumb_gray=(image_gray or [])[:256] if image_gray else None,
        )
        skey = stimulus_key_visual_context(vision_hash=sig)
        self.speech.hear(say, boost=1.2)
        org.perceive({"text": say})
        org.brain.propagate(steps=2)
        grown = self._wire_from_input(org, had_text=True, had_vision=True, teach=True)
        grown += wire_visual_association(org.brain, self.speech, name=name, color=color, boost=0.65)
        if org.brain.plasticity:
            org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
        result = self.teacher.teach(skey, say)
        self.composer.absorb(f"{name} {color} {say}".strip(), boost=1.2)
        consolidated = self.visual_binder._object_names.get(obj_sig) == name
        if consolidated and color and color not in ("colore", "grigio"):
            color_phrase = f"è {color}"
            for _ in range(2):
                self.dialogue.teach(f"che colore è il {name}", color_phrase)
                wire_dialogue_pathway(org.brain, self.speech, when=f"che colore è il {name}", say=color_phrase)
                self.composer.absorb(color_phrase, boost=0.9)
        trials = self.visual_binder._object_trials.get(obj_sig, 0)
        if persist:
            self._persist()
        return {
            **result,
            "name": name,
            "object_sig": obj_sig,
            "consolidated": consolidated,
            "trials": trials,
            "confidence": round(self.visual_binder.confidence(self._last_visual_features, object_sig=obj_sig), 2),
            "symbols": self._last_visual_features.get("symbols", []),
            "stimulus_key": skey,
            "new_wires": grown,
        }

    def teach_repetition(
        self,
        phrase: str,
        *,
        stimulus_key: str | None = None,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
    ) -> dict[str, Any]:
        org = self._ensure()
        if image_gray or image_b64:
            grid, sig = self._decode_vision(
                image_gray=image_gray,
                image_b64=image_b64,
                image_w=image_w,
                image_h=image_h,
            )
            self._last_vision_hash = sig or vision_hash(grid)
            org.perceive({"image": grid, "width": image_w, "height": image_h})
            org.brain.propagate(steps=1)
            self.visual_binder.bind(
                self._last_vision_hash,
                phrase,
                features=self._last_visual_features,
                object_sig=str(self._last_visual_features.get("object_sig", "")),
            )
        skey = stimulus_key or self._context_stimulus_key()
        if not skey:
            skey = stimulus_key_from_sensory(text=phrase)
        self.speech.hear(phrase, boost=1.3)
        org.perceive({"text": phrase})
        org.brain.propagate(steps=2)
        grown = self._wire_from_input(
            org,
            had_text=True,
            had_vision=bool(image_gray or image_b64),
            teach=True,
        )
        if org.brain.plasticity:
            org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
        result = self.teacher.teach(skey, phrase)
        if result.get("learned"):
            from mind.types import Fragment

            org.memory.add(
                Fragment(
                    id=f"taught_{skey}",
                    title=phrase,
                    weight=0.7,
                    sensation_id="taught",
                    hooks=phrase.lower().split()[:6],
                    functions=["learned"],
                )
            )
        self._persist()
        return {
            **result,
            "stimulus_key": skey,
            "syllables_heard": self.speech.phonemes.count,
            "synapses": org.brain.synapse_count,
            "synapses_grown": org.brain.synapse_count - self._synapses_at_birth,
            "new_wires": grown,
        }

    def teach_dialogue(
        self,
        when: str,
        say: str,
        *,
        kind: str = "speech",
    ) -> dict[str, Any]:
        """Insegna: quando senti «when» → rispondi «say» (dialogo o codice)."""
        org = self._ensure()
        when = when.strip()
        say = say.strip()
        self.speech.hear(when, boost=0.6)
        self.speech.hear(say, boost=1.0)
        self.composer.absorb(when, boost=0.8)
        self.composer.absorb(say, boost=1.2)
        org.perceive({"text": f"{when} → {say}"})
        org.brain.propagate(steps=2)
        grown = self._wire_from_input(org, had_text=True, had_vision=False, teach=True)
        for w in list(self.self_learner.gaps()):
            if w in when.lower() or w in say.lower():
                self.self_learner.note_learned(w)
        if kind == "code":
            self.code.learn_snippet(when, say)
            post = [n.id for n in org.brain.get_neurons("motor", "text_generator")][:8]
            if not post:
                post = [n.id for n in org.brain.get_neurons("motor", "speech_phoneme_generator")][:8]
            pre = active_ids(org.brain, "sensory", "text_semantic_encoder")
            wire_coactive(org.brain, pre[:10], post, max_new=4)
        if org.brain.plasticity:
            org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
        wire_dialogue_pathway(org.brain, self.speech, when=when, say=say)
        if self._last_vision_hash:
            self.visual_binder.bind(self._last_vision_hash, f"{when} {say}")
        result = self.dialogue.teach(when, say, kind=kind)  # type: ignore[arg-type]
        if result.get("learned"):
            from mind.types import Fragment

            org.memory.add(
                Fragment(
                    id=f"dialogue_{normalize_dialogue_key(when)}",
                    title=f"{when} → {say}",
                    weight=0.75,
                    sensation_id="dialogue",
                    hooks=when.lower().split()[:6],
                    functions=["dialogue", kind],
                )
            )
        self._persist()
        return {
            **result,
            "synapses": org.brain.synapse_count,
            "synapses_grown": org.brain.synapse_count - self._synapses_at_birth,
            "new_wires": grown,
            "dialogue_pairs": len(self.dialogue.all_pairs()),
            "words_known": self.composer.lexicon.count,
        }

    def read(self, text: str) -> dict[str, Any]:
        """Lettura autonoma — testo come flusso sensoriale, impara da solo."""
        org = self._ensure()
        self._last_sense_t = time.time()
        if self._reading is None:
            self._reading = ReadingChannel(org.brain)
        result = self._reading.perceive(text)
        self.composer.absorb(text, boost=0.45)
        if self.speech:
            self.speech.hear(text, boost=0.5)
        org.perceive({"text": text[:400]})
        org.brain.propagate(steps=2)
        self._wire_from_input(org, had_text=True, had_vision=False)
        active = self.tasks.active()
        if active and active.kind == "read_aloud":
            self.tasks.evaluate_attempt(text[:120])
        self._persist()
        return {
            "reading": result.to_dict(),
            "words_known": self.composer.lexicon.count,
            "task": self.tasks.active().to_dict() if self.tasks.active() else None,
        }

    def assign_task(self, kind: TaskKind, prompt: str, target: str) -> dict[str, Any]:
        """Assegna un compito — responsabilità e portare a termine."""
        self._ensure()
        t = self.tasks.assign(kind, prompt, target)
        self._persist()
        return {"task": t.to_dict(), "tasks": self.tasks.to_dict()}

    def absorb_vocabulary(self, texts: list[str], *, boost: float = 1.0) -> dict[str, Any]:
        """Assorbe testi nel lessico — migliaia di parole senza coppie dialogo."""
        org = self._ensure()
        before = self.composer.lexicon.count
        n = 0
        for raw in texts:
            t = raw.strip()
            if not t:
                continue
            self.composer.absorb(t, boost=boost)
            if self.speech:
                self.speech.hear(t, boost=0.35)
            n += 1
        if n:
            sample = " ".join(texts[:3])[:240]
            org.perceive({"text": sample})
            org.brain.propagate(steps=1)
            if org.brain.plasticity:
                org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
        self._persist()
        return {
            "absorbed": n,
            "words_known": self.composer.lexicon.count,
            "words_new": self.composer.lexicon.count - before,
        }

    def research_once(self) -> dict[str, Any]:
        org = self._ensure()
        if self._researched:
            return {"already": True, "topics": []}
        topics_cfg = org.dna.genome.get("baby", {}).get("research_topics", [])
        topics = research_human_senses(org.memory, topics=topics_cfg, lang="it")
        self._researched = True
        self._persist()
        return {"researched": True, "topics": topics}
