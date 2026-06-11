from organism.cognition.working_memory import WorkingMemory


def test_parallel_streams():
    wm = WorkingMemory(capacity=7)
    wm.activate(["codice", "funzione"], weight=1.0, link_to=["codice"])
    wm.activate(["texture", "gpu"], weight=1.0, link_to=["texture"])
    streams = wm.parallel_streams()
    assert len(streams) >= 2
    flat = {w for s in streams for w in s}
    assert "codice" in flat or "funzione" in flat
    assert "texture" in flat or "gpu" in flat
