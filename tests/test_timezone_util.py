"""Fuso orario — compatibile Windows .exe."""

from organism.timezone_util import TZ, organism_timezone


def test_organism_timezone():
    tz = organism_timezone()
    assert tz is not None
    assert TZ is not None
