"""Download page API."""

from organism.nursery.download_info import download_info


def test_download_info_defaults():
    info = download_info()
    assert info["product"] == "ORGANISM"
    assert info["version"]
    assert "filename" in info["windows"]
    assert "mirror_github" in info["windows"]
    assert "build_url" in info["windows"]
