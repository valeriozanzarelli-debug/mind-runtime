# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — ORGANISM Windows .exe"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent

block_cipher = None

added_datas = [
    (str(ROOT / "organism" / "nursery" / "static"), "organism/nursery/static"),
    (str(ROOT / "organism" / "dna"), "organism/dna"),
    (str(ROOT / "data" / "dna"), "data/dna"),
]

hiddenimports = [
    "organism",
    "organism.nursery",
    "organism.nursery.server",
    "organism.autonomous.baby_agent",
    "organism.brain",
    "organism.cognition",
    "organism.dna",
    "organism.motor",
    "organism.sensory",
    "mind",
    "yaml",
    "PIL",
    "PIL.Image",
]

a = Analysis(
    [str(ROOT / "organism" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=added_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "torch", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ORGANISM-Windows",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
