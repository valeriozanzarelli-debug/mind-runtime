# PyInstaller spec per CEREBRUM (Windows).
# Costruisce un singolo cerebrum.exe che avvia il runtime cerebrale locale.
# Con torch+CUDA installati nell'ambiente di build, la GPU viene sfruttata a runtime.

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

hidden = [
    "cerebrum", "cerebrum.brain", "cerebrum.server", "cerebrum.cli",
    "cerebrum.neuro", "cerebrum.neuro.field",
    "cerebrum.body", "cerebrum.body.neurochemistry", "cerebrum.body.homeostasis",
    "cerebrum.body.drives", "cerebrum.body.reflexes",
    "cerebrum.sense", "cerebrum.sense.vision", "cerebrum.sense.language",
    "cerebrum.motor", "cerebrum.motor.speech",
    "cerebrum.mind", "cerebrum.mind.memory", "cerebrum.mind.consciousness",
    "numpy",
]

a = Analysis(
    ["../cerebrum/__main__.py"],
    pathex=[".."],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "tkinter"],
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
    name="cerebrum",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
