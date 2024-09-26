# -*- mode: python ; coding: utf-8 -*-
import os
from mtcnn import MTCNN
from tkinter import filedialog, messagebox, ttk, scrolledtext
from ttkthemes import ThemedTk

block_cipher = None

# Define the location of the Haarcascade and MTCNN weights file
mtcnn_weights_path = 'mtcnn_weights.npy'

a = Analysis(
    ['Obscurrra.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('icon.ico', '.'),
        ('obscuRRRa Logo Full.png', '.'),
        ('obscuRRRa Logo Monogram.png', '.'),
        ('obscuRRRa Profile image.png', '.'),
        (get_resource_path(mtcnn_weights_path), '.')  # Ensure MTCNN weights file is included
    ],
    hiddenimports=[
        'pkg_resources',
        'pkg_resources.extern',
        'setuptools',
        'tensorflow',
        'tensorflow._api.v2.compat.v1',
        'tensorflow._api.v2.compat.v1.compat',
        'tensorflow._api.v2.compat.v1.compat.v2',
        'tensorflow._api.v2.compat.v2',
        'tensorflow._api.v2.compat.v2.compat',
        'tensorflow._api.v2.compat.v2.compat.v1',
        'mtcnn'  # Explicitly include mtcnn in hidden imports
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pandas',
        'scipy',
        'sklearn'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Obscurrra',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Obscurrra',
)
