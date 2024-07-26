# -*- mode: python ; coding: utf-8 -*-
import mtcnn
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

import os
import logging
from PIL import Image, ImageTk
import threading
import sys
import cv2
import glob
import time
from mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor

block_cipher = None

# Define the location of the MTCNN weights file (relative to the script)
mtcnn_weights_path = 'mtcnn_weights.npy'  # Now it's in the same folder

a = Analysis(
    ['Obscurrra.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('icon.ico', '.'),
        ('obscuRRRa Logo Full.png', '.'),
        ('obscuRRRa Logo Monogram.png', '.'),
        ('obscuRRRa Profile image.png', '.'),
        (mtcnn_weights_path, 'mtcnn_weights'),  # No need for a separate folder
        ('haarcascade_frontalface_default.xml', 'haarcascades'),
        ('haarcascade_profileface.xml', 'haarcascades')
    ],
    hiddenimports=[
        'pkg_resources',
        'pkg_resources.extern',
        'setuptools',
        'mtcnn'
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
    upx=True,  # Enable UPX compression
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