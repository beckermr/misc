#!/bin/bash

ns=10000
mw=32
python blend_code.py --max-workers=${mw} --shear=0.02 --n-samples=${ns}
python blend_code.py --max-workers=${mw} --shear=-0.02 --n-samples=${ns}

python blend_code.py --blend --max-workers=${mw} --shear=0.02 --n-samples=${ns}
python blend_code.py --blend --max-workers=${mw} --shear=-0.02 --n-samples=${ns}
