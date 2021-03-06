#!/bin/bash

ns=100000
mw=64
# python blend_code.py --max-workers=${mw} --shear=0.02 --n-samples=${ns}
# python blend_code.py --max-workers=${mw} --shear=-0.02 --n-samples=${ns}
# python blend_code.py --blend --max-workers=${mw} --shear=0.02 --n-samples=${ns}
# python blend_code.py --blend --max-workers=${mw} --shear=-0.02 --n-samples=${ns}

# python blend_code.py --max-workers=${mw} --shear=0.02 --n-samples=${ns} --top
# python blend_code.py --max-workers=${mw} --shear=-0.02 --n-samples=${ns} --top
python blend_code.py --blend --max-workers=${mw} --shear=0.02 --n-samples=${ns} --top
python blend_code.py --blend --max-workers=${mw} --shear=-0.02 --n-samples=${ns} --top
