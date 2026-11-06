# Shim -> consolidated common dataloader (GAN_attacks/common/common_dataloader.py).
# Standardization/resizing are controlled via ImageFolder(standardize=, resize=)
# from the caller (see utils.init_dataloader). Kept as a module named `dataloader`
# so existing `import dataloader` call sites keep working.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from common_dataloader import * 
