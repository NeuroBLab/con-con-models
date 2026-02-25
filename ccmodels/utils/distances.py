import os
import importlib

#Check if USE_FREQ flag is defined and equals true. If so, then we are using frequencies
_using_sfq = os.getenv("USE_FREQ", "false") == "true"

#Getter for the main variable
def using_spatial_freq():
    return _using_sfq

_importedmodule = importlib.import_module("ccmodels.utils.spfrequtils" if _using_sfq else "ccmodels.utils.angleutils")

#Just set the methods to be visible from here so from other methods can be called as mod.f and not as mod._importedmodule.f
signed_dist             = _importedmodule.signed_dist
signed_dist_vectorized  = _importedmodule.signed_dist_vectorized
unsigned_dist           = _importedmodule.unsigned_dist
construct_delta_ori     = _importedmodule.construct_delta_ori 

if _using_sfq:
    print("Hey, this is using freqs!!!")




