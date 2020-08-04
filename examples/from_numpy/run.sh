# DYLD_LIBRARY_PATH="$(rustc --print sysroot)/lib:$DYLD_LIBRARY_PATH" python main.py
LD_LIBRARY_PATH="$(rustc --print sysroot)/lib:$LD_LIBRARY_PATH" python main.py