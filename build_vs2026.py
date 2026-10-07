"""Build script for gaussian-splatting submodules with VS 2026 + CUDA 13.2"""
import os
import subprocess
import sys

# === VS 2026 paths ===
VS2026 = r"C:\Program Files\Microsoft Visual Studio\18\Community"
MSVC = os.path.join(VS2026, r"VC\Tools\MSVC\14.51.36231")
WIN_SDK = r"C:\Program Files (x86)\Windows Kits\10"
SDK_VER = "10.0.28000.0"
CUDA_HOME = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

# Set environment variables
os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["VCINSTALLDIR"] = os.path.join(VS2026, "VC\\")
os.environ["VSINSTALLDIR"] = VS2026 + "\\"
os.environ["VCToolsInstallDir"] = MSVC + "\\"
os.environ["WindowsSdkDir"] = WIN_SDK + "\\"
os.environ["WindowsSdkVersion"] = SDK_VER
os.environ["WindowsSdkVerBinPath"] = os.path.join(WIN_SDK, "bin", SDK_VER) + "\\"
os.environ["WindowsSdkBinPath"] = os.path.join(WIN_SDK, "bin") + "\\"

# INCLUDE
include_paths = [
    os.path.join(MSVC, "include"),
    os.path.join(MSVC, "ATLMFC", "include"),
    os.path.join(VS2026, r"VC\Auxiliary\VS\include"),
    os.path.join(WIN_SDK, "Include", SDK_VER, "ucrt"),
    os.path.join(WIN_SDK, "Include", SDK_VER, "um"),
    os.path.join(WIN_SDK, "Include", SDK_VER, "shared"),
]
os.environ["INCLUDE"] = ";".join(include_paths)

# LIB
lib_paths = [
    os.path.join(MSVC, "lib", "x64"),
    os.path.join(MSVC, "ATLMFC", "lib", "x64"),
    os.path.join(WIN_SDK, "Lib", SDK_VER, "ucrt", "x64"),
    os.path.join(WIN_SDK, "Lib", SDK_VER, "um", "x64"),
]
os.environ["LIB"] = ";".join(lib_paths)

# PATH - prepend MSVC bin, Windows SDK bin, CUDA bin
path_prepend = [
    os.path.join(MSVC, r"bin\Hostx64\x64"),
    os.path.join(WIN_SDK, "bin", SDK_VER, "x64"),
    os.path.join(CUDA_HOME, "bin"),
]
os.environ["PATH"] = os.pathsep.join(path_prepend) + os.pathsep + os.environ.get("PATH", "")

# Verify rc.exe is findable
import shutil
rc = shutil.which("rc.exe")
print(f"rc.exe found at: {rc}")
cl = shutil.which("cl.exe")
print(f"cl.exe found at: {cl}")
nvcc = shutil.which("nvcc.exe")
print(f"nvcc.exe found at: {nvcc}")
print()

# Build diff-gaussian-rasterization
print("=" * 60)
print("Building diff-gaussian-rasterization...")
print("=" * 60)
dgr_dir = r"D:\02_学习科研\Git clone code\gaussian-splatting\submodules\diff-gaussian-rasterization"
# Clean
import shutil as sh
build_dir = os.path.join(dgr_dir, "build")
if os.path.exists(build_dir):
    sh.rmtree(build_dir)
egg_info = os.path.join(dgr_dir, "diff_gaussian_rasterization.egg-info")
if os.path.exists(egg_info):
    sh.rmtree(egg_info)

# Don't capture output - let it print directly to avoid encoding issues
result = subprocess.run(
    [sys.executable, "setup.py", "build_ext", "--inplace"],
    cwd=dgr_dir,
)
print(f"\nExit code: {result.returncode}")
