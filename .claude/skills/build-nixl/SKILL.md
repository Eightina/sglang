---
name: build-nixl
description: >
  Build and install nixl from source using local CUDA and HPC-X UCX.
  Use this skill whenever the user asks to build, compile, install, or
  set up nixl from source — including when they mention "local CUDA",
  "hpcx", "ucx", "meson build", or want to rebuild nixl after changes.
  Also use it when encountering nixl build errors related to abseil,
  asio, subprojects, CUDA detection, or pkg-config. Even if the user
  just says "install nixl" without specifying "from source", this skill
  applies because pip wheels may not match the local CUDA/UCX setup.
---

# Build & Install nixl from Source

This skill captures the complete, battle-tested process for building nixl
in an environment with:

- **Ubuntu 24.04** (GCC 13.3)
- **CUDA 13.0** at `/usr/local/cuda`
- **HPC-X UCX** at `/opt/hpcx/ucx`
- **Blackwell GPUs** (sm_120)

The build system is **Meson + Ninja**. nixl has several C++ subproject
dependencies that Meson tries to auto-download — in restricted-network
environments, several of these downloads fail and need manual workarounds.
This skill walks through all of them.

## Quick Reference

```bash
# 1. Ensure CUDA is on PATH
export PATH=/usr/local/cuda/bin:$PATH

# 2. Pre-install dependencies that Meson can't auto-download
#    (abseil from source, asio headers + pkg-config)

# 3. Configure
meson setup builddir \
  --prefix=/usr/local \
  -Ducx_path=/opt/hpcx/ucx \
  -Dcudapath_inc=/usr/local/cuda/include \
  -Dcudapath_lib=/usr/local/cuda/lib64 \
  -Dcudapath_stub=/usr/local/cuda/lib64/stubs \
  -Dnixl_cuda_arch_list=120 \
  -Dbuildtype=release \
  -Dbuild_tests=false \
  -Dbuild_examples=false

# 4. Build & install
meson compile -C builddir -j$(nproc)
meson install -C builddir
ldconfig
```

The sections below explain each dependency and pitfall in detail.

---

## Step 1: CUDA on PATH

Meson discovers `nvcc` via PATH — not via `cudapath_*` options. The
`cudapath_inc`/`cudapath_lib` options only tell Meson where to find
headers and libraries *after* nvcc is located. If nvcc isn't on PATH,
you'll get:

```
ERROR: Unknown compiler(s): [['nvcc']]
```

Fix:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

Verify with `nvcc --version` before proceeding.

### CUDA Architecture

Use `-Dnixl_cuda_arch_list=120` for Blackwell (RTX PRO 4000 / B-series).
The default `auto` builds for all architectures (80,86,89,90,100,103,120)
which is much slower. For a single-GPU-arch dev box, targeting just your
GPU is 5-10x faster.

Check your GPU: `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`

---

## Step 2: Pre-install Dependencies

nixl uses Meson subprojects (`.wrap` files) to auto-download dependencies.
In environments where `wrapdb.mesonbuild.com` and `sourceforge.net` are
unreachable (403 Forbidden), you must pre-install these manually.

### 2a. Abseil (abseil-cpp) — MUST build from source

**The problem:** The `abseil-cpp.wrap` file downloads the source tarball
successfully (it's cached in `subprojects/packagecache/`), but the
*patch* file (which provides the `meson.build` for building abseil as a
subproject) must come from `wrapdb.mesonbuild.com`, which is often
blocked. Without the patch, Meson sees the abseil source directory but
finds no `meson.build` inside it:

```
ERROR: Subproject abseil-cpp is buildable: NO
meson.build:158:16: ERROR: Subproject exists but has no meson.build file.
```

**The fix:** Build and install abseil from source so Meson finds it via
pkg-config instead of falling back to the subproject.

The tarball is usually already cached at
`subprojects/packagecache/abseil-cpp-20250814.1.tar.gz`. If not,
download from GitHub releases.

```bash
# Extract (adjust path if tarball isn't cached)
mkdir -p /opt/abseil-src
cd /opt/abseil-src
tar xzf /root/nixl/subprojects/packagecache/abseil-cpp-20250814.1.tar.gz \
  --strip-components=1

# Build with CMake
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_CXX_STANDARD=17 \
  -DABSL_BUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON
cmake --build build -j$(nproc)
make -C build install

# Update shared library cache
ldconfig

# Verify — Meson needs ALL of these to succeed:
pkg-config --modversion absl_base        # should print 20250814
pkg-config --modversion absl_log
pkg-config --modversion absl_status
pkg-config --modversion absl_strings
pkg-config --modversion absl_synchronization
pkg-config --modversion absl_time
```

**Important:** The system `libabsl-dev` package on Ubuntu 24.04 is
version 20210324, which is too old. nixl requires `absl_log` which was
added in newer releases. If you accidentally installed it, remove with
`apt-get remove -y libabsl-dev libabsl20210324` before building.

### 2b. Asio — header-only install + pkg-config

**The problem:** The `asio.wrap` tries to download from `sourceforge.net`
(primary) and `wrapdb.mesonbuild.com` (fallback), both of which return
403 Forbidden. Asio is needed by the UCX backend, OBJ plugin, Azure Blob
plugin, and core listener.

**The fix:** Download from GitHub (which is usually reachable) and install
as a header-only library with a hand-crafted pkg-config file.

```bash
# Download from GitHub (reliable)
cd /tmp
wget -q https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-30-2.tar.gz
tar xzf asio-1-30-2.tar.gz

# Install headers
cp -r asio-asio-1-30-2/asio/include/* /usr/local/include/

# Create pkg-config file so Meson's dependency('asio') finds it
mkdir -p /usr/local/lib/pkgconfig
cat > /usr/local/lib/pkgconfig/asio.pc << 'EOF'
prefix=/usr/local
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include

Name: asio
Description: Asio C++ Library
Version: 1.30.2
Cflags: -I${includedir}
EOF

# Verify
pkg-config --modversion asio   # should print 1.30.2
```

### 2c. Dependencies that auto-download fine

These subprojects download from GitHub and work without intervention:

| Subproject | Source | Notes |
|---|---|---|
| taskflow | github.com/taskflow | Header-only, works out of the box |
| tomlplusplus | github.com/marzer | Header-only, works out of the box |
| liburing | github.com/axboe | System lib found via pkg-config on Ubuntu |
| prometheus-cpp | github.com/jupp0r | Built via CMake subproject |

### 2d. Optional dependencies (auto-disabled if missing)

These are *not* errors — nixl gracefully disables features:

- **etcd-cpp-api** — needed for distributed key-value store support
- **libfabric** — alternative to UCX for RDMA
- **doca-gpunetio** — DOCA GPU net I/O (requires CUDA >= 12.8 + DOCA)
- **cuobjclient** — CUObject client library

---

## Step 3: Configure

```bash
cd /root/nixl   # or wherever the nixl source is

export PATH=/usr/local/cuda/bin:$PATH

rm -rf builddir   # clean slate if re-configuring

meson setup builddir \
  --prefix=/usr/local \
  -Ducx_path=/opt/hpcx/ucx \
  -Dcudapath_inc=/usr/local/cuda/include \
  -Dcudapath_lib=/usr/local/cuda/lib64 \
  -Dcudapath_stub=/usr/local/cuda/lib64/stubs \
  -Dnixl_cuda_arch_list=120 \
  -Dbuildtype=release \
  -Dbuild_tests=false \
  -Dbuild_examples=false
```

### Key Meson Options

| Option | Purpose | Example |
|---|---|---|
| `ucx_path` | Path to UCX install | `/opt/hpcx/ucx` |
| `cudapath_inc` | CUDA include dir | `/usr/local/cuda/include` |
| `cudapath_lib` | CUDA lib dir | `/usr/local/cuda/lib64` |
| `cudapath_stub` | CUDA stub libs | `/usr/local/cuda/lib64/stubs` |
| `nixl_cuda_arch_list` | CUDA SM targets | `120` (Blackwell) |
| `buildtype` | `release` or `debug` | `release` |
| `build_tests` | Build test suite | `false` (skip for faster build) |
| `build_examples` | Build examples | `false` |
| `disable_gds_backend` | Disable GDS plugins | `false` |
| `enable_plugins` | Only build listed plugins | `UCX,GDS,POSIX` |
| `disable_plugins` | Exclude listed plugins | `MOONCAKE,INFINIA` |
| `static_plugins` | Build plugins as built-in | `UCX` |

### Expected Configure Output

When successful, you should see:

```
Message: Using system Abseil dependencies
Cuda compiler for the host machine: nvcc (nvcc 13.0.88)
Message: nvcc version: 13.0.88
Message: CUDA targets: sm_120 + compute_120
Build targets in project: 25
```

The UCX GPU Device API will likely show `NO` — this is normal for
standard HPC-X UCX builds. The UCX backend still works fine without it.

---

## Step 4: Build

```bash
meson compile -C builddir -j$(nproc)
```

This compiles all C++ sources, CUDA kernels, and Python bindings.
On a 256-core machine this takes ~2 minutes. On fewer cores, scale
accordingly.

---

## Step 5: Install

```bash
meson install -C builddir
ldconfig
```

### What Gets Installed

| Component | Location |
|---|---|
| Core libraries | `/usr/local/lib/x86_64-linux-gnu/libnixl*.so` |
| Plugins | `/usr/local/lib/x86_64-linux-gnu/plugins/libplugin_*.so` |
| C++ headers | `/usr/local/include/nixl*.h` |
| Python bindings | `/usr/local/lib/python3/dist-packages/nixl_cu13/` |

The Python package name includes the CUDA major version: `nixl_cu12` for
CUDA 12, `nixl_cu13` for CUDA 13.

### Verify Installation

```bash
# Check libraries
ls /usr/local/lib/x86_64-linux-gnu/libnixl*.so

# Check plugins
ls /usr/local/lib/x86_64-linux-gnu/plugins/

# Check UCX plugin links to the right UCX
ldd /usr/local/lib/x86_64-linux-gnu/plugins/libplugin_UCX.so | grep uc

# Check Python import
python3 -c "import nixl_cu13; print('OK:', dir(nixl_cu13))"
```

Expected UCX linkage:
```
libucp.so.0 => /opt/hpcx/ucx/lib/libucp.so.0
libucs.so.0 => /opt/hpcx/ucx/lib/libucs.so.0
libuct.so.0 => /opt/hpcx/ucx/lib/libuct.so.0
```

---

## Troubleshooting

### "Subproject exists but has no meson.build file"
→ Abseil patch download failed. Build abseil from source (Step 2a).

### "WrapDB connection failed ... 403 Forbidden"
→ Network restriction. Manually install the failing dependency (usually
  abseil or asio — see Step 2).

### "Unknown compiler(s): [['nvcc']]"
→ CUDA not on PATH. Run `export PATH=/usr/local/cuda/bin:$PATH`.

### "absl_log not found" after installing libabsl-dev
→ System abseil is too old. Remove it and build from source (Step 2a).

### Python import fails with "libnixl.so: cannot open shared object"
→ Run `ldconfig` after install, or add the lib dir to `LD_LIBRARY_PATH`.

### Build is very slow
→ Use `-Dnixl_cuda_arch_list=120` (or your GPU arch) instead of `auto`
  to avoid compiling for all CUDA architectures.

### Want to rebuild after code changes
```bash
export PATH=/usr/local/cuda/bin:$PATH
meson compile -C builddir -j$(nproc)
meson install -C builddir
ldconfig
```
No need to re-run `meson setup` unless you change build options.

### Clean rebuild
```bash
rm -rf builddir
# then re-run meson setup from Step 3
```
