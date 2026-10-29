#!/usr/bin/env python3
"""Can Triton emit the FP4->FP16 hardware convert that XQA uses?

flashinfer's XQA kernel dequantizes an FP4 KV cache with

    .reg .b8 byte0, byte2;
    mov.b32 {byte0, _, byte2, _}, %2;
    cvt.rn.f16x2.e2m1x2 %0, byte0;

(csrc/xqa/utils.cuh, gated on __CUDA_ARCH__ >= 1000). ptxas assembles that to
a single F2FP.F16.E2M1.UNPACK_B -- but only for the arch-specific targets
sm_120a / sm_100a; plain sm_120 rejects it.

Our kernel instead gathers a 4096-entry LUT, which the memory twin priced at
+0.0429 ms over pure loads on stage 1 (0.1260 ms). So the question worth real
money is whether the hardware instruction is reachable from Triton at all, or
whether reaching it is a genuine reason to drop to CUDA.

This decides it empirically: compile a Triton kernel that emits the asm, then
read back the target Triton actually used and check the SASS.
"""
import torch
import triton
import triton.language as tl

N = 256


@triton.jit
def _fp4_cvt_kernel(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i)
    # Two packed e2m1 values from the low byte -> two halves in one u32.
    y = tl.inline_asm_elementwise(
        "{ .reg .b8 b0; mov.b32 {b0, _, _, _}, $1; cvt.rn.f16x2.e2m1x2 $0, b0; }",
        "=r,r",
        [x],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    tl.store(Y + i, y)


def main():
    x = torch.arange(N, dtype=torch.int32, device="cuda").view(torch.int32)
    y = torch.zeros(N, dtype=torch.int32, device="cuda")
    try:
        compiled = _fp4_cvt_kernel[(1,)](x, y, BLOCK=N, num_warps=4)
    except Exception as e:
        print("COMPILE FAILED")
        msg = str(e)
        print(msg[:1500])
        return

    print("COMPILE OK")
    meta = compiled.metadata
    print("triton target arch :", getattr(meta, "arch", "?"))
    print("num_warps / regs   :", meta.num_warps, "/", compiled.n_regs)

    asm = compiled.asm
    ptx = asm.get("ptx", "")
    hit = [l.strip() for l in ptx.splitlines() if "e2m1" in l]
    print("PTX lines with e2m1:", hit if hit else "(none)")

    sass = asm.get("cubin")
    if sass is not None:
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
            f.write(sass)
            path = f.name
        try:
            out = subprocess.run(
                ["nvdisasm", "-c", path], capture_output=True, text=True
            ).stdout
            f2fp = sorted({w for w in out.split() if w.startswith("F2FP")})
            print("SASS F2FP forms    :", f2fp if f2fp else "(none)")
        finally:
            os.unlink(path)

    # Values: e2m1 code -> value. Low nibble of each input byte.
    torch.cuda.synchronize()
    got = y.view(torch.float16).view(N, 2)[:, 0][:8].tolist()
    codes = list(range(8))
    expect = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    print("codes 0..7 decoded :", got)
    print("e2m1 reference     :", expect)
    print("match              :", got == expect)


if __name__ == "__main__":
    main()
