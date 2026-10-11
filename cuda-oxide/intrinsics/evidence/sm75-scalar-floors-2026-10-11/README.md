This directory records the evidence behind the `sm_75` floors of the scalar
arithmetic and scalar math intrinsics (issue #1462). The evidence scanner reads
only JSON files directly in the parent directory and skips this attachment.

LLVM route. `stages.jsonl` has one row per stage group:
`[evidence file, JSON pointer, stage, list digest, [[intrinsic, artifact kind, SHA-256], ...]]`.
The detail of each `rust-llvm-23.1.3-aab11ee0-scalar-*` stage cites a
digest. The digest is the SHA-256 of the sorted `"<intrinsic> <kind> <SHA-256>\n"`
lines. Artifact kinds: `canonical-ir` is `llvm-as` then `llvm-dis` of
`intrinsics/probes/<id>.ll` without the `ModuleID` and `source_filename` lines,
`ptx` is the output of `llc -march=nvptx64 -mcpu=sm_75 -mattr=+ptx70`, and
`cubin` is the output of `ptxas -arch=sm_75` on that PTX. The commands use the
pinned `nightly-2026-10-11` `llvm-as`, `llvm-dis` and `llc`, and CUDA 13.3.0
`ptxas` V13.3.73.

```python
import hashlib, json
from pathlib import Path

for file, pointer, stage, digest, entries in map(json.loads, Path("stages.jsonl").read_text().splitlines()):
    entries_text = "".join(f"{name} {kind} {sha}\n" for name, kind, sha in sorted(map(tuple, entries)))
    assert hashlib.sha256(entries_text.encode()).hexdigest() == digest, (file, pointer)
print("Verified all stage digest lists")
```

libNVVM route. `<fixture>.nvvm.ll` is the typed-pointer NVVM IR cited by the
`cuda-13.3-libnvvm-13.3.33-scalar-*` evidence. `@llvm.used` names the kernel;
without it nvJitLink dead-strips the kernel from the linked cubin. The CUDA
stages ran in `nvidia/cuda:13.3.0-devel-ubuntu24.04`, whose `libnvvm.so.4.0.0`,
`ptxas` and `libnvJitLink.so.13.3.33` match the SHA-256 values recorded in those
evidence files. `nvvmtool.cpp` builds the compile and link step:

```sh
g++ -std=c++17 -I/usr/local/cuda-13.3/include -I/usr/local/cuda-13.3/nvvm/include \
  nvvmtool.cpp -L/usr/local/cuda-13.3/nvvm/lib64 -lnvvm \
  -L/usr/local/cuda-13.3/lib64 -lnvJitLink -o nvvmtool
./nvvmtool FIXTURE.nvvm.ll FIXTURE        # FIXTURE.ptx, .ltoir, .linked.cubin, .linked.ptx
ptxas -arch=sm_75 FIXTURE.ptx -o FIXTURE.sm_75.cubin
```

`nvvmtool` compiles with `-arch=compute_75` (PTX, then `-gen-lto`) and links
the LTOIR with `-arch=sm_75 -lto` (cubin, then `-ptx`).
