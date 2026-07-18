# SDX AOT package (@PLATFORM@, @VARIANT@)

Ahead-of-time compiled (GraalVM native-image) export of the deeplearning4j Java
stack — no JVM required.

- `bin/sdx-llm` — CLI: `generate`, `import` (GGUF -> SDZ for the C runtime in the
  main SDX SDK package), `tokenize`, `info`, `vlm` (SmolDocling document
  extraction), `transcribe` (Whisper STT). Run with `--help`.
- `lib/libsdx_llm.*` — shared library exposing the `sdxLlm*` / `sdxVlm*` /
  `sdxAudio*` C ABI declared in `include/sdx_llm_c.h`.
- `lib/` also carries the side-loaded native libraries (ND4J backend, BLAS,
  tokenizers, ffmpeg, JDK AWT/sound). Keep `bin/` and `lib/` siblings — the
  binaries resolve `../lib` at startup; `SDX_NATIVE_LIB_DIR` overrides.

Built by the `sdx-aot` Maven module (`-Psdx-aot,native`, plus `-Pcuda` for the
CUDA variant). See ADR 0109 in the deeplearning4j repository.
