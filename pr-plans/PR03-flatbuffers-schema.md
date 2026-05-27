# PR03: FlatBuffers Schema & Generated Code

**Estimated files:** ~324
**Merge layer:** 1
**Complexity:** Low (mostly generated/vendored code)
**Reviewers:** Core team (schema review only)

## Description

FlatBuffers runtime headers, schema files (.fbs), generated C++/Java code,
TypeScript/JS webjar (vendored node_modules), and FlatBuffersMapper serde.
Most of these files are auto-generated or vendored — the actual reviewable
surface is small (the .fbs schema files and FlatBuffersMapper.java).

## File Categories

### FlatBuffers C++ runtime headers (~30)
- `libnd4j/include/flatbuffers/` — all .h files (base.h, flatbuffers.h, etc.)

### FlatBuffers duplicate tree (~32)
- `libnd4j/libnd4j/include/flatbuffers/` — mirror of above (likely build artifact)

### Generated graph code (~62)
- `libnd4j/include/graph/generated/` — generated C++ headers (*_generated.h)
- `libnd4j/include/graph/generated/graph/*.java` — generated Java graph classes

### Schema source (1)
- `libnd4j/include/graph/scheme/array.fbs`

### Webjar/TypeScript (~198)
- `nd4j/nd4j-web/nd4j-webjar/` — node_modules/flatbuffers JS runtime + TypeScript graph bindings
  - `node_modules/flatbuffers/js/` — vendored FlatBuffers JS library
  - `node_modules/flatbuffers/mjs/` — ESM variant
  - `src/main/typescript/graph/` — generated TS graph types
  - `src/main/typescript/sd/graph/` — generated TS SameDiff graph types

### Java serde (1)
- `nd4j/.../autodiff/samediff/serde/FlatBuffersMapper.java`

### ADRs (1 — only those actually changed in the diff)
- `ADRs/0035-Samediff-Extended-Storage-Format.md` — SDNB/SDZ unified container format with sharding and metadata

## Review Notes

- Only the `.fbs` schema file and `FlatBuffersMapper.java` need careful review
- Generated code should match the schema — verify with `flatc-generate.sh`
- Webjar node_modules are vendored — diff is noisy but low risk
