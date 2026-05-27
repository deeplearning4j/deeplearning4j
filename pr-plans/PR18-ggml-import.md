# PR18: GGML Import

**Estimated files:** ~84
**Merge layer:** 6
**Complexity:** Medium
**Reviewers:** Import/model team

## Description

GGML/GGUF model import and quantization: architecture detection and mapping
for 15+ model families, GGML/GGUF format readers/writers, quantization types
(Q2_K through IQ4_XS), dequantizers, and model export.

## File Categories

### Architecture detection (~22)
- `ArchitectureConfig.java`
- `ArchitectureRegistry.java`
- `ExportArchitecture.java`
- `ExportArchitectureRegistry.java`
- `LayerTensorDiscovery.java`
- `ModelArchitecture.java`
- Model-specific architectures:
  - `GemmaArchitecture.java`
  - `GenericArchitecture.java`
  - `GLMArchitecture.java`
  - `GptOssArchitecture.java`
  - `GraniteArchitecture.java`
  - `LFM2Architecture.java`
  - `Llama4Architecture.java`
  - `LLaMAArchitecture.java`
  - `LLaMAExportArchitecture.java`
  - `MistralArchitecture.java`
  - `NemotronArchitecture.java`
  - `OLMoArchitecture.java`
  - `OpenELMArchitecture.java`
  - `PhiArchitecture.java`
  - `WhisperArchitecture.java`

### Format readers/writers (~11)
- `GGMLDataType.java`
- `GGMLFormat.java`
- `GGMLFormatDetector.java`
- `GGMLHeader.java`
- `GGMLMetadata.java`
- `GGMLReader.java`
- `GGMLTensorInfo.java`
- `GGMLWriter.java`
- `GGUFHeader.java`
- `GGUFReader.java`
- `GGUFWriter.java`

### Quantization (~43)
- Core: `Quantizer.java`, `QuantizerFactory.java`, `QuantizerInfo.java`,
  `Dequantizer.java`, `DequantizerFactory.java`
- Adaptive: `AdaptiveLayerQuantizer.java`, `AdaptiveQuantConfig.java`,
  `DynamicQuant*.java`
- `GGMLQuantType.java`
- Standard types: `Q2_K`, `Q3_K`, `Q4_0`, `Q4_1`, `Q4_K`,
  `Q5_0`, `Q5_1`, `Q5_K`, `Q6_K`, `Q8_0`, `Q8_K`
- IQ types: `IQ1_M`, `IQ1_S`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`,
  `IQ3_S`, `IQ3_XXS`, `IQ4_NL`, `IQ4_XS`
- Ternary: `TQ1_0`, `TQ2_0`

### Conversion (~2)
- `ConversionOptions.java`
- `GGMLToSameDiffConverter.java`

### Export (~3)
- `ExportOptions.java`
- `SameDiffToGGMLConverter.java`
- `TensorExportInfo.java`

### Top-level (~4)
- `GGMLExportException.java`
- `GGMLImportException.java`
- `GGMLModelExport.java`
- `GGMLModelImport.java`

### Module info (1)
- `module-info.java`

### ADRs (3)
- `ADRs/0052 - GGML-GGUF Model Import.md` — nd4j-ggml module with memory-mapped I/O and configurable dequantization
- `ADRs/0053 - GGML Quantization Handling.md` — Dequantizer interface for Q4_0 through IQ4_XS block-quantization schemes
- `ADRs/0054 - GGML Architecture Detection.md` — Strategy-pattern architecture auto-detection from GGUF metadata for 15+ model families

## Review Focus

- Quantization correctness — dequantized values must match reference
- Architecture mapping — layer names must match GGUF metadata keys
- New model architectures (GLM, Llama4, Granite) — verify tensor mapping
