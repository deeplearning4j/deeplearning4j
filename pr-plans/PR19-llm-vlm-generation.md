# PR19: LLM/VLM Generation Pipeline

**Estimated files:** ~217 (128 samediff-llm + 42 samediff-vlm + 14 samediff-audio + 33 samediff-pipeline-*)
**Merge layer:** 6
**Complexity:** High
**Reviewers:** LLM/inference team

## Description

LLM/VLM inference and generation pipeline spanning multiple modules:
- **samediff-llm** (~128 files): GenerationPipeline, KV cache management,
  sampling, speculative decoding, eval/benchmark, model editing, tokenizers
- **samediff-vlm** (~42 files): Vision-language model pipeline, visual embeddings,
  image preprocessing, vision encoder integration, OCR engines (Tesseract, PaddleOCR,
  EasyOCR, TrOCR, DeepSeek), VLM eval benchmarks
- **samediff-audio** (~14 files): Whisper speech-to-text, audio preprocessing, TTS
- **samediff-pipeline-*** (~33 files): Pipeline infrastructure modules for GGML,
  SafeTensors, and ONNX model loading

## File Categories

### Generation core (~50)
- `GenerationPipeline.java`
- `GenerationPipelineConfig.java`
- `GenerationResult.java`
- `TextGenerator.java`
- `FrozenDecodeStep.java`
- `DecodeOptions.java`
- `DecoderInputBuilder.java`
- `DecoderUtils.java`
- `DecodeStepDiagnostics.java`
- `ModelIOConfig.java`
- `StaticKvCacheDecodeLoop.java`

### KV cache management (~20)
- `KvCacheManager.java`
- `KvCacheStrategy.java`
- `UnifiedKvCacheManager.java`
- `PagedKVCache.java`
- `PerLayerPagedKVCache.java`
- `EvictablePagedKVCache.java`
- `QuantizedPagedKVCache.java`
- `MLAKVCache.java`
- `TieredKVCacheManager.java`
- `PerLayerKVPolicy.java`
- `KVCacheCheckpoint.java`
- `KVCacheCheckpointManager.java`
- `KVCacheDiskOffloader.java`
- `KVCacheHostOffloader.java`
- `KVCachePrefixTree.java`
- `RadixPrefixCache.java`
- `PrefixLookupResult.java`

### Eviction policies
- `EvictionPolicy.java`
- `H2OEvictionPolicy.java`
- `SinkAwareEvictionPolicy.java`
- `StreamingLLMEvictionPolicy.java`
- `TokenEvictionPolicy.java`
- `AttentionSinkDetector.java`

### Sampling
- `Sampler.java`
- `SamplerUtils.java`
- `SamplingConfig.java`
- `GreedySampler.java`
- `CompositeSampler.java`

### Speculative decoding
- `SpeculativeDecodeLoop.java`
- `SpeculativeKVCacheManager.java`
- `Speculator.java`
- `DraftModelSpeculator.java`
- `NgramSpeculator.java`
- `TreeAttentionVerifier.java`

### Batching
- `BatchCompactor.java`
- `BatchGenerationState.java`
- `ContinuousBatchScheduler.java`
- `ChunkedPrefillEngine.java`
- `BeamKVCacheManager.java`

### Utilities
- `SameDiffMemoryUtils.java`
- `ReferenceTokenStream.java`
- `TurboQuantCodebook.java`
- `package-info.java`

### Tokenizers (~8)
- `Tokenizer.java`
- `TokenizerFactory.java`
- `TokenizerException.java`
- `HuggingFaceTokenizer.java`
- `SentencePieceTokenizer.java`
- `CLIPTokenizer.java`
- `ChatTemplate.java`
- `Encoding.java`

### Config (~3)
- `ModelConfig.java`
- `PreprocessorConfig.java`
- `TokenizerConfig.java`

### Data
- `LLMModelDownloader.java`

### Model editing/abliteration (~7)
- `AbliterationConfig.java`
- `AbliterationResult.java`
- `AbliterationWorkflow.java`
- `DefaultPromptSets.java`
- `RefusalDirection.java`
- `RefusalDirectionFinder.java`
- `WeightOrthogonalizer.java`

### Eval/benchmark (~25)
- `AnswerExtractor.java`, `EvalConfig.java`, `EvalResult.java`
- `EvalRunner.java`, `GenerationQualityValidator.java`
- `PerplexityEvaluator.java`, `SampleResult.java`
- Benchmarks: `ArcBenchmark`, `Gsm8kBenchmark`, `HellaSwagBenchmark`,
  `MMLUBenchmark`, `TruthfulQABenchmark`, `WinograndeBenchmark`, `BenchmarkTask`, `OutputType`
- Datasets: `CsvDataset`, `CustomDataset`, `DatasetCache`, `EvalDataset`,
  `EvalSample`, `HuggingFaceDataset`, `JsonlDataset`
- Metrics: ANLS, BLEU, ExactMatch, F1, MultipleChoiceAccuracy,
  RelaxedAccuracy, Rouge, VqaAccuracy
- Validation: `BenchmarkConfig`, `BenchmarkRunner`, `DecodeInputEvolutor`,
  `DecodeStepValidator`, `DecodeValidationFramework`, `DivergenceReport`,
  `DspAccuracyValidator`, `MultiLevelComparator`, `OpDivergence`,
  `ReplayMetadataTracker`, `ValidationConfig`, `ValidationResult`

### Multi-model pipeline (~8)
- `ModelRole.java`, `ModelType.java`
- `MultiModelPipeline.java`, `MultiModelPipelineConfig.java`
- `PipelineResult.java`, `PipelineStage.java`, `StageOutput.java`
- `package-info.java`

### Model downloader
- `ModelDownloader.java`

---

## samediff-vlm Module (~42 files)

VLM (Vision-Language Model) inference pipeline with multi-modal support.

### VLM pipeline core
- `VLMGenerationPipeline.java`
- `VLMPipelineConfig.java`
- `VLMPipelineResult.java`

### Visual embeddings & encoding
- `EmbeddingMerger.java`
- `PipelinedVisionEncoder.java`
- `VisionEncoderConfig.java`

### Image preprocessing
- `ImageTiler.java`
- `ImagePromptBuilder.java`
- `ImagePreprocessor.java`

### Model-specific implementations (SmolDocling, etc.)

### Build
- `pom.xml`, `module-info.java`

---

## samediff-audio Module (~14 files)

Audio/speech model support (Whisper, TTS).

### Audio pipeline
- Whisper model implementation
- Audio preprocessing (mel spectrogram, feature extraction)
- TTS (text-to-speech) support

### Build
- `pom.xml`, `module-info.java`

---

## samediff-pipeline-* Modules (~33 files)

Pipeline infrastructure for loading models from different formats.

### samediff-pipeline-core (~15 files)
- Core pipeline interfaces and shared utilities

### samediff-pipeline-ggml (~8 files)
- GGML/GGUF format pipeline loader

### samediff-pipeline-safetensors (~7 files)
- SafeTensors format pipeline loader

### samediff-pipeline-onnx (~3 files)
- ONNX format pipeline loader

---

### ADR (1 — only those actually changed in the diff)
- `ADRs/0064 - VLM Inference Pipeline.md` — Multi-model, multi-GPU VLM pipeline with DSP-backed autoregressive generation

## Review Focus

- GenerationPipeline — main decode loop correctness
- KV cache lifecycle — no stale/leaked caches
- Speculative decoding — draft/verification token matching
- Quantized KV cache — accuracy impact
- VLM pipeline — vision encoder + text decoder integration
- Audio pipeline — Whisper model accuracy
- Pipeline loaders — format compatibility across GGML/SafeTensors/ONNX
