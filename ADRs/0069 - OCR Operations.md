# ADR: OCR Operations

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

Document understanding tasks — extracting structured text, tables, and layout information from images of documents — are a primary use case for Vision-Language Models (VLMs). Models like SmolDocling are specifically designed for document OCR, producing structured output with bounding boxes, text content, and document element classification.

Previously, OCR capabilities required external libraries (Tesseract, EasyOCR) or cloud APIs, creating deployment complexity and preventing end-to-end GPU-accelerated processing. With the VLM inference pipeline (ADR 0064) now capable of running vision-language models natively, we can implement OCR as a first-class operation within the ND4J/SameDiff framework.

The key requirements for an integrated OCR system are:

1. **Native Model Loading**: Load ONNX-exported OCR models directly via SameDiff import
2. **Image Preprocessing**: Standard normalization, resizing, and tiling for document images
3. **Multi-Language Support**: Handle diverse scripts without per-language model switching
4. **Structured Output**: Return bounding boxes, text content, and confidence scores
5. **GPU Acceleration**: Leverage the existing CUDA infrastructure for inference

## Decision

We implement an OCR engine abstraction with a DeepSeek-based implementation that uses SameDiff for model execution, integrated with the VLM image preprocessing pipeline.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     OCR Engine Abstraction                      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ AbstractOCREngine                                         │  │
│  │  - initialize(): Load model                               │  │
│  │  - recognize(File): Process file                          │  │
│  │  - recognize(BufferedImage): Process image                │  │
│  │  - getSupportedLanguages(): Query capabilities            │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│  ┌──────────────────────────┴───────────────────────────────┐  │
│  │ DeepSeekOCREngine                                         │  │
│  │                                                           │  │
│  │  ┌─────────────────┐  ┌────────────────────────────────┐ │  │
│  │  │ Vision Encoder   │  │ Text Decoder                   │ │  │
│  │  │ (SameDiff/ONNX)  │  │ (SameDiff/ONNX)               │ │  │
│  │  │                  │  │                                │ │  │
│  │  │ Image → Features │  │ Features → Text + BBoxes      │ │  │
│  │  └─────────────────┘  └────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ OCRConfig                                           │ │  │
│  │  │  - imageSize (default 1024)                         │ │  │
│  │  │  - imageMean, imageStd (ImageNet normalization)     │ │  │
│  │  │  - maxTokens                                        │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Engine Abstraction

The `AbstractOCREngine` base class defines the OCR contract:

```java
public abstract class AbstractOCREngine {
    public abstract void initialize() throws Exception;
    public abstract OCRResult recognize(File imageFile) throws Exception;
    public abstract OCRResult recognize(BufferedImage image) throws Exception;
    public abstract List<String> getSupportedLanguages();
    public abstract void close();
}

public class OCRResult {
    List<TextRegion> regions;  // Detected text regions with bounding boxes
    String fullText;           // Concatenated text output
    double confidence;         // Overall confidence score
}

public class TextRegion {
    BoundingBox bbox;    // [x, y, width, height]
    String text;         // Recognized text content
    double confidence;   // Per-region confidence
    String language;     // Detected language
}
```

### DeepSeekOCREngine Implementation

```java
public class DeepSeekOCREngine extends AbstractOCREngine {
    private SameDiff visionEncoder;
    private SameDiff textDecoder;
    private OCRConfig config;

    public static DeepSeekOCREngine create(File modelDirectory) {
        return create(modelDirectory, OCRConfig.defaults());
    }

    public void initialize() {
        visionEncoder = OnnxFrameworkImporter.import(
            new File(modelDir, "vision_encoder.onnx"));
        textDecoder = OnnxFrameworkImporter.import(
            new File(modelDir, "text_decoder.onnx"));
    }

    public OCRResult recognize(BufferedImage image) {
        // 1. Preprocess image (resize, normalize, tile)
        INDArray preprocessed = preprocess(image, config.imageSize,
            config.imageMean, config.imageStd);

        // 2. Vision encoding → feature tensor
        INDArray features = visionEncoder.output(
            Map.of("pixel_values", preprocessed)).get("features");

        // 3. Autoregressive text decoding
        String text = decode(textDecoder, features, config.maxTokens);

        // 4. Parse structured output (text + bounding boxes)
        return parseOCROutput(text);
    }
}
```

### Multi-Language Support

Supported languages are model-dependent, not engine-dependent:

```java
private static final List<String> SUPPORTED_LANGUAGES = List.of(
    "en", "zh", "ja", "ko", "ar", "hi", "ru", "de", "fr", "es", "pt", "it"
);
```

The model handles language detection and switching internally — no per-language weight files or explicit language selection is needed. The tokenizer vocabulary covers all supported scripts.

### Image Preprocessing

Standard preprocessing pipeline reuses the VLM image preprocessing infrastructure:

1. **Resize**: Scale to `config.imageSize × config.imageSize` (default 1024)
2. **Normalize**: ImageNet mean/std normalization (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)
3. **Tile**: For high-resolution documents, split into overlapping tiles for parallel encoding
4. **Tensor Conversion**: Convert to `[1, 3, H, W]` float tensor for model input

### Integration with VLM Pipeline

The OCR engine integrates with the broader VLM pipeline:

- Uses the same `VLMImagePreprocessor` for image handling
- Can share the vision encoder with the VLM pipeline when the same model supports both OCR and general VLM tasks
- Output parsing handles structured formats (DocTags, markdown, bounding box coordinates)

## Consequences

### Advantages

**End-to-End GPU Acceleration**: The entire OCR pipeline (preprocessing, encoding, decoding) runs on GPU via SameDiff, eliminating CPU-GPU data transfer bottlenecks.

**No External Dependencies**: OCR capability is built into the framework — no Tesseract, cloud APIs, or external processes needed.

**Multi-Language**: Single model handles 12+ languages without model switching or language-specific configuration.

**Structured Output**: Bounding boxes and text regions enable document layout understanding, not just text extraction.

### Disadvantages

**Model Size**: OCR models are 2-5GB, adding significant download and memory requirements compared to traditional OCR engines like Tesseract (~50MB).

**ONNX Import Dependency**: Models must be exported to ONNX format and importable by SameDiff. Unsupported ONNX ops would block model loading.

**Accuracy Variability**: Model-based OCR accuracy varies by document type, font, and language. Traditional OCR engines like Tesseract may perform better on specific document types (e.g., typewritten text in English).

## References

- DeepSeekOCREngine.java in samediff-vlm/src/main/java/org/eclipse/deeplearning4j/vlm/input/ocr/
- AbstractOCREngine.java
- VLMImagePreprocessor.java
- ADR 0064 - VLM Inference Pipeline
