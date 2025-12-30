# GGML Architecture Detection

## Status

Implemented

Proposed by: Adam Gibson (30-12-2024)

## Context

GGML/GGUF models embed architecture information in their metadata, allowing tools like llama.cpp to automatically configure the appropriate model structure. Different architectures (LLaMA, Mistral, BERT, GPT-2, Falcon, etc.) have different layer structures, attention mechanisms, and tensor naming conventions.

When importing GGML models into ND4J/SameDiff, we need to:

1. Detect the model architecture from metadata
2. Map GGML tensor names to SameDiff variable names
3. Build the appropriate computational graph for the architecture
4. Configure architecture-specific parameters (RoPE, layer normalization, etc.)

## Related Work

- [ADR 0052 - GGML/GGUF Model Import](./0052%20-%20GGML-GGUF%20Model%20Import.md): Parent ADR for GGML import
- [ADR 0053 - GGML Quantization Handling](./0053%20-%20GGML%20Quantization%20Handling.md): Quantization/dequantization handling
- GGUF metadata specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

## Proposal

We implement an extensible architecture detection and handling system using the Strategy pattern:

### 1. Architecture Interface

```java
public interface ModelArchitecture {
    /**
     * Get the canonical name of this architecture.
     */
    String getName();

    /**
     * Get all architecture variants this handler supports.
     * E.g., LLaMA handler supports: llama, llama2, llama3, mistral, codellama
     */
    Set<String> getSupportedVariants();

    /**
     * Check if this handler can process the given model metadata.
     */
    boolean canHandle(GGMLMetadata metadata);

    /**
     * Extract architecture-specific configuration from metadata.
     */
    ArchitectureConfig getConfig(GGMLMetadata metadata);

    /**
     * Build the SameDiff computational graph for this architecture.
     */
    SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights,
                        ConversionOptions options);

    /**
     * Get tensor name mapping patterns for this architecture.
     * Maps GGML tensor names to SameDiff variable names.
     */
    Map<String, String> getTensorNamePatterns();
}
```

### 2. Architecture Registry

```java
public class ArchitectureRegistry {
    private static final Map<String, ModelArchitecture> architectures = new ConcurrentHashMap<>();

    static {
        register(new LLaMAArchitecture());
        register(new GenericArchitecture());
        // Future: register(new BERTArchitecture());
        // Future: register(new GPT2Architecture());
    }

    public static void register(ModelArchitecture arch) {
        architectures.put(arch.getName(), arch);
        for (String variant : arch.getSupportedVariants()) {
            architectures.put(variant, arch);
        }
    }

    public static ModelArchitecture detectArchitecture(GGMLMetadata metadata) {
        String archName = metadata.getArchitecture();

        // Try direct lookup first
        ModelArchitecture arch = architectures.get(archName);
        if (arch != null && arch.canHandle(metadata)) {
            return arch;
        }

        // Try all registered architectures
        for (ModelArchitecture candidate : architectures.values()) {
            if (candidate.canHandle(metadata)) {
                return candidate;
            }
        }

        // Fall back to generic architecture
        return architectures.get("generic");
    }
}
```

### 3. Architecture Configuration

```java
@Builder
@Data
public class ArchitectureConfig {
    private int hiddenSize;
    private int numLayers;
    private int numAttentionHeads;
    private int numKVHeads;           // For GQA (grouped query attention)
    private int intermediateSize;
    private int vocabSize;
    private int contextLength;

    @Builder.Default
    private float layerNormEpsilon = 1e-5f;

    @Builder.Default
    private float ropeFreqBase = 10000.0f;

    public int getHeadDimension() {
        return hiddenSize / numAttentionHeads;
    }
}
```

### 4. LLaMA Architecture Implementation

The LLaMA architecture handler supports the LLaMA family of models:

```java
public class LLaMAArchitecture implements ModelArchitecture {
    private static final Set<String> SUPPORTED = Set.of(
        "llama", "llama2", "llama3", "mistral", "mixtral",
        "codellama", "vicuna", "alpaca"
    );

    @Override
    public String getName() { return "llama"; }

    @Override
    public Set<String> getSupportedVariants() { return SUPPORTED; }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        return arch != null && SUPPORTED.contains(arch.toLowerCase());
    }

    @Override
    public ArchitectureConfig getConfig(GGMLMetadata metadata) {
        return ArchitectureConfig.builder()
            .hiddenSize(metadata.getHiddenSize())
            .numLayers(metadata.getNumLayers())
            .numAttentionHeads(metadata.getNumAttentionHeads())
            .numKVHeads(metadata.getNumKVHeads())
            .vocabSize(metadata.getVocabSize())
            .contextLength(metadata.getContextLength())
            .build();
    }

    @Override
    public Map<String, String> getTensorNamePatterns() {
        return Map.of(
            "token_embd.weight", "model.embed_tokens.weight",
            "output.weight", "lm_head.weight",
            "output_norm.weight", "model.norm.weight",
            "blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight",
            "blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight",
            "blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight",
            "blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight",
            "blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight",
            "blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.up_proj.weight",
            "blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.down_proj.weight",
            "blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight",
            "blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight"
        );
    }
}
```

### 5. Generic Architecture (Fallback)

```java
public class GenericArchitecture implements ModelArchitecture {
    @Override
    public String getName() { return "generic"; }

    @Override
    public Set<String> getSupportedVariants() {
        return Set.of("generic", "unknown");
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        return true; // Handles any architecture as fallback
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata,
                               Map<String, INDArray> weights,
                               ConversionOptions options) {
        // Creates a simple graph with all weights as constants
        // Suitable for inspection and weight extraction
        SameDiff sd = SameDiff.create();
        for (var entry : weights.entrySet()) {
            sd.constant(entry.getKey(), entry.getValue());
        }
        return sd;
    }
}
```

### 6. Architecture Detection from GGUF Metadata

GGUF files contain architecture information in the metadata key-value pairs:

```java
// Key metadata fields for architecture detection
String arch = metadata.get("general.architecture");      // "llama", "mistral", etc.
String modelName = metadata.get("general.name");         // "Meta-Llama-3.1-8B"
int hiddenSize = metadata.getInt("llama.embedding_length");
int numLayers = metadata.getInt("llama.block_count");
int numHeads = metadata.getInt("llama.attention.head_count");
int numKVHeads = metadata.getInt("llama.attention.head_count_kv");
int contextLength = metadata.getInt("llama.context_length");
```

## Architecture-Specific Features

### LLaMA/Mistral Features

- **RoPE (Rotary Position Embeddings)**: Position encoding applied to Q and K tensors
- **RMSNorm**: Root Mean Square Layer Normalization instead of LayerNorm
- **SwiGLU FFN**: Gated activation with SiLU (Swish) activation
- **GQA (Grouped Query Attention)**: Fewer KV heads than Q heads for efficiency

### Future Architectures

| Architecture | Key Features |
|-------------|--------------|
| BERT | Bidirectional attention, [CLS]/[SEP] tokens, MLM head |
| GPT-2 | Unidirectional attention, learned position embeddings |
| Falcon | Multi-query attention, parallel attention/FFN |
| Phi | Partial rotary embeddings, parallel attention |
| Gemma | GeGLU activation, different normalization |
| Qwen | Similar to LLaMA with different FFN |

## Consequences

### Advantages

1. **Extensibility**: New architectures can be added by implementing `ModelArchitecture` interface.

2. **Automatic Detection**: Architecture is detected from GGUF metadata without user intervention.

3. **Consistent Naming**: Tensor names are mapped to a consistent SameDiff naming convention.

4. **Configuration Extraction**: Architecture-specific parameters are extracted from metadata.

5. **Graceful Fallback**: Unknown architectures fall back to the generic handler.

6. **Variant Support**: Related architectures (LLaMA, Mistral, CodeLlama) share the same handler.

### Drawbacks

1. **Initial Coverage**: Only LLaMA-family architectures are fully implemented initially.

2. **Metadata Dependency**: Relies on correct metadata in GGUF files.

3. **Architecture Evolution**: New architecture variants may require handler updates.

4. **Graph Complexity**: Building full computational graphs for some architectures is complex.

## Appendix A: GGUF Metadata Keys by Architecture

### LLaMA/Mistral
```
general.architecture = "llama" | "mistral"
llama.embedding_length = 4096
llama.block_count = 32
llama.attention.head_count = 32
llama.attention.head_count_kv = 8
llama.context_length = 4096
llama.feed_forward_length = 11008
llama.rope.freq_base = 10000.0
llama.attention.layer_norm_rms_epsilon = 1e-5
```

### BERT (Future)
```
general.architecture = "bert"
bert.embedding_length = 768
bert.block_count = 12
bert.attention.head_count = 12
bert.context_length = 512
```

## Appendix B: Tensor Name Mapping Examples

| GGML Name | SameDiff Name |
|-----------|---------------|
| `token_embd.weight` | `model.embed_tokens.weight` |
| `blk.0.attn_q.weight` | `model.layers.0.self_attn.q_proj.weight` |
| `blk.15.ffn_gate.weight` | `model.layers.15.mlp.gate_proj.weight` |
| `output_norm.weight` | `model.norm.weight` |
| `output.weight` | `lm_head.weight` |

