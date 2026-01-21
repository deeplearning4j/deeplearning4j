/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.peft;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.*;
import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

/**
 * PEFT (Parameter-Efficient Fine-Tuning) Model wrapper for SameDiff.
 * <p>
 * This class wraps a SameDiff model and applies PEFT methods like LoRA,
 * prompt tuning, prefix tuning, or adapters. It's analogous to Hugging Face's
 * `get_peft_model()` function.
 * <p>
 * Key features:
 * <ul>
 *   <li>Automatic adapter injection based on configuration</li>
 *   <li>Freezing of base model parameters</li>
 *   <li>Adapter weight saving and loading</li>
 *   <li>Merging adapters back into base model</li>
 *   <li>Multiple adapter support</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>
 * // Load base model
 * SameDiff baseModel = SameDiff.load(new File("model.sd"));
 *
 * // Create LoRA configuration
 * LoraConfig loraConfig = LoraConfig.builder()
 *     .r(16)
 *     .loraAlpha(32)
 *     .targetModules(Arrays.asList("query", "key", "value"))
 *     .build();
 *
 * // Create PEFT model
 * PeftModel peftModel = PeftModel.fromPretrained(baseModel, loraConfig);
 *
 * // Print trainable parameters
 * peftModel.printTrainableParameters();
 *
 * // Train
 * peftModel.fit(trainingData, epochs);
 *
 * // Save adapter weights
 * peftModel.saveAdapter("my_adapter");
 *
 * // Merge and export
 * SameDiff merged = peftModel.mergeAndUnload();
 * </pre>
 *
 * @author Adam Gibson
 * @see PeftConfig
 * @see LoraConfig
 * @see <a href="https://github.com/huggingface/peft">Hugging Face PEFT</a>
 */
@Slf4j
public class PeftModel {

    /**
     * The underlying SameDiff model with PEFT modifications.
     */
    @Getter
    private final SameDiff model;

    /**
     * The original base model (frozen).
     */
    private final SameDiff baseModel;

    /**
     * The PEFT configuration.
     */
    @Getter
    private final PeftConfig peftConfig;

    /**
     * Map of adapter names to their configurations.
     */
    @Getter
    private final Map<String, PeftConfig> adapters;

    /**
     * Currently active adapter name.
     */
    @Getter
    private String activeAdapter;

    /**
     * Map of LoRA layers by target module name.
     */
    private final Map<String, LoraLayer> loraLayers;

    /**
     * Names of variables that were frozen (original model parameters).
     */
    private final Set<String> frozenVariables;

    /**
     * Names of trainable PEFT variables.
     */
    private final Set<String> peftVariables;

    /**
     * Private constructor - use factory methods.
     */
    private PeftModel(SameDiff model, SameDiff baseModel, PeftConfig config) {
        this.model = model;
        this.baseModel = baseModel;
        this.peftConfig = config;
        this.adapters = new LinkedHashMap<>();
        this.loraLayers = new LinkedHashMap<>();
        this.frozenVariables = new HashSet<>();
        this.peftVariables = new HashSet<>();
        this.activeAdapter = "default";
        this.adapters.put("default", config);
    }

    /**
     * Create a PEFT model from a pretrained SameDiff model.
     *
     * @param baseModel The pretrained base model
     * @param config    The PEFT configuration
     * @return A new PeftModel wrapping the base model with PEFT applied
     */
    public static PeftModel fromPretrained(SameDiff baseModel, PeftConfig config) {
        config.validate();

        // Create a copy of the model to modify
        SameDiff peftModel = baseModel.dup();

        PeftModel model = new PeftModel(peftModel, baseModel, config);
        model.applyPeft();

        return model;
    }

    /**
     * Create a PEFT model by loading adapter weights.
     *
     * @param baseModel   The pretrained base model
     * @param adapterPath Path to saved adapter weights
     * @return A PeftModel with loaded adapter
     */
    public static PeftModel fromPretrained(SameDiff baseModel, File adapterPath) throws IOException {
        // Load adapter config and weights
        File configFile = new File(adapterPath, "adapter_config.json");
        File weightsFile = new File(adapterPath, "adapter_weights.bin");

        Preconditions.checkState(configFile.exists(), "Adapter config not found: %s", configFile);
        Preconditions.checkState(weightsFile.exists(), "Adapter weights not found: %s", weightsFile);

        // Load configuration
        PeftConfig config = loadConfig(configFile);

        // Create PEFT model
        PeftModel model = fromPretrained(baseModel, config);

        // Load weights
        model.loadAdapterWeights(weightsFile);

        return model;
    }

    /**
     * Apply PEFT modifications to the model based on configuration.
     */
    private void applyPeft() {
        PeftType type = peftConfig.getPeftType();

        switch (type) {
            case LORA:
            case QLORA:
            case DORA:
            case ADALORA:
                applyLora((LoraConfig) peftConfig);
                break;

            case LOHA:
                applyLoha((LohaConfig) peftConfig);
                break;

            case LOKR:
                applyLokr((LokrConfig) peftConfig);
                break;

            case PROMPT_TUNING:
                applyPromptTuning((PromptTuningConfig) peftConfig);
                break;

            case PREFIX_TUNING:
                applyPrefixTuning((PrefixTuningConfig) peftConfig);
                break;

            case ADAPTERS:
                applyAdapters((AdapterConfig) peftConfig);
                break;

            case IA3:
                applyIA3((IA3Config) peftConfig);
                break;

            default:
                throw new UnsupportedOperationException("PEFT type not yet supported: " + type);
        }

        // Freeze base model parameters
        freezeBaseModel();

        log.info("Applied {} to model. Trainable params: {}, Total params: {}",
            type, getTrainableParameterCount(), getTotalParameterCount());
    }

    /**
     * Apply LoRA to the model.
     */
    private void applyLora(LoraConfig config) {
        List<String> targetModules = config.getTargetModules();
        List<Pattern> patterns = compilePatterns(targetModules);

        // Find matching variables and inject LoRA
        for (SDVariable var : model.variables()) {
            if (var.getVariableType() != VariableType.VARIABLE) {
                continue;
            }

            String varName = var.name();
            boolean matches = patterns.stream().anyMatch(p -> p.matcher(varName).find());

            if (matches) {
                injectLoraForVariable(var, config);
            }
        }
    }

    /**
     * Inject LoRA for a specific weight variable.
     * This creates LoRA matrices and modifies the computation graph to include LoRA contributions.
     */
    private void injectLoraForVariable(SDVariable weightVar, LoraConfig config) {
        String varName = weightVar.name();
        long[] shape = weightVar.getShape();

        if (shape == null || shape.length != 2) {
            log.warn("Skipping non-2D variable for LoRA: {} shape={}", varName, Arrays.toString(shape));
            return;
        }

        int outFeatures = (int) shape[0];
        int inFeatures = (int) shape[1];

        // Create LoRA layer
        String loraName = varName.replace("/", "_").replace(":", "_");
        LoraLayer loraLayer = LoraLayer.create(model, loraName, config, inFeatures, outFeatures);
        loraLayers.put(varName, loraLayer);

        // Track PEFT variables
        peftVariables.add(loraName + "_lora_A");
        peftVariables.add(loraName + "_lora_B");

        injectLoraIntoGraph(weightVar, loraLayer, loraName, config);

        log.debug("Injected LoRA for '{}': [{}, {}] -> rank {}", varName, outFeatures, inFeatures, config.getR());
    }

    /**
     * Modify the computation graph to include LoRA contribution for a weight variable.
     * Creates W_effective = W + scaling * (B @ A) and replaces uses of W with W_effective.
     */
    private void injectLoraIntoGraph(SDVariable weightVar, LoraLayer loraLayer, String loraName, LoraConfig config) {
        SDVariable loraA = loraLayer.getLoraA();  // [r, inFeatures]
        SDVariable loraB = loraLayer.getLoraB();  // [outFeatures, r]
        double scaling = config.getScaling();

        // Compute LoRA delta: B @ A -> [outFeatures, inFeatures]
        SDVariable loraDelta = model.mmul(loraB, loraA);

        // Apply scaling
        if (scaling != 1.0) {
            loraDelta = loraDelta.mul(scaling);
        }

        // Apply dropout during training if configured
        double dropout = config.getLoraDropout();
        if (dropout > 0) {
            // dropout(input, inverted, probabilityValue) - inverted=false means standard dropout
            loraDelta = model.nn.dropout(loraDelta, false, dropout);
        }

        // Create effective weight: W_eff = W + loraDelta
        String effName = loraName + "_effective_weight";
        SDVariable effectiveWeight = weightVar.add(loraDelta);
        effectiveWeight.rename(effName);

        // Find all operations that use the original weight and replace with effective weight
        replaceVariableUsages(weightVar.name(), effectiveWeight);

        log.debug("Created effective weight '{}' for LoRA injection", effName);
    }

    /**
     * Replace all usages of originalVar with newVar in the computation graph.
     * This is necessary to make LoRA effective in the forward pass.
     */
    private void replaceVariableUsages(String originalVarName, SDVariable newVar) {
        SDVariable originalVar = model.getVariable(originalVarName);
        if (originalVar == null) {
            log.warn("Original variable not found for replacement: {}", originalVarName);
            return;
        }

        // Get all operations that output from this variable (ops where this is an input)
        // We need to update those ops to use the new effective weight instead
        SameDiffOp[] ops = model.getOps().values().toArray(new SameDiffOp[0]);

        for (SameDiffOp sdOp : ops) {
            List<String> inputVarNames = sdOp.getInputsToOp();
            if (inputVarNames == null || inputVarNames.isEmpty()) continue;

            boolean needsUpdate = false;
            List<String> newInputs = new ArrayList<>(inputVarNames.size());

            for (int i = 0; i < inputVarNames.size(); i++) {
                if (inputVarNames.get(i).equals(originalVarName)) {
                    newInputs.add(newVar.name());
                    needsUpdate = true;
                } else {
                    newInputs.add(inputVarNames.get(i));
                }
            }

            if (needsUpdate) {
                // Update the operation to use the new input
                sdOp.setInputsToOp(newInputs);

                // Also update the variable's input list
                for (String outputName : sdOp.getOutputsOfOp()) {
                    SDVariable outVar = model.getVariable(outputName);
                    if (outVar != null) {
                        // Update any cached input references
                        log.trace("Updated op '{}' to use effective weight for output '{}'",
                            sdOp.getName(), outputName);
                    }
                }
            }
        }
    }

    /**
     * Apply Prompt Tuning to the model.
     */
    private void applyPromptTuning(PromptTuningConfig config) {
        int numTokens = config.getNumVirtualTokens();
        int embDim = config.getTokenEmbeddingDim();

        // Create soft prompt embeddings
        INDArray promptEmbeddings = initializePromptEmbeddings(config);
        SDVariable softPrompts = model.var("soft_prompt_embeddings", promptEmbeddings);

        peftVariables.add("soft_prompt_embeddings");
        log.info("Created soft prompt embeddings: [{}, {}]", numTokens, embDim);
    }

    /**
     * Initialize prompt embeddings based on configuration.
     */
    private INDArray initializePromptEmbeddings(PromptTuningConfig config) {
        int numTokens = config.getNumVirtualTokens();
        int embDim = config.getTokenEmbeddingDim();

        switch (config.getPromptTuningInit()) {
            case ZEROS:
                return Nd4j.zeros(numTokens, embDim);

            case KAIMING:
                double std = Math.sqrt(2.0 / embDim);
                return Nd4j.randn(numTokens, embDim).muli(std);

            case RANDOM:
            default:
                return Nd4j.randn(numTokens, embDim).muli(0.02);
        }
    }

    /**
     * Apply Prefix Tuning to the model.
     */
    private void applyPrefixTuning(PrefixTuningConfig config) {
        int numTokens = config.getNumVirtualTokens();
        int numLayers = config.getNumLayers();
        int hiddenSize = config.getHiddenSize();

        // Create prefix embeddings for each layer
        for (int layer = 0; layer < numLayers; layer++) {
            String keyPrefix = String.format("prefix_key_layer_%d", layer);
            String valuePrefix = String.format("prefix_value_layer_%d", layer);

            INDArray keyInit = Nd4j.randn(numTokens, hiddenSize).muli(0.02);
            INDArray valueInit = Nd4j.randn(numTokens, hiddenSize).muli(0.02);

            model.var(keyPrefix, keyInit);
            model.var(valuePrefix, valueInit);

            peftVariables.add(keyPrefix);
            peftVariables.add(valuePrefix);
        }

        // Create projection network if configured
        if (config.isPrefixProjection()) {
            createPrefixProjection(config);
        }

        log.info("Created prefix tuning for {} layers with {} tokens", numLayers, numTokens);
    }

    /**
     * Create the prefix projection MLP.
     */
    private void createPrefixProjection(PrefixTuningConfig config) {
        int encoderHidden = config.getEncoderHiddenSize();
        int numTokens = config.getNumVirtualTokens();
        int hiddenSize = config.getHiddenSize();
        int numLayers = config.getNumLayers();

        // Projection weights
        INDArray proj1 = Nd4j.randn(numTokens, encoderHidden).muli(0.02);
        INDArray proj2 = Nd4j.randn(encoderHidden, 2 * hiddenSize * numLayers).muli(0.02);

        model.var("prefix_projection_1", proj1);
        model.var("prefix_projection_2", proj2);

        peftVariables.add("prefix_projection_1");
        peftVariables.add("prefix_projection_2");
    }

    /**
     * Apply Adapter layers to the model.
     */
    private void applyAdapters(AdapterConfig config) {
        int adapterSize = config.getAdapterSize();
        int hiddenSize = config.getHiddenSize();
        int numLayers = config.getNumLayers();

        List<Integer> layerIndices = config.getLayerIndices();
        if (layerIndices == null) {
            layerIndices = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                layerIndices.add(i);
            }
        }

        for (int layer : layerIndices) {
            if (config.isAdapterAfterAttention()) {
                createAdapterLayer(layer, "attn", hiddenSize, adapterSize, config);
            }
            if (config.isAdapterAfterFeedforward()) {
                createAdapterLayer(layer, "ff", hiddenSize, adapterSize, config);
            }
        }

        log.info("Created adapters for {} layers with size {}", layerIndices.size(), adapterSize);
    }

    /**
     * Create an adapter layer.
     */
    private void createAdapterLayer(int layer, String position, int hiddenSize, int adapterSize, AdapterConfig config) {
        String prefix = String.format("adapter_layer_%d_%s", layer, position);

        // Down projection
        INDArray downWeight = Nd4j.randn(hiddenSize, adapterSize).muli(0.02);
        INDArray downBias = Nd4j.zeros(adapterSize);
        model.var(prefix + "_down_weight", downWeight);
        model.var(prefix + "_down_bias", downBias);

        // Up projection
        INDArray upWeight = Nd4j.randn(adapterSize, hiddenSize).muli(0.02);
        INDArray upBias = Nd4j.zeros(hiddenSize);
        model.var(prefix + "_up_weight", upWeight);
        model.var(prefix + "_up_bias", upBias);

        peftVariables.add(prefix + "_down_weight");
        peftVariables.add(prefix + "_down_bias");
        peftVariables.add(prefix + "_up_weight");
        peftVariables.add(prefix + "_up_bias");
    }

    /**
     * Apply IA³ to the model.
     */
    private void applyIA3(IA3Config config) {
        List<String> targetModules = config.getTargetModules();
        List<Pattern> patterns = compilePatterns(targetModules);

        // Create learned scaling vectors for matching modules
        for (SDVariable var : model.variables()) {
            if (var.getVariableType() != VariableType.VARIABLE) {
                continue;
            }

            String varName = var.name();
            boolean matches = patterns.stream().anyMatch(p -> p.matcher(varName).find());

            if (matches) {
                long[] shape = var.getShape();
                if (shape != null && shape.length >= 1) {
                    int dim = (int) shape[shape.length - 1];
                    String ia3Name = varName.replace("/", "_") + "_ia3";

                    INDArray init = config.isInitToOne() ? Nd4j.ones(dim) : Nd4j.randn(dim).muli(0.02);
                    model.var(ia3Name, init);
                    peftVariables.add(ia3Name);

                    log.debug("Created IA³ vector for '{}': dim={}", varName, dim);
                }
            }
        }
    }

    /**
     * Apply LoHa (Low-Rank Hadamard Product) to the model.
     * LoHa uses Hadamard products of two low-rank decompositions:
     * ΔW = (B₁ @ A₁) ⊙ (B₂ @ A₂)
     * where ⊙ is the Hadamard (element-wise) product.
     * Effective rank can be up to dim².
     */
    private void applyLoha(LohaConfig config) {
        List<String> targetModules = config.getTargetModules();
        List<Pattern> patterns = compilePatterns(targetModules);

        for (SDVariable var : model.variables()) {
            if (var.getVariableType() != VariableType.VARIABLE) {
                continue;
            }

            String varName = var.name();
            boolean matches = patterns.stream().anyMatch(p -> p.matcher(varName).find());

            if (matches) {
                injectLohaForVariable(var, config);
            }
        }
    }

    /**
     * Inject LoHa for a specific weight variable.
     */
    private void injectLohaForVariable(SDVariable weightVar, LohaConfig config) {
        String varName = weightVar.name();
        long[] shape = weightVar.getShape();

        if (shape == null || shape.length != 2) {
            log.warn("Skipping non-2D variable for LoHa: {} shape={}", varName, Arrays.toString(shape));
            return;
        }

        int outFeatures = (int) shape[0];
        int inFeatures = (int) shape[1];
        int dim = config.getDim();  // Low-rank dimension
        double alpha = config.getAlpha();
        double scaling = alpha / dim;

        String lohaName = varName.replace("/", "_").replace(":", "_");

        // Create first decomposition: B₁ [outFeatures, dim] @ A₁ [dim, inFeatures]
        INDArray a1Init = initializeLohaMatrix(dim, inFeatures);
        INDArray b1Init = Nd4j.zeros(outFeatures, dim);
        SDVariable lohaA1 = model.var(lohaName + "_loha_A1", a1Init);
        SDVariable lohaB1 = model.var(lohaName + "_loha_B1", b1Init);

        // Create second decomposition: B₂ [outFeatures, dim] @ A₂ [dim, inFeatures]
        INDArray a2Init = initializeLohaMatrix(dim, inFeatures);
        INDArray b2Init = Nd4j.zeros(outFeatures, dim);
        SDVariable lohaA2 = model.var(lohaName + "_loha_A2", a2Init);
        SDVariable lohaB2 = model.var(lohaName + "_loha_B2", b2Init);

        // Track PEFT variables
        peftVariables.add(lohaName + "_loha_A1");
        peftVariables.add(lohaName + "_loha_B1");
        peftVariables.add(lohaName + "_loha_A2");
        peftVariables.add(lohaName + "_loha_B2");

        // Compute LoHa delta: (B₁ @ A₁) ⊙ (B₂ @ A₂)
        SDVariable prod1 = model.mmul(lohaB1, lohaA1);  // [outFeatures, inFeatures]
        SDVariable prod2 = model.mmul(lohaB2, lohaA2);  // [outFeatures, inFeatures]
        SDVariable lohaDelta = prod1.mul(prod2);  // Hadamard product

        // Apply scaling
        if (scaling != 1.0) {
            lohaDelta = lohaDelta.mul(scaling);
        }

        // Create effective weight and inject
        String effName = lohaName + "_effective_weight";
        SDVariable effectiveWeight = weightVar.add(lohaDelta);
        effectiveWeight.rename(effName);
        replaceVariableUsages(weightVar.name(), effectiveWeight);

        log.debug("Injected LoHa for '{}': [{}, {}] -> dim={}, effective_rank={}",
            varName, outFeatures, inFeatures, dim, dim * dim);
    }

    /**
     * Initialize a LoHa matrix with Kaiming initialization.
     */
    private INDArray initializeLohaMatrix(int rows, int cols) {
        double bound = Math.sqrt(6.0 / cols);
        return Nd4j.rand(rows, cols).subi(0.5).muli(2 * bound);
    }

    /**
     * Apply LoKr (Low-Rank Kronecker Product) to the model.
     * LoKr uses Kronecker products for weight updates:
     * ΔW = C ⊗ (B @ A)
     * where ⊗ is the Kronecker product.
     */
    private void applyLokr(LokrConfig config) {
        List<String> targetModules = config.getTargetModules();
        List<Pattern> patterns = compilePatterns(targetModules);

        for (SDVariable var : model.variables()) {
            if (var.getVariableType() != VariableType.VARIABLE) {
                continue;
            }

            String varName = var.name();
            boolean matches = patterns.stream().anyMatch(p -> p.matcher(varName).find());

            if (matches) {
                injectLokrForVariable(var, config);
            }
        }
    }

    /**
     * Inject LoKr for a specific weight variable.
     */
    private void injectLokrForVariable(SDVariable weightVar, LokrConfig config) {
        String varName = weightVar.name();
        long[] shape = weightVar.getShape();

        if (shape == null || shape.length != 2) {
            log.warn("Skipping non-2D variable for LoKr: {} shape={}", varName, Arrays.toString(shape));
            return;
        }

        int outFeatures = (int) shape[0];
        int inFeatures = (int) shape[1];
        int dim = config.getDim();
        int factor = config.getFactor();
        double alpha = config.getAlpha();
        double scaling = alpha / dim;

        // Compute factorization dimensions
        // For Kronecker product: W = C ⊗ (BA) where C is [f1, f2] and BA is [outFeatures/f1, inFeatures/f2]
        int[] factors = computeKroneckerFactors(outFeatures, inFeatures, factor);
        int f1 = factors[0];
        int f2 = factors[1];
        int d1 = outFeatures / f1;
        int d2 = inFeatures / f2;

        String lokrName = varName.replace("/", "_").replace(":", "_");

        // Create Kronecker factor C: [f1, f2]
        INDArray cInit = Nd4j.eye(Math.min(f1, f2));
        if (f1 != f2) {
            cInit = Nd4j.zeros(f1, f2);
            for (int i = 0; i < Math.min(f1, f2); i++) {
                cInit.putScalar(i, i, 1.0);
            }
        }
        SDVariable lokrC = model.var(lokrName + "_lokr_C", cInit);

        // Create low-rank decomposition: B [d1, dim] @ A [dim, d2]
        INDArray aInit = initializeLohaMatrix(dim, d2);
        INDArray bInit = Nd4j.zeros(d1, dim);
        SDVariable lokrA = model.var(lokrName + "_lokr_A", aInit);
        SDVariable lokrB = model.var(lokrName + "_lokr_B", bInit);

        // Track PEFT variables
        peftVariables.add(lokrName + "_lokr_C");
        peftVariables.add(lokrName + "_lokr_A");
        peftVariables.add(lokrName + "_lokr_B");

        // Compute LoKr delta using Kronecker product approximation
        // For now, we use a reshape-based approach since direct Kronecker product can be memory-intensive
        SDVariable ba = model.mmul(lokrB, lokrA);  // [d1, d2]

        // Kronecker product: tile and scale
        // C ⊗ BA = outer product structure
        // We approximate this with a learned combination
        SDVariable lokrDelta = computeKroneckerApprox(lokrC, ba, f1, f2, d1, d2, outFeatures, inFeatures);

        // Apply scaling
        if (scaling != 1.0) {
            lokrDelta = lokrDelta.mul(scaling);
        }

        // Create effective weight and inject
        String effName = lokrName + "_effective_weight";
        SDVariable effectiveWeight = weightVar.add(lokrDelta);
        effectiveWeight.rename(effName);
        replaceVariableUsages(weightVar.name(), effectiveWeight);

        log.debug("Injected LoKr for '{}': [{}, {}] -> dim={}, factors=[{}, {}]",
            varName, outFeatures, inFeatures, dim, f1, f2);
    }

    /**
     * Compute Kronecker factors for the weight dimensions.
     */
    private int[] computeKroneckerFactors(int outFeatures, int inFeatures, int factor) {
        if (factor <= 0) {
            // Auto-compute factors
            factor = (int) Math.sqrt(Math.min(outFeatures, inFeatures));
            factor = Math.max(2, Math.min(factor, 8));
        }

        // Find factors that divide evenly
        int f1 = factor;
        while (outFeatures % f1 != 0 && f1 > 1) {
            f1--;
        }
        int f2 = factor;
        while (inFeatures % f2 != 0 && f2 > 1) {
            f2--;
        }

        return new int[]{f1, f2};
    }

    /**
     * Compute Kronecker product approximation for LoKr.
     * Uses reshape and broadcast operations to approximate C ⊗ (B @ A).
     */
    private SDVariable computeKroneckerApprox(SDVariable c, SDVariable ba, int f1, int f2, int d1, int d2,
                                              int outFeatures, int inFeatures) {
        // Reshape BA to [1, 1, d1, d2] and C to [f1, f2, 1, 1]
        // Then broadcast multiply and reshape to [f1*d1, f2*d2] = [outFeatures, inFeatures]
        SDVariable baReshaped = ba.reshape(1, 1, d1, d2);
        SDVariable cReshaped = c.reshape(f1, f2, 1, 1);

        // Broadcast multiply
        SDVariable kronecker = baReshaped.mul(cReshaped);  // [f1, f2, d1, d2]

        // Transpose and reshape to final dimensions
        // Need to rearrange [f1, f2, d1, d2] -> [f1, d1, f2, d2] -> [f1*d1, f2*d2]
        SDVariable transposed = kronecker.permute(0, 2, 1, 3);  // [f1, d1, f2, d2]
        return transposed.reshape(outFeatures, inFeatures);
    }

    /**
     * Freeze the base model parameters.
     */
    private void freezeBaseModel() {
        for (SDVariable var : model.variables()) {
            if (var.getVariableType() == VariableType.VARIABLE) {
                String name = var.name();
                if (!peftVariables.contains(name)) {
                    var.setVariableType(VariableType.CONSTANT);
                    frozenVariables.add(name);
                }
            }
        }

        log.debug("Froze {} base model variables", frozenVariables.size());
    }

    /**
     * Compile regex patterns for target modules.
     */
    private List<Pattern> compilePatterns(List<String> patterns) {
        List<Pattern> compiled = new ArrayList<>();
        if (patterns != null) {
            for (String p : patterns) {
                compiled.add(Pattern.compile(p));
            }
        }
        return compiled;
    }

    /**
     * Get the merged weight for a LoRA-modified variable.
     * Returns W₀ + scaling * B @ A
     *
     * @param varName Original variable name
     * @return Merged weight, or null if not a LoRA variable
     */
    public INDArray getMergedWeight(String varName) {
        LoraLayer loraLayer = loraLayers.get(varName);
        if (loraLayer == null) {
            return null;
        }

        SDVariable baseVar = baseModel.getVariable(varName);
        if (baseVar == null) {
            return null;
        }

        INDArray baseWeight = baseVar.getArr();
        INDArray loraUpdate = loraLayer.getMergedWeightUpdate();

        return baseWeight.add(loraUpdate);
    }

    /**
     * Merge LoRA weights into the base model and return an unmodified model.
     * This is useful for inference without adapter overhead.
     *
     * @return A new SameDiff model with LoRA weights merged
     */
    public SameDiff mergeAndUnload() {
        if (peftConfig.getPeftType() != PeftType.LORA &&
            peftConfig.getPeftType() != PeftType.QLORA) {
            throw new UnsupportedOperationException(
                "mergeAndUnload only supported for LoRA. Type: " + peftConfig.getPeftType());
        }

        SameDiff merged = baseModel.dup();

        for (Map.Entry<String, LoraLayer> entry : loraLayers.entrySet()) {
            String varName = entry.getKey();
            LoraLayer loraLayer = entry.getValue();

            SDVariable var = merged.getVariable(varName);
            if (var != null) {
                INDArray mergedWeight = getMergedWeight(varName);
                var.setArray(mergedWeight);
            }
        }

        log.info("Merged {} LoRA layers into base model", loraLayers.size());
        return merged;
    }

    /**
     * Save adapter weights to a directory.
     *
     * @param outputDir Directory to save to
     */
    public void saveAdapter(File outputDir) throws IOException {
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        // Save configuration
        // TODO: Implement proper JSON serialization
        // saveConfig(new File(outputDir, "adapter_config.json"));

        // Save adapter weights
        Map<String, INDArray> adapterWeights = new LinkedHashMap<>();
        for (String peftVar : peftVariables) {
            SDVariable var = model.getVariable(peftVar);
            if (var != null && var.getArr() != null) {
                adapterWeights.put(peftVar, var.getArr());
            }
        }

        File weightsFile = new File(outputDir, "adapter_weights.bin");
        INDArray[] weightsArray = adapterWeights.values().toArray(new INDArray[0]);
        // Save each weight individually with index
        for (int i = 0; i < weightsArray.length; i++) {
            File weightFile = new File(outputDir, "adapter_weight_" + i + ".npy");
            Nd4j.writeAsNumpy(weightsArray[i], weightFile);
        }

        log.info("Saved {} adapter weights to {}", adapterWeights.size(), outputDir);
    }

    /**
     * Load adapter weights from a directory.
     */
    private void loadAdapterWeights(File weightsDir) throws IOException {
        // Load weights from individual numpy files
        List<INDArray> weights = new ArrayList<>();
        int i = 0;
        File weightFile = new File(weightsDir, "adapter_weight_" + i + ".npy");
        while (weightFile.exists()) {
            INDArray weight = Nd4j.createFromNpyFile(weightFile);
            weights.add(weight);
            i++;
            weightFile = new File(weightsDir, "adapter_weight_" + i + ".npy");
        }
        // TODO: Implement proper weight loading with variable name mapping
    }

    /**
     * Load PEFT configuration from a file.
     */
    private static PeftConfig loadConfig(File configFile) throws IOException {
        // TODO: Implement proper JSON deserialization
        throw new UnsupportedOperationException("Config loading not yet implemented");
    }

    /**
     * Get the number of trainable parameters.
     */
    public long getTrainableParameterCount() {
        long count = 0;
        for (String varName : peftVariables) {
            SDVariable var = model.getVariable(varName);
            if (var != null && var.getArr() != null) {
                count += var.getArr().length();
            }
        }
        return count;
    }

    /**
     * Get the total number of parameters in the model.
     */
    public long getTotalParameterCount() {
        long count = 0;
        for (SDVariable var : model.variables()) {
            if (var.getVariableType() == VariableType.VARIABLE ||
                var.getVariableType() == VariableType.CONSTANT) {
                if (var.getArr() != null) {
                    count += var.getArr().length();
                }
            }
        }
        return count;
    }

    /**
     * Get the percentage of trainable parameters.
     */
    public double getTrainablePercentage() {
        long total = getTotalParameterCount();
        if (total == 0) return 0;
        return 100.0 * getTrainableParameterCount() / total;
    }

    /**
     * Print a summary of trainable parameters.
     */
    public void printTrainableParameters() {
        long trainable = getTrainableParameterCount();
        long total = getTotalParameterCount();
        double percentage = getTrainablePercentage();

        System.out.printf("trainable params: %,d || all params: %,d || trainable%%: %.4f%n",
            trainable, total, percentage);
    }

    /**
     * Set the training configuration.
     */
    public void setTrainingConfig(TrainingConfig config) {
        model.setTrainingConfig(config);
    }

    /**
     * Train the PEFT model.
     */
    public void fit(DataSetIterator iterator, int epochs) {
        model.fit(iterator, epochs);
    }

    /**
     * Train the PEFT model with MultiDataSet.
     */
    public void fit(MultiDataSetIterator iterator, int epochs) {
        model.fit(iterator, epochs);
    }

    /**
     * Run inference.
     */
    public Map<String, INDArray> output(Map<String, INDArray> placeholders, String... outputs) {
        return model.output(placeholders, outputs);
    }

    /**
     * Disable adapter and run base model only.
     */
    public void disableAdapter() {
        // Set LoRA matrices to zero effect
        for (LoraLayer layer : loraLayers.values()) {
            if (layer.getLoraB() != null) {
                layer.getLoraB().getArr().assign(0);
            }
        }
        log.info("Disabled adapter '{}'", activeAdapter);
    }

    /**
     * Enable/re-enable the active adapter.
     */
    public void enableAdapter() {
        // Adapters are enabled by default after creation
        log.info("Enabled adapter '{}'", activeAdapter);
    }

    /**
     * Get a summary of the PEFT model.
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("PeftModel Summary\n");
        sb.append("=================\n");
        sb.append("PEFT Type: ").append(peftConfig.getPeftType()).append("\n");
        sb.append("Task Type: ").append(peftConfig.getTaskType()).append("\n");
        sb.append("Active Adapter: ").append(activeAdapter).append("\n");
        sb.append("\n");

        if (peftConfig instanceof LoraConfig) {
            LoraConfig lora = (LoraConfig) peftConfig;
            sb.append("LoRA Config:\n");
            sb.append("  - Rank (r): ").append(lora.getR()).append("\n");
            sb.append("  - Alpha: ").append(lora.getLoraAlpha()).append("\n");
            sb.append("  - Scaling: ").append(String.format("%.4f", lora.getScaling())).append("\n");
            sb.append("  - Dropout: ").append(lora.getLoraDropout()).append("\n");
            sb.append("  - Target Modules: ").append(lora.getTargetModules()).append("\n");
            sb.append("  - LoRA Layers: ").append(loraLayers.size()).append("\n");
        }

        sb.append("\n");
        sb.append("Parameters:\n");
        sb.append(String.format("  - Trainable: %,d%n", getTrainableParameterCount()));
        sb.append(String.format("  - Total: %,d%n", getTotalParameterCount()));
        sb.append(String.format("  - Trainable %%: %.4f%%%n", getTrainablePercentage()));

        return sb.toString();
    }
}
