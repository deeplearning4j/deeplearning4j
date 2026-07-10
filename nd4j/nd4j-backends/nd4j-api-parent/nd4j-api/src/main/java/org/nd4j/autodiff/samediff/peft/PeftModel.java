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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.*;
import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.DoraMatMul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GgmlQMatMul;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.charset.StandardCharsets;
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
        // saveAdapter() writes adapter_config.json plus one name-keyed <var>.npy per
        // adapter variable into adapterPath — there is no monolithic weights blob.
        File configFile = new File(adapterPath, "adapter_config.json");
        Preconditions.checkState(configFile.exists(), "Adapter config not found: %s", configFile);

        // Load configuration
        PeftConfig config = loadConfig(configFile);

        // Create PEFT model (re-injects the adapter structure so variable names match)
        PeftModel model = fromPretrained(baseModel, config);

        // Load the per-variable .npy weights from the adapter directory, keyed by name
        model.loadAdapterWeights(adapterPath);

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
            case ADALORA:
                applyLora((LoraConfig) peftConfig);
                break;
            case DORA:
                applyDora((DoraConfig) peftConfig);
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

        // If the base model was trained (or had gradients computed) before PEFT wrapping,
        // the duplicated graph carries a backprop function that predates adapter injection
        // and does not contain the adapter variables. Drop it so the next fit() rebuilds
        // gradients against the modified graph.
        model.invalidateGradFunction();

        log.info("Applied {} to model. Trainable params: {}, Total params: {}",
            type, getTrainableParameterCount(), getTotalParameterCount());
    }

    /**
     * Apply LoRA to the model.
     *
     * <p>This performs two passes:
     * <ol>
     *   <li>Dense pass — scans VARIABLE-type weight tensors (rank-2) and injects the
     *       classical LoRA residual (W_eff = W + scaling*(B@A)).</li>
     *   <li>Quantized pass (QLORA) — scans ops whose opName() is "ggml_qmatmul", matches
     *       the packed-weight variable name against targetModules regexes, and injects a
     *       graph-level LoRA residual at the op's OUTPUT variable.  loraA/loraB are created
     *       in the adapter dtype from config; the packed weight is frozen as CONSTANT.</li>
     * </ol>
     *
     * <p>Hybrid models (some quantized, some dense layers) are handled naturally — both
     * passes run and each matches only the layers they own.
     */
    private void applyLora(LoraConfig config) {
        List<String> targetModules = config.getTargetModules();
        List<Pattern> patterns = compilePatterns(targetModules);

        // ── Dense pass ─────────────────────────────────────────────────────────
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

        // ── Quantized pass (ggml_qmatmul ops) ──────────────────────────────────
        if (config.getPeftType() == PeftType.QLORA || config.getPeftType() == PeftType.LORA) {
            injectLoraForQuantizedOps(config, patterns);
        }
    }

    /**
     * Scan the graph for "ggml_qmatmul" ops whose packed-weight variable name matches
     * the targetModules patterns, and inject a graph-level LoRA residual at each one.
     *
     * <p>For each matched op:
     * <ul>
     *   <li>Read activations = input(0), packedWeights = input(1).</li>
     *   <li>Read (quantType, N, K) from the op's iArgs.</li>
     *   <li>Create loraA [rank, K] (Kaiming init) and loraB [N, rank] (zero init) in the
     *       adapter dtype from config (QLoraConfig.loraDataType or FLOAT).</li>
     *   <li>Compute delta = scaling * ((A_2D @ loraAᵀ) @ loraBᵀ), handling rank-3
     *       activations by reshaping [B,S,K] → [B*S,K] and back.</li>
     *   <li>Find the output variable of the matched op, add the delta to it, rename the
     *       combined variable, and rewire all consumers via replaceVariableUsages.</li>
     *   <li>Freeze the packed-weight variable as CONSTANT.</li>
     * </ul>
     */
    private void injectLoraForQuantizedOps(LoraConfig config, List<Pattern> patterns) {
        int rank = config.getR();
        double scaling = config.getScaling();

        DataType adapterDtype = (config instanceof QLoraConfig)
            ? ((QLoraConfig) config).getLoraDataType()
            : DataType.FLOAT;
        if (adapterDtype == null) adapterDtype = DataType.FLOAT;

        // Snapshot ops — we modify the graph mid-loop
        List<SameDiffOp> opsSnapshot = new ArrayList<>(model.getOps().values());

        for (SameDiffOp sdOp : opsSnapshot) {
            DifferentialFunction fn = sdOp.getOp();
            if (fn == null || !"ggml_qmatmul".equals(fn.opName())) {
                continue;
            }

            // Input 1 is the packed weight variable
            List<String> opInputs = sdOp.getInputsToOp();
            if (opInputs == null || opInputs.size() < 2) continue;

            String packedWeightVarName = opInputs.get(1);
            boolean matches = patterns.stream().anyMatch(p -> p.matcher(packedWeightVarName).find());
            if (!matches) continue;

            // Read quantType, N, K from iArgs via the public iArgs() accessor
            GgmlQMatMul ggmlOp = (GgmlQMatMul) fn;
            long[] iArgsArr = ggmlOp.iArgs();
            if (iArgsArr == null || iArgsArr.length < 3) {
                log.warn("ggml_qmatmul op '{}' has insufficient iArgs; skipping QLoRA injection",
                    sdOp.getName());
                continue;
            }
            int quantType = (int) iArgsArr[0];
            long N = iArgsArr[1];
            long K = iArgsArr[2];

            // Output variable of this op (the base result before LoRA)
            List<String> opOutputs = sdOp.getOutputsOfOp();
            if (opOutputs == null || opOutputs.isEmpty()) continue;
            String baseOutputVarName = opOutputs.get(0);
            SDVariable baseOutput = model.getVariable(baseOutputVarName);
            if (baseOutput == null) continue;

            // Activation variable (input 0)
            SDVariable activations = model.getVariable(opInputs.get(0));
            if (activations == null) continue;

            // Build unique names for the adapter variables
            String safeName = packedWeightVarName.replace("/", "_").replace(":", "_");
            String loraAName = safeName + "_qlora_A";
            String loraBName = safeName + "_qlora_B";

            // loraA [rank, K], loraB [N, rank]. A packed weight can feed multiple
            // qmatmul ops (for example lm_logits and lm_logits_last). Reuse one
            // adapter variable pair for all such consumers instead of creating
            // duplicate SameDiff variable names.
            SDVariable loraA = model.getVariable(loraAName);
            SDVariable loraB = model.getVariable(loraBName);
            if (loraA == null && loraB == null) {
                double bound = Math.sqrt(6.0 / K);
                INDArray loraAInit = Nd4j.rand(adapterDtype, rank, K).subi(0.5).muli(2 * bound);
                INDArray loraBInit = Nd4j.zeros(adapterDtype, N, rank);
                loraA = model.var(loraAName, loraAInit);
                loraB = model.var(loraBName, loraBInit);
                peftVariables.add(loraAName);
                peftVariables.add(loraBName);
            } else if (loraA == null || loraB == null) {
                throw new IllegalStateException("Incomplete QLoRA adapter variables for packed weight "
                    + packedWeightVarName + ": loraA=" + (loraA != null) + ", loraB=" + (loraB != null));
            }

            // Compute LoRA delta = scaling * ((A_flat @ loraAᵀ) @ loraBᵀ)
            // Handle rank-3 activations [B,S,K] → reshape to [B*S,K] for the mmuls.
            long[] actShape = activations.getShape();
            boolean rank3 = (actShape != null && actShape.length == 3);

            SDVariable aFlat;
            if (rank3) {
                // [B,S,K] → [B*S, K]
                // Shape may be symbolic (-1); use -1 for the merged dim
                aFlat = activations.reshape(-1, K);
            } else {
                aFlat = activations;
            }

            // aFlat [M,K] @ loraAᵀ [K,rank] → [M,rank]
            SDVariable afterA = model.mmul(aFlat, model.transpose(loraA));
            // [M,rank] @ loraBᵀ [rank,N] → [M,N]
            SDVariable afterB = model.mmul(afterA, model.transpose(loraB));

            SDVariable delta;
            if (rank3 && actShape != null) {
                // reshape back to [B,S,N]
                SDVariable deltaFlat = afterB.mul(scaling);
                delta = deltaFlat.reshape(actShape[0], actShape[1], N);
            } else {
                delta = afterB.mul(scaling);
            }

            // New output = base + delta
            String loraOutputName = baseOutputVarName + "_qlora_out";
            SDVariable loraOutput = baseOutput.add(delta);
            loraOutput.rename(loraOutputName);

            // Rewire all consumers of the base output to use the new combined output
            replaceVariableUsages(baseOutputVarName, loraOutput);

            // Freeze the packed weight
            SDVariable packedWeightVar = model.getVariable(packedWeightVarName);
            if (packedWeightVar != null
                && packedWeightVar.getVariableType() == VariableType.VARIABLE) {
                INDArray arr = packedWeightVar.getArr();
                packedWeightVar.setVariableType(VariableType.CONSTANT);
                if (arr != null) model.setArrayForVariable(packedWeightVarName, arr);
                frozenVariables.add(packedWeightVarName);
            }

            log.info("QLoRA: injected LoRA residual for quantized op on '{}' " +
                "(quantType={}, N={}, K={}, rank={}, scaling={})",
                packedWeightVarName, quantType, N, K, rank, scaling);
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

        // Create LoRA layer, passing the pretrained weight for LoftQ initialization
        String loraName = varName.replace("/", "_").replace(":", "_");
        INDArray pretrainedWeight = (config instanceof LoftQConfig) ? weightVar.getArr() : null;
        LoraLayer loraLayer = LoraLayer.create(model, loraName, config, inFeatures, outFeatures, pretrainedWeight);
        loraLayers.put(varName, loraLayer);

        // LoftQ: freeze the iterated quantized base Q_n as the base weight so that
        // base + scaling*(B@A) ≈ W. Without this the adapters (fit to W - Q_n) are added to
        // the original W, which makes multi-iteration LoftQ worse than a single iteration.
        INDArray loftqBase = loraLayer.getLoftqQuantizedBase();
        if (loftqBase != null) {
            model.setArrayForVariable(varName, loftqBase.castTo(weightVar.dataType()));
        }

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

        // The op that PRODUCES newVar (e.g. the add computing W_eff = W + lora delta)
        // legitimately consumes the original variable and must NOT be rewired — doing so
        // creates a self-cycle (W_eff = W_eff + delta) that the execution DAG rejects.
        Variable newVarMeta = model.getVariables().get(newVar.name());
        String producerOfNewVar = newVarMeta == null ? null : newVarMeta.getOutputOfOp();
        Variable originalMeta = model.getVariables().get(originalVarName);

        // Get all operations that output from this variable (ops where this is an input)
        // We need to update those ops to use the new effective weight instead
        SameDiffOp[] ops = model.getOps().values().toArray(new SameDiffOp[0]);

        for (SameDiffOp sdOp : ops) {
            if (sdOp.getName().equals(producerOfNewVar)) continue;
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

                // Keep the variable -> consumer reverse index consistent: graph traversal
                // (gradient construction, DAG building) reads Variable.inputsForOp, so a
                // rewired consumer must move from the original variable's list to the
                // effective weight's list.
                if (originalMeta != null && originalMeta.getInputsForOp() != null) {
                    originalMeta.getInputsForOp().remove(sdOp.getName());
                }
                if (newVarMeta != null) {
                    if (newVarMeta.getInputsForOp() == null) {
                        newVarMeta.setInputsForOp(new ArrayList<>());
                    }
                    if (!newVarMeta.getInputsForOp().contains(sdOp.getName())) {
                        newVarMeta.getInputsForOp().add(sdOp.getName());
                    }
                }
                log.trace("Rewired op '{}' input {} -> {}", sdOp.getName(), originalVarName, newVar.name());
            }
        }
    }

    /**
     * Apply DoRA (Weight-Decomposed Low-Rank Adaptation) to the model.
     *
     * <p>DoRA uses the {@link DoraMatMul} fused op rather than plain LoRA injection.
     * For each matching rank-2 weight variable it:
     * <ol>
     *   <li>Creates loraA [r, inF], loraB [outF, r], and magnitude [outF] vectors.</li>
     *   <li>Rewires consumers of the weight variable to use a DoraMatMul op that takes
     *       (input, weight, loraA, loraB, magnitude) — this requires knowing the
     *       activation variable, which is found by scanning op inputs.</li>
     * </ol>
     *
     * <p>Because {@link DoraMatMul} requires the activation as an explicit input (not
     * just the weight), DoRA injection is activation-aware: we look for any op that
     * consumes the weight as input(1) (the conventional weight slot), take input(0) as
     * the activation, and replace the entire op with a DoraMatMul.
     */
    private void applyDora(DoraConfig config) {
        List<String> targetModules = config.getTargetModules();
        List<Pattern> patterns = compilePatterns(targetModules);
        int r = config.getR();
        double scaling = config.getScaling();

        for (SDVariable var : model.variables()) {
            if (var.getVariableType() != VariableType.VARIABLE) continue;
            String varName = var.name();
            boolean matches = patterns.stream().anyMatch(p -> p.matcher(varName).find());
            if (!matches) continue;

            long[] shape = var.getShape();
            if (shape == null || shape.length != 2) {
                log.warn("Skipping non-2D variable for DoRA: {} shape={}", varName, Arrays.toString(shape));
                continue;
            }

            int outFeatures = (int) shape[0];
            int inFeatures  = (int) shape[1];
            String safeName = varName.replace("/", "_").replace(":", "_");

            // Create adapter variables
            double bound = Math.sqrt(6.0 / inFeatures);
            INDArray aInit = Nd4j.rand(r, inFeatures).subi(0.5).muli(2 * bound);
            INDArray bInit = Nd4j.zeros(outFeatures, r);
            // Magnitude: init to column norms of W, or ones as fallback
            INDArray wArr = var.getArr();
            INDArray magInit;
            if ("pretrained".equalsIgnoreCase(config.getMagnitudeInit()) && wArr != null) {
                // Compute per-row L2 norm of W [outF, inF] -> [outF]
                magInit = wArr.norm2(1);  // row norms
            } else {
                magInit = Nd4j.ones(outFeatures);
            }

            SDVariable loraA    = model.var(safeName + "_dora_A",   aInit);
            SDVariable loraB    = model.var(safeName + "_dora_B",   bInit);
            SDVariable magnitude = model.var(safeName + "_dora_mag", magInit);

            peftVariables.add(safeName + "_dora_A");
            peftVariables.add(safeName + "_dora_B");
            peftVariables.add(safeName + "_dora_mag");

            // Find ops that use this weight as an input and inject DoraMatMul
            SameDiffOp[] opsSnapshot = model.getOps().values().toArray(new SameDiffOp[0]);
            for (SameDiffOp sdOp : opsSnapshot) {
                List<String> opInputs = sdOp.getInputsToOp();
                if (opInputs == null || !opInputs.contains(varName)) continue;

                int weightIdx = opInputs.indexOf(varName);
                // Conventional slot: weight is index 1, activation is index 0
                if (weightIdx != 1 || opInputs.size() < 2) continue;

                SDVariable activation = model.getVariable(opInputs.get(0));
                if (activation == null) continue;

                // Build DoraMatMul and get its output
                DoraMatMul doraOp = new DoraMatMul(model, activation, var, loraA, loraB,
                    magnitude, scaling);
                SDVariable doraOut = model.updateVariableNameAndReference(
                    doraOp.outputVariables()[0], null);
                doraOut.rename(safeName + "_dora_out");

                // Replace the old op's output with the DoraMatMul output
                List<String> oldOutputs = sdOp.getOutputsOfOp();
                if (oldOutputs != null && !oldOutputs.isEmpty()) {
                    replaceVariableUsages(oldOutputs.get(0), doraOut);
                }
                log.debug("Injected DoRA for '{}': activation={}", varName, opInputs.get(0));
                break;  // one injection per weight var
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
     *
     * Correctly moves arrays from variablesArrays to constantArrays in the SameDiff
     * storage maps, since the storage is type-keyed: setting variableType alone is
     * insufficient — we must also re-register the array under the CONSTANT storage.
     */
    private void freezeBaseModel() {
        for (SDVariable var : model.variables()) {
            if (var.getVariableType() == VariableType.VARIABLE) {
                String name = var.name();
                if (!peftVariables.contains(name)) {
                    // Capture the array BEFORE changing the type (array is in variablesArrays).
                    INDArray arr = var.getArr();
                    // Change the variable type field.
                    var.setVariableType(VariableType.CONSTANT);
                    // Re-register the array under the CONSTANT storage map.
                    // setArrayForVariable dispatches to constantArrays when type == CONSTANT.
                    if (arr != null) {
                        model.setArrayForVariable(name, arr);
                    }
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
     * Merge LoRA weights into the base model and return a clean model.
     *
     * <p><b>Important:</b> this operation is only valid for models with dense (FP32/FP16)
     * base weights.  If the model contains quantized (GGML-packed INT8) base weights, the
     * adapter delta cannot be folded back into the packed bytes — call this method and it
     * will throw an {@link UnsupportedOperationException} with an explanation.  Keep the
     * adapter separate in that case.
     *
     * @return A new SameDiff model with LoRA weights merged into the dense base weights.
     */
    public SameDiff mergeAndUnload() {
        // Guard: detect any quantized LoRA targets
        boolean hasQuantizedBase = false;
        for (SameDiffOp sdOp : model.getOps().values()) {
            DifferentialFunction fn = sdOp.getOp();
            if (fn != null && "ggml_qmatmul".equals(fn.opName())) {
                List<String> opInputs = sdOp.getInputsToOp();
                if (opInputs != null && opInputs.size() >= 2) {
                    String pwName = opInputs.get(1);
                    boolean matched = loraLayers.containsKey(pwName)
                        || peftVariables.stream().anyMatch(v -> v.startsWith(
                            pwName.replace("/","_").replace(":","_") + "_qlora_"));
                    if (matched) { hasQuantizedBase = true; break; }
                }
            }
        }
        if (hasQuantizedBase) {
            throw new UnsupportedOperationException(
                "mergeAndUnload() is not supported for QLoRA (quantized base) models. " +
                "The packed GGML bytes cannot absorb a floating-point adapter delta. " +
                "Keep the adapter separate and load it at inference time via loadAdapterWeights().");
        }

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
     * <p>Saves the adapter in a portable format:
     * <ul>
     *   <li>{@code adapter_config.json} — minimal JSON manifest with peftType, r, alpha,
     *       scaling, targetModules, and adapter variable names.</li>
     *   <li>One {@code <varName>.npy} file per adapter variable (name-keyed, not
     *       positional, so loading is robust to variable ordering changes).</li>
     * </ul>
     *
     * <p>The base model is NOT saved.  Only adapter (PEFT) variables are written.
     *
     * @param outputDir Directory to save to (created if necessary)
     */
    public void saveAdapter(File outputDir) throws IOException {
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        // --- Save adapter weights as per-variable .npy files (name-keyed) ---
        List<String> savedNames = new ArrayList<>();
        for (String peftVar : peftVariables) {
            SDVariable var = model.getVariable(peftVar);
            if (var == null) continue;
            INDArray arr = var.getArr();
            if (arr == null) continue;
            // Use a filesystem-safe version of the variable name
            String fileName = peftVar.replace("/", "__").replace(":", "__") + ".npy";
            File weightFile = new File(outputDir, fileName);
            Nd4j.writeAsNumpy(arr, weightFile);
            savedNames.add(peftVar);
        }

        // --- Save minimal JSON config (hand-built, no Jackson dep required) ---
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"peftType\": \"").append(peftConfig.getPeftType().name()).append("\",\n");
        if (peftConfig instanceof LoraConfig) {
            LoraConfig lc = (LoraConfig) peftConfig;
            json.append("  \"r\": ").append(lc.getR()).append(",\n");
            json.append("  \"loraAlpha\": ").append(lc.getLoraAlpha()).append(",\n");
            json.append("  \"scaling\": ").append(lc.getScaling()).append(",\n");
            json.append("  \"initLoraWeights\": \"").append(lc.getInitLoraWeights()).append("\",\n");
            if (lc.getTargetModules() != null) {
                json.append("  \"targetModules\": [");
                for (int i = 0; i < lc.getTargetModules().size(); i++) {
                    if (i > 0) json.append(", ");
                    json.append("\"").append(lc.getTargetModules().get(i)).append("\"");
                }
                json.append("],\n");
            }
            if (peftConfig instanceof QLoraConfig) {
                QLoraConfig qc = (QLoraConfig) peftConfig;
                json.append("  \"loraDataType\": \"")
                    .append(qc.getLoraDataType() != null ? qc.getLoraDataType().name() : "FLOAT")
                    .append("\",\n");
            }
        }
        json.append("  \"adapterVariables\": [");
        for (int i = 0; i < savedNames.size(); i++) {
            if (i > 0) json.append(", ");
            json.append("\"").append(savedNames.get(i)).append("\"");
        }
        json.append("]\n");
        json.append("}\n");

        File configFile = new File(outputDir, "adapter_config.json");
        try (PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream(configFile), StandardCharsets.UTF_8))) {
            pw.print(json.toString());
        }

        log.info("Saved {} adapter variables to {} (config: {})",
            savedNames.size(), outputDir.getAbsolutePath(), configFile.getName());
    }

    /**
     * Load adapter weights from a weights file or directory.
     *
     * <p>Expects the directory produced by {@link #saveAdapter(File)}: each adapter
     * variable is stored as a {@code <varName>.npy} file (name-keyed).
     * The method maps each .npy back to the corresponding SameDiff variable by name.
     */
    private void loadAdapterWeights(File weightsDir) throws IOException {
        // Look for .npy files and infer the variable name from the file name
        File dir = weightsDir.isDirectory() ? weightsDir : weightsDir.getParentFile();
        File[] npyFiles = dir.listFiles(f -> f.getName().endsWith(".npy"));
        if (npyFiles == null || npyFiles.length == 0) {
            log.warn("No .npy adapter weight files found in {}", dir.getAbsolutePath());
            return;
        }

        int loaded = 0;
        for (File npyFile : npyFiles) {
            // Reverse the name encoding: __ → /
            String rawName = npyFile.getName().replace(".npy", "")
                .replace("__", "/");
            // Also try the underscore-encoded form (some names may have been mangled)
            SDVariable var = model.getVariable(rawName);
            if (var == null) {
                // Fallback: try the filename itself (no slash expansion)
                String altName = npyFile.getName().replace(".npy", "");
                var = model.getVariable(altName);
            }
            if (var == null) {
                log.warn("No SameDiff variable found for adapter weight file '{}'; skipping",
                    npyFile.getName());
                continue;
            }
            INDArray arr = Nd4j.createFromNpyFile(npyFile);
            var.setArray(arr);
            loaded++;
            log.debug("Loaded adapter weight '{}' from {}", var.name(), npyFile.getName());
        }
        log.info("Loaded {} adapter weight(s) from {}", loaded, dir.getAbsolutePath());
    }

    /**
     * Load PEFT configuration from a JSON file (format produced by {@link #saveAdapter}).
     *
     * <p>This is a minimal parser that reads the fields written by saveAdapter.
     * For full Jackson deserialization of externally-produced configs, extend this method.
     */
    private static PeftConfig loadConfig(File configFile) throws IOException {
        // Read the whole file
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(configFile), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line).append("\n");
        }
        String json = sb.toString();

        // Minimal field extraction (no Jackson dependency)
        String peftTypeStr = extractJsonString(json, "peftType");
        PeftType peftType = PeftType.valueOf(peftTypeStr != null ? peftTypeStr : "LORA");

        int r          = parseJsonInt(json, "r", 8);
        int loraAlpha  = parseJsonInt(json, "loraAlpha", 16);
        String init    = extractJsonString(json, "initLoraWeights");
        if (init == null) init = "kaiming_uniform";

        List<String> targetModules = parseJsonStringArray(json, "targetModules");

        if (peftType == PeftType.QLORA) {
            String dtStr = extractJsonString(json, "loraDataType");
            DataType dt = dtStr != null ? DataType.valueOf(dtStr) : DataType.FLOAT;
            return QLoraConfig.builder()
                .r(r).loraAlpha(loraAlpha).initLoraWeights(init)
                .targetModules(targetModules).loraDataType(dt)
                .build();
        }

        return LoraConfig.builder()
            .r(r).loraAlpha(loraAlpha).initLoraWeights(init)
            .targetModules(targetModules)
            .build();
    }

    // ─── Minimal JSON helpers ─────────────────────────────────────────────────

    private static String extractJsonString(String json, String key) {
        String pattern = "\"" + key + "\"\\s*:\\s*\"([^\"]+)\"";
        java.util.regex.Matcher m = java.util.regex.Pattern.compile(pattern).matcher(json);
        return m.find() ? m.group(1) : null;
    }

    private static int parseJsonInt(String json, String key, int defaultVal) {
        String pattern = "\"" + key + "\"\\s*:\\s*(\\d+)";
        java.util.regex.Matcher m = java.util.regex.Pattern.compile(pattern).matcher(json);
        return m.find() ? Integer.parseInt(m.group(1)) : defaultVal;
    }

    private static List<String> parseJsonStringArray(String json, String key) {
        List<String> result = new ArrayList<>();
        String pattern = "\"" + key + "\"\\s*:\\s*\\[([^\\]]+)\\]";
        java.util.regex.Matcher outer = java.util.regex.Pattern.compile(pattern).matcher(json);
        if (!outer.find()) return result;
        String inner = outer.group(1);
        java.util.regex.Matcher items = java.util.regex.Pattern.compile("\"([^\"]+)\"").matcher(inner);
        while (items.find()) result.add(items.group(1));
        return result;
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
