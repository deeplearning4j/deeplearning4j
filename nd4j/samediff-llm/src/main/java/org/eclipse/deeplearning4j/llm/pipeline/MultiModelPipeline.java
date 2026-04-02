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

package org.eclipse.deeplearning4j.llm.pipeline;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Multi-model pipeline for chaining multiple LLM models together.
 *
 * <p>Similar to the VLM OCR pipeline in samediff-vlm, this framework allows
 * assembling models for different roles (summarization, classification, sorting,
 * relation extraction, language detection, NER) and passing outputs from one
 * model to the next.</p>
 *
 * <p>Key features:</p>
 * <ul>
 *   <li>Modular model registration with role-based configuration</li>
 *   <li>DSP (Dynamic Shape Plan) optimizations with CUDA graph support</li>
 *   <li>Memory leak protection via {@link SameDiffMemoryUtils}</li>
 *   <li>GPU memory pool trimming between stages</li>
 *   <li>Model reuse detection and optimization</li>
 *   <li>Structured output parsing for each model type</li>
 *   <li>KV cache strategy support (STATIC, PAGED, QUANTIZED, TURBOQUANT)</li>
 *   <li>Uses {@link org.eclipse.deeplearning4j.llm.generation.StaticKvCacheDecodeLoop} 
 *       for efficient autoregressive decoding with {@link org.eclipse.deeplearning4j.llm.generation.KvCacheManager}</li>
 * </ul>
 *
 * <p>Architecture:</p>
 * <ul>
 *   <li>Each model is wrapped in a {@code GenerationPipeline} which handles:
 *     <ul>
 *       <li>Token embedding via direct lookup or embed_tokens model</li>
 *       <li>KV cache management via configured {@code KvCacheStrategy}</li>
 *       <li>Autoregressive decode loop with {@code StaticKvCacheDecodeLoop}</li>
 *       <li>Memory-safe cleanup via {@code SameDiffMemoryUtils}</li>
 *     </ul>
 *   </li>
 *   <li>Pipeline executes stages sequentially, passing output of one as input to next</li>
 *   <li>DSP is disabled during first model compilation (reduces peak memory), re-enabled after</li>
 *   <li>GPU memory pools are trimmed between stages to prevent fragmentation</li>
 * </ul>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * MultiModelPipelineConfig config = MultiModelPipelineConfig.builder()
 *     .tokenizer(tokenizer)
 *     .kvCacheStrategy(KvCacheStrategy.STATIC)
 *     .maxTokensPerModel(256)
 *     .build();
 *
 * try (MultiModelPipeline pipeline = new MultiModelPipeline(config)) {
 *     // Register models for different roles
 *     pipeline.registerModel("summarizer", ModelType.SUMMARIZER, summarizerModel);
 *     pipeline.registerModel("classifier", ModelType.CLASSIFIER, classifierModel);
 *     pipeline.registerModel("ner", ModelType.NER_MODEL, nerModel);
 *
 *     // Define pipeline stages
 *     List<PipelineStage> stages = Arrays.asList(
 *         new PipelineStage("summarizer", "summary"),
 *         new PipelineStage("classifier", "category"),
 *         new PipelineStage("ner", "entities")
 *     );
 *
 *     // Execute pipeline
 *     PipelineResult result = pipeline.execute(stages, "Long input text...");
 *
 *     // Access stage outputs
 *     String summary = result.getStageOutput("summary").getText();
 *     String category = result.getStageOutput("category").getStructuredData();
 *     List<ModelType.Entity> entities = (List<ModelType.Entity>) 
 *         result.getStageOutput("entities").getStructuredData();
 * }
 * }</pre>
 *
 * @see ModelRole
 * @see ModelType
 * @see MultiModelPipelineConfig
 * @see PipelineStage
 * @see PipelineResult
 * @see org.eclipse.deeplearning4j.llm.generation.GenerationPipeline
 * @see org.eclipse.deeplearning4j.llm.generation.StaticKvCacheDecodeLoop
 * @see org.eclipse.deeplearning4j.llm.generation.KvCacheStrategy
 * @see org.eclipse.deeplearning4j.llm.generation.KvCacheManager
 */
@Slf4j
public class MultiModelPipeline implements AutoCloseable {

    private final MultiModelPipelineConfig config;
    private final Tokenizer tokenizer;

    /** Registered models: modelId -> RegisteredModel */
    private final Map<String, MultiModelPipelineConfig.RegisteredModel> modelRegistry;

    /** Loaded GenerationPipelines: modelId -> GenerationPipeline */
    private final Map<String, GenerationPipeline> generationPipelines;

    /** Model reuse tracking: SameDiff hash -> modelId */
    private final Map<Integer, String> modelIdentityMap;

    /** Whether the pipeline has been closed */
    private volatile boolean closed = false;

    /**
     * Create a new multi-model pipeline.
     *
     * @param config the pipeline configuration
     * @throws IOException if model loading fails
     */
    public MultiModelPipeline(MultiModelPipelineConfig config) throws IOException {
        this.config = config;
        this.tokenizer = config.getTokenizer();
        this.modelRegistry = new ConcurrentHashMap<>(config.getModelRegistry());
        this.generationPipelines = new ConcurrentHashMap<>();
        this.modelIdentityMap = new ConcurrentHashMap<>();

        if (tokenizer == null) {
            throw new IllegalArgumentException("Tokenizer is required");
        }

        // Load models from paths
        loadModelsFromPaths();

        // Initialize GenerationPipelines for each model
        initializeGenerationPipelines();

        log.info("MultiModelPipeline created with {} models, DSP={}, CUDA graphs={}, memory trimming={}",
                modelRegistry.size(),
                config.isDspEnabled(),
                config.isCudaGraphsEnabled(),
                config.isTrimMemoryBetweenStages());
    }

    /**
     * Register a model with the pipeline.
     *
     * @param modelId the unique model ID
     * @param modelType the model type (defines role and prompt)
     * @param model the loaded SameDiff model
     */
    public void registerModel(String modelId, ModelType modelType, SameDiff model) {
        checkNotClosed();
        
        // Check for model reuse
        int modelHash = System.identityHashCode(model);
        String existingModelId = modelIdentityMap.get(modelHash);
        if (existingModelId != null) {
            log.info("Model {} is the same instance as {}, reusing", modelId, existingModelId);
        } else {
            modelIdentityMap.put(modelHash, modelId);
        }

        MultiModelPipelineConfig.RegisteredModel registeredModel = 
                MultiModelPipelineConfig.RegisteredModel.builder()
                        .modelId(modelId)
                        .modelType(modelType)
                        .model(model)
                        .ownedByPipeline(false)
                        .dspEnabled(config.isDspEnabled())
                        .build();

        modelRegistry.put(modelId, registeredModel);
        log.info("Registered model '{}' with role {}", modelId, modelType.getRole());
    }

    /**
     * Execute a pipeline with the given stages and input.
     *
     * @param stages the pipeline stages to execute
     * @param input the initial input text
     * @return the pipeline result with all stage outputs
     */
    public PipelineResult execute(List<PipelineStage> stages, String input) {
        checkNotClosed();

        if (stages == null || stages.isEmpty()) {
            throw new IllegalArgumentException("At least one pipeline stage is required");
        }

        log.info("Executing pipeline with {} stages", stages.size());

        PipelineResult result = new PipelineResult();
        String currentInput = input;

        // Track which models have been used for DSP optimization
        Set<String> usedModels = new HashSet<>();

        for (int i = 0; i < stages.size(); i++) {
            PipelineStage stage = stages.get(i);
            log.info("Executing stage {}/{}: {} (model: {})", 
                    i + 1, stages.size(), stage.getStageId(), stage.getModelId());

            try {
                // Execute the stage
                StageOutput stageOutput = executeStage(stage, currentInput, result, usedModels);
                result.addStageOutput(stage.getStageId(), stageOutput);

                // Update input for next stage
                currentInput = prepareInputForNextStage(stageOutput, stage, result);

                // Memory management: free memory after stage if configured
                if (config.isFreeMemoryPerStage() && i < stages.size() - 1) {
                    // Don't free if the same model is used in the next stage
                    String nextModelId = stages.get(i + 1).getModelId();
                    if (!nextModelId.equals(stage.getModelId())) {
                        trimMemoryForModel(stage.getModelId());
                    }
                }

                // Trim GPU memory pools between stages if configured
                if (config.isTrimMemoryBetweenStages() && i < stages.size() - 1) {
                    trimGpuMemoryPools();
                }

                usedModels.add(stage.getModelId());

            } catch (Exception e) {
                log.error("Pipeline failed at stage {}/{} ({}): {}", 
                        i + 1, stages.size(), stage.getStageId(), e.getMessage());
                result.setErrorMessage(e.getMessage());
                result.setFailedStageIndex(i);
                throw new RuntimeException("Pipeline execution failed at stage " + stage.getStageId(), e);
            }
        }

        result.setSuccess(true);
        log.info("Pipeline completed successfully with {} stages", stages.size());
        return result;
    }

    /**
     * Execute a single pipeline stage.
     */
    private StageOutput executeStage(PipelineStage stage, String input, PipelineResult context, Set<String> usedModels) throws IOException {
        String modelId = stage.getModelId();
        MultiModelPipelineConfig.RegisteredModel registeredModel = modelRegistry.get(modelId);
        
        if (registeredModel == null) {
            throw new IllegalArgumentException("Model not found: " + modelId);
        }

        ModelType modelType = registeredModel.getModelType();
        SameDiff model = registeredModel.getModel();
        GenerationPipeline pipeline = generationPipelines.get(modelId);

        if (pipeline == null) {
            throw new IllegalStateException("GenerationPipeline not initialized for model: " + modelId);
        }

        // Build prompt with template
        Map<String, String> params = new HashMap<>(modelType.getDefaultParams());
        if (config.getRoleParameters().containsKey(modelId)) {
            params.putAll(config.getRoleParameters().get(modelId));
        }
        params.putAll(stage.getParameters());

        String prompt = modelType.buildPrompt(input, params);
        log.debug("Stage {} prompt (first 200 chars): {}", stage.getStageId(), 
                prompt.length() > 200 ? prompt.substring(0, 200) + "..." : prompt);

        // DSP optimization: disable during first invocation, enable after compilation
        boolean dspWasEnabled = false;
        String prevCudaGraphs = null;
        
        if (config.isDspEnabled() && !usedModels.contains(modelId)) {
            // First invocation: disable DSP to reduce peak memory during compilation
            dspWasEnabled = InferenceSession.isDynamicShapePlanEnabled();
            prevCudaGraphs = System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED);
            InferenceSession.setDynamicShapePlanEnabled(false);
            System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, "false");
            log.debug("DSP disabled for first invocation of model {}", modelId);
        }

        try {
            // Generate output
            GenerationResult generationResult = pipeline.generate(prompt, config.getMaxTokensPerModel());
            String rawOutput = generationResult.getText();

            // Parse output according to model type
            ModelType.ParsedOutput parsedOutput = modelType.parseOutput(rawOutput);

            log.info("Stage {} completed: {} tokens, {} ms", 
                    stage.getStageId(), 
                    generationResult.getTokenIds() != null ? generationResult.getTokenIds().length : 0,
                    generationResult.getGenerationTimeMs());

            return StageOutput.builder()
                    .rawText(rawOutput)
                    .parsedOutput(parsedOutput)
                    .stageId(stage.getStageId())
                    .modelId(modelId)
                    .modelRole(modelType.getRole())
                    .durationMs(generationResult.getGenerationTimeMs())
                    .tokenCount(generationResult.getTokenIds() != null ? generationResult.getTokenIds().length : 0)
                    .build();

        } finally {
            // Re-enable DSP after first compilation
            if (config.isDspEnabled() && !usedModels.contains(modelId)) {
                InferenceSession.setDynamicShapePlanEnabled(dspWasEnabled);
                if (prevCudaGraphs != null) {
                    System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, prevCudaGraphs);
                }
                usedModels.add(modelId);
                log.debug("DSP re-enabled for model {} after compilation", modelId);
            }
        }
    }

    /**
     * Prepare input for the next stage from the current stage output.
     */
    private String prepareInputForNextStage(StageOutput output, PipelineStage currentStage, PipelineResult context) {
        // By default, use the raw text output
        // Subclasses can override to use structured output
        return output.getRawText();
    }

    /**
     * Trim memory for a specific model.
     */
    private void trimMemoryForModel(String modelId) {
        if (!config.isFreeMemoryPerStage()) {
            return;
        }

        MultiModelPipelineConfig.RegisteredModel model = modelRegistry.get(modelId);
        if (model == null || model.getModel() == null) {
            return;
        }

        SameDiff sd = model.getModel();
        try {
            // Reset DSP executor for next use
            if (config.isDspEnabled()) {
                DynamicShapePlanExecutor dsp = sd.getOrCreateSession().getDynamicShapePlanExecutor();
                if (dsp != null) {
                    dsp.resetForNextPage();
                }
            }

            // Clear placeholders and op inputs
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
            sd.resetSession();

            log.debug("Trimmed memory for model {}", modelId);
        } catch (Exception e) {
            log.warn("Error trimming memory for model {}: {}", modelId, e.getMessage());
        }
    }

    /**
     * Trim GPU memory pools across all devices.
     */
    private void trimGpuMemoryPools() {
        if (!config.isTrimMemoryBetweenStages()) {
            return;
        }

        try {
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }
            
            log.debug("Trimmed GPU memory pools on {} devices", numDevices);
        } catch (Exception e) {
            log.debug("Error trimming GPU memory pools: {}", e.getMessage());
        }
    }

    /**
     * Load models from registered paths.
     */
    private void loadModelsFromPaths() throws IOException {
        for (Map.Entry<String, MultiModelPipelineConfig.RegisteredModel> entry : modelRegistry.entrySet()) {
            MultiModelPipelineConfig.RegisteredModel registeredModel = entry.getValue();
            
            if (registeredModel.getModel() == null && registeredModel.getModelPath() != null) {
                log.info("Loading model '{}' from path: {}", entry.getKey(), registeredModel.getModelPath());
                
                File modelFile = new File(registeredModel.getModelPath());
                if (!modelFile.exists()) {
                    throw new IOException("Model file not found: " + registeredModel.getModelPath());
                }

                SameDiff model = loadModel(modelFile);
                
                MultiModelPipelineConfig.RegisteredModel updated = 
                        MultiModelPipelineConfig.RegisteredModel.builder()
                                .modelId(registeredModel.getModelId())
                                .modelType(registeredModel.getModelType())
                                .model(model)
                                .modelPath(registeredModel.getModelPath())
                                .ownedByPipeline(registeredModel.isOwnedByPipeline())
                                .dspEnabled(registeredModel.isDspEnabled())
                                .generationPipeline(registeredModel.getGenerationPipeline())
                                .build();
                modelRegistry.put(entry.getKey(), updated);
                
                log.info("Loaded model '{}' ({} ops)", entry.getKey(), model.getOps().size());
            }
        }
    }

    /**
     * Initialize GenerationPipelines for all registered models.
     */
    private void initializeGenerationPipelines() throws IOException {
        for (Map.Entry<String, MultiModelPipelineConfig.RegisteredModel> entry : modelRegistry.entrySet()) {
            String modelId = entry.getKey();
            MultiModelPipelineConfig.RegisteredModel registeredModel = entry.getValue();
            SameDiff model = registeredModel.getModel();

            if (model == null) {
                throw new IllegalStateException("Model not loaded: " + modelId);
            }

            // Enable DSP if configured
            if (config.isDspEnabled() && registeredModel.isDspEnabled()) {
                boolean noFreeze = Boolean.parseBoolean(
                        System.getProperty(ND4JSystemProperties.DSP_NO_FREEZE, "false"));
                if (!noFreeze) {
                    model.setDspAutoCompileEnabled(true);
                    model.setDspNativeAutoCompileEnabled(true);
                    log.info("DSP auto-compile enabled for model {}", modelId);
                }
            }

            // Create GenerationPipeline for the model with full KV cache support
            GenerationPipeline pipeline = GenerationPipeline.create(
                    org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig.builder()
                            .decoder(model)
                            .embedTokens(model) // Reuse decoder for simplicity (production needs separate embed_tokens)
                            .tokenizer(tokenizer)
                            .samplingConfig(config.getSamplingConfig())
                            .maxNewTokens(config.getMaxTokensPerModel())
                            .dspEnabled(config.isDspEnabled())
                            .kvCacheStrategy(config.getKvCacheStrategy())
                            .turboQuantBits(config.getTurboQuantBits())
                            .build());

            generationPipelines.put(modelId, pipeline);
            log.info("Initialized GenerationPipeline for model {} with KV cache strategy={}", 
                    modelId, config.getKvCacheStrategy());
        }
    }

    /**
     * Load a SameDiff model from a file.
     */
    private SameDiff loadModel(File modelFile) throws IOException {
        String name = modelFile.getName().toLowerCase();
        
        if (name.endsWith(".sdz") || name.endsWith(".sdnb") || name.endsWith(".fb")) {
            log.info("Loading SameDiff model: {}", modelFile.getName());
            long start = System.currentTimeMillis();
            SameDiff sd = SameDiff.load(modelFile, false);
            log.info("Loaded model in {}ms: {}", System.currentTimeMillis() - start, modelFile.getName());
            return sd;
        }

        throw new IOException("Unsupported model format: " + modelFile.getName() + 
                ". Only SDZ/SDNB/FB formats are supported natively.");
    }

    /**
     * Get a registered model by ID.
     */
    public MultiModelPipelineConfig.RegisteredModel getModel(String modelId) {
        return modelRegistry.get(modelId);
    }

    /**
     * Get all registered model IDs.
     */
    public Set<String> getModelIds() {
        return Collections.unmodifiableSet(modelRegistry.keySet());
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("MultiModelPipeline has been closed");
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        log.info("Closing MultiModelPipeline...");

        // Close all GenerationPipelines
        for (Map.Entry<String, GenerationPipeline> entry : generationPipelines.entrySet()) {
            try {
                entry.getValue().close();
            } catch (Exception e) {
                log.warn("Error closing GenerationPipeline for {}: {}", entry.getKey(), e.getMessage());
            }
        }
        generationPipelines.clear();

        // Free owned models
        for (Map.Entry<String, MultiModelPipelineConfig.RegisteredModel> entry : modelRegistry.entrySet()) {
            MultiModelPipelineConfig.RegisteredModel model = entry.getValue();
            if (model.isOwnedByPipeline() && model.getModel() != null) {
                try {
                    SameDiffMemoryUtils.freeModelArrays(model.getModel());
                    model.getModel().close();
                    log.info("Freed owned model: {}", entry.getKey());
                } catch (Exception e) {
                    log.warn("Error freeing model {}: {}", entry.getKey(), e.getMessage());
                }
            }
        }

        // Trim GPU memory pools
        trimGpuMemoryPools();

        log.info("MultiModelPipeline closed");
    }
}
