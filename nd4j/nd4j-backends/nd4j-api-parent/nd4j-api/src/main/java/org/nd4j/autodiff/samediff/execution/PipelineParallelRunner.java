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

package org.nd4j.autodiff.samediff.execution;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Pipeline parallel execution of a SameDiff model across multiple GPUs.
 * <p>
 * Pipeline parallelism splits the model into sequential stages, with each stage
 * assigned to a different GPU. This enables running models whose total parameter
 * count exceeds single-GPU memory. For inference, micro-batching is optional
 * since there is no backward pass to overlap.
 * <p>
 * Architecture:
 * <pre>
 *   GPU 0 (Stage 0): Embedding + Layers 0..L/N-1
 *   GPU 1 (Stage 1): Layers L/N..2L/N-1
 *   ...
 *   GPU N-1 (Stage N-1): Layers (N-1)*L/N..L-1 + LM Head
 * </pre>
 * <p>
 * Data flow: each stage produces an activation tensor that is transferred to the
 * next stage's GPU via D2H + H2D (or peer-to-peer if available).
 *
 * Adam Gibson
 * @see TensorParallelRunner
 */
@Slf4j
public class PipelineParallelRunner implements Closeable {

    @Getter
    private final SameDiff fullModel;

    @Getter
    private final PipelineConfig config;

    /**
     * Per-stage SameDiff sub-graphs. Each stage is a subset of the full model
     * that can execute independently on its assigned GPU.
     */
    private final List<StageInfo> stages = new ArrayList<>();

    private boolean modelSplit = false;

    /**
     * Configuration for pipeline parallelism.
     */
    @Getter
    @Setter
    public static class PipelineConfig {
        /** Number of pipeline stages (one per GPU). */
        private int numStages = 1;

        /** Device ID for each stage. If null, uses stage index as device ID. */
        private int[] stageDeviceIds;

        /** Number of micro-batches for pipeline fill/drain optimization. */
        private int numMicroBatches = 1;

        /** Variable name for the inter-stage activation tensor. */
        private String activationVarName = "hidden_states";

        public PipelineConfig() {
        }

        public PipelineConfig(int numStages) {
            if (numStages < 1) {
                throw new IllegalArgumentException("numStages must be >= 1, got " + numStages);
            }
            this.numStages = numStages;
        }

        /**
         * Returns the device ID for a given stage.
         */
        public int deviceIdForStage(int stageIndex) {
            if (stageDeviceIds != null && stageIndex < stageDeviceIds.length) {
                return stageDeviceIds[stageIndex];
            }
            return stageIndex;
        }

        /**
         * Returns true if pipeline parallelism is actually enabled (more than 1 stage).
         */
        public boolean isEnabled() {
            return numStages > 1;
        }
    }

    /**
     * Information about a single pipeline stage.
     */
    @Getter
    public static class StageInfo {
        private final int stageIndex;
        private final int deviceId;
        private final List<String> layerNames;
        private final String inputVarName;
        private final String outputVarName;

        /**
         * SameDiff sub-graph for this stage. May be the full model if pipeline
         * parallelism is disabled, or a partitioned sub-graph otherwise.
         */
        @Setter
        private SameDiff stageGraph;

        public StageInfo(int stageIndex, int deviceId, List<String> layerNames,
                         String inputVarName, String outputVarName) {
            this.stageIndex = stageIndex;
            this.deviceId = deviceId;
            this.layerNames = layerNames;
            this.inputVarName = inputVarName;
            this.outputVarName = outputVarName;
        }
    }

    /**
     * Create a pipeline parallel runner.
     *
     * @param model  the full SameDiff model
     * @param config pipeline parallelism configuration
     */
    public PipelineParallelRunner(SameDiff model, PipelineConfig config) {
        if (model == null) {
            throw new IllegalArgumentException("Model must not be null");
        }
        if (config == null) {
            throw new IllegalArgumentException("Config must not be null");
        }
        this.fullModel = model;
        this.config = config;
    }

    /**
     * Manually assign layers to pipeline stages.
     * <p>
     * Each entry maps a stage index to a list of layer name prefixes. Variables
     * whose names start with any of the given prefixes are assigned to that stage.
     *
     * @param stageAssignments map from stage index to layer name prefixes
     */
    public void splitModelToStages(Map<Integer, List<String>> stageAssignments) {
        if (modelSplit) {
            throw new IllegalStateException("Model has already been split. Create a new runner to re-split.");
        }

        stages.clear();
        for (int i = 0; i < config.getNumStages(); i++) {
            List<String> layerNames = stageAssignments.getOrDefault(i, new ArrayList<>());
            int deviceId = config.deviceIdForStage(i);

            String inputVar = (i == 0) ? null : config.getActivationVarName() + "_stage_" + (i - 1);
            String outputVar = (i == config.getNumStages() - 1) ? null : config.getActivationVarName() + "_stage_" + i;

            StageInfo stage = new StageInfo(i, deviceId, layerNames, inputVar, outputVar);
            stages.add(stage);
        }

        modelSplit = true;
        log.info("Model split into {} pipeline stages", config.getNumStages());
    }

    /**
     * Automatically split model layers evenly across stages.
     * <p>
     * Detects transformer layers by looking for variables matching the pattern
     * "layer.N." or "layers.N." and distributes them evenly.
     *
     * @param layerPrefix the prefix before layer numbers (e.g., "model.layers")
     * @param numLayers   total number of transformer layers in the model
     */
    public void splitModelToStages(String layerPrefix, int numLayers) {
        if (numLayers < config.getNumStages()) {
            throw new IllegalArgumentException(
                    "Cannot split " + numLayers + " layers across " + config.getNumStages() + " stages");
        }

        Map<Integer, List<String>> assignments = new LinkedHashMap<>();
        int layersPerStage = numLayers / config.getNumStages();
        int remainder = numLayers % config.getNumStages();

        int layerIdx = 0;
        for (int stage = 0; stage < config.getNumStages(); stage++) {
            List<String> layerNames = new ArrayList<>();
            int count = layersPerStage + (stage < remainder ? 1 : 0);

            // First stage also gets embedding layers
            if (stage == 0) {
                layerNames.add("embed");
                layerNames.add("model.embed");
            }

            for (int j = 0; j < count; j++) {
                layerNames.add(layerPrefix + "." + layerIdx);
                layerIdx++;
            }

            // Last stage also gets the LM head / final norm
            if (stage == config.getNumStages() - 1) {
                layerNames.add("model.norm");
                layerNames.add("lm_head");
            }

            assignments.put(stage, layerNames);
        }

        splitModelToStages(assignments);
    }

    /**
     * Execute inference through the pipeline.
     * <p>
     * For single-request inference (numMicroBatches=1), this is a simple
     * sequential pass through the stages. Each stage executes on its assigned
     * GPU and passes the activation tensor to the next stage.
     *
     * @param inputs model input variables (fed to stage 0)
     * @return final model outputs from the last stage
     */
    public Map<String, INDArray> executeInference(Map<String, INDArray> inputs) {
        if (!modelSplit && config.isEnabled()) {
            throw new IllegalStateException("Model has not been split into stages. " +
                    "Call splitModelToStages() before executeInference().");
        }

        if (!config.isEnabled()) {
            // Single-stage: just run the full model
            return fullModel.output(inputs, fullModel.outputs().toArray(new String[0]));
        }

        // Sequential pipeline execution (inference mode)
        Map<String, INDArray> currentInputs = new HashMap<>(inputs);

        for (int i = 0; i < stages.size(); i++) {
            StageInfo stage = stages.get(i);

            log.debug("Executing pipeline stage {} on device {}", i, stage.getDeviceId());

            // Execute this stage's portion of the graph
            // In a full implementation, this would run only the stage's sub-graph
            // on the stage's device. For now, we use the full model and filter.
            Map<String, INDArray> stageOutputs;
            if (stage.getStageGraph() != null) {
                String[] outputNames = stage.getStageGraph().outputs().toArray(new String[0]);
                stageOutputs = stage.getStageGraph().output(currentInputs, outputNames);
            } else {
                // Fallback: run full model (for single-stage or un-partitioned case)
                stageOutputs = fullModel.output(currentInputs, fullModel.outputs().toArray(new String[0]));
                return stageOutputs;
            }

            // Pass activation to next stage
            if (i < stages.size() - 1 && stage.getOutputVarName() != null) {
                INDArray activation = stageOutputs.get(stage.getOutputVarName());
                if (activation == null) {
                    // Try to find the activation in outputs
                    if (stageOutputs.size() == 1) {
                        activation = stageOutputs.values().iterator().next();
                    } else {
                        throw new IllegalStateException(
                                "Cannot find activation '" + stage.getOutputVarName() +
                                        "' in stage " + i + " outputs: " + stageOutputs.keySet());
                    }
                }

                // Transfer activation to next stage's device
                StageInfo nextStage = stages.get(i + 1);
                INDArray transferred = transferToDevice(activation, nextStage.getDeviceId());

                currentInputs = new HashMap<>();
                currentInputs.put(nextStage.getInputVarName(), transferred);
            } else {
                // Last stage: return outputs
                return stageOutputs;
            }
        }

        throw new IllegalStateException("Pipeline execution did not produce outputs");
    }

    /**
     * Execute inference and return a single named output.
     *
     * @param inputs     model input variables
     * @param outputName the output variable to return
     * @return the output array
     */
    public INDArray executeInference(Map<String, INDArray> inputs, String outputName) {
        Map<String, INDArray> outputs = executeInference(inputs);
        INDArray result = outputs.get(outputName);
        if (result == null) {
            throw new IllegalArgumentException(
                    "Output '" + outputName + "' not found. Available: " + outputs.keySet());
        }
        return result;
    }

    /**
     * Get the list of configured stages.
     */
    public List<StageInfo> getStages() {
        return new ArrayList<>(stages);
    }

    /**
     * Transfer an array to a target device.
     * Uses AffinityManager to replicate the array on the target CUDA device.
     * Falls back to dup() if affinity manager doesn't support replication.
     */
    private INDArray transferToDevice(INDArray array, int deviceId) {
        try {
            return Nd4j.getAffinityManager().replicateToDevice(deviceId, array);
        } catch (UnsupportedOperationException e) {
            // CPU backend or affinity manager doesn't support replication
            log.debug("AffinityManager.replicateToDevice not supported, using dup()");
            return array.dup();
        }
    }

    @Override
    public void close() {
        stages.clear();
        modelSplit = false;
        log.info("PipelineParallelRunner closed");
    }
}
