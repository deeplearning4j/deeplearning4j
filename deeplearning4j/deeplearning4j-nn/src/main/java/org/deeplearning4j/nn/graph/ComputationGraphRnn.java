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

package org.deeplearning4j.nn.graph;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.FwdPassType;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.util.CrashReportingUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Helper class that contains RNN-specific functionality extracted from {@link ComputationGraph}.
 * <p>
 * This class holds a reference to a {@link ComputationGraph} and delegates to its package-visible
 * methods and fields to carry out all RNN operations. The public API of {@code ComputationGraph}
 * is unchanged; every public RNN method on that class is a one-line delegate to this helper.
 */
@Slf4j
class ComputationGraphRnn {

    private final ComputationGraph graph;

    ComputationGraphRnn(ComputationGraph graph) {
        this.graph = graph;
    }

    // -------------------------------------------------------------------------
    // rnnTimeStep
    // -------------------------------------------------------------------------

    /**
     * If this ComputationGraph contains one or more RNN layers: conduct forward pass (prediction)
     * but using previous stored state for any RNN layers. The activations for the final step are
     * also stored in the RNN layers for use next time rnnTimeStep() is called.<br>
     * This method can be used to generate output one or more steps at a time instead of always having to do
     * forward pass from t=0. Example uses are for streaming data, and for generating samples from network output
     * one step at a time (where samples are then fed back into the network as input)<br>
     * If no previous state is present in RNN layers (i.e., initially or after calling rnnClearPreviousState()),
     * the default initialization (usually 0) is used.<br>
     * Supports mini-batch (i.e., multiple predictions/forward pass in parallel) as well as for single examples.<br>
     *
     * @param inputs Input to network. May be for one or multiple time steps. For single time step:
     *               input has shape [miniBatchSize,inputSize] or [miniBatchSize,inputSize,1]. miniBatchSize=1 for single example.<br>
     *               For multiple time steps: [miniBatchSize,inputSize,inputTimeSeriesLength]
     * @return Output activations. If output is RNN layer (such as RnnOutputLayer): if all inputs have shape [miniBatchSize,inputSize]
     * i.e., is 2d, then outputs have shape [miniBatchSize,outputSize] (i.e., also 2d) instead of [miniBatchSize,outputSize,1].<br>
     * Otherwise output is 3d [miniBatchSize,outputSize,inputTimeSeriesLength] when using RnnOutputLayer (or unmodified otherwise).
     */
    INDArray[] rnnTimeStep(INDArray... inputs) {
        return rnnTimeStepHelper(null, inputs);
    }

    /**
     * See {@link ComputationGraph#rnnTimeStep(INDArray...)} for details.<br>
     * If no memory workspace is provided, the output will be detached (not in any workspace).<br>
     * If a memory workspace is provided, the output activation array (i.e., the INDArray returned by this method)
     * will be placed in the specified workspace. This workspace must be opened by the user before calling this method -
     * and the user is responsible for (a) closing this workspace, and (b) ensuring the output array is not used out
     * of scope (i.e., not used after closing the workspace to which it belongs - as this is likely to cause either
     * an exception when used, or a crash).
     *
     * @param outputWorkspace Output workspace. May be null
     * @param inputs          Input activations
     * @return The output/activations from the network (either detached or in the specified workspace if provided)
     */
    INDArray[] rnnTimeStep(MemoryWorkspace outputWorkspace, INDArray... inputs) {
        try {
            return rnnTimeStepHelper(outputWorkspace, inputs);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(graph, e);
            throw e;
        }
    }

    private INDArray[] rnnTimeStepHelper(MemoryWorkspace outputWs, INDArray... inputs) {
        boolean inputIs2d = true;
        for (INDArray i : inputs) {
            if (i.rank() != 2) {
                inputIs2d = false;
                break;
            }
        }

        INDArray[] outputs = graph.outputOfLayersDetached(false, FwdPassType.RNN_TIMESTEP,
                graph.getOutputLayerIndices(), inputs, null, null, true, false, outputWs);

        //As per MultiLayerNetwork.rnnTimeStep(): if inputs are all 2d, then outputs are all 2d
        if (inputIs2d) {
            for (int i = 0; i < outputs.length; i++) {
                if (outputs[i].rank() == 3 && outputs[i].size(2) == 1) {
                    //Return 2d output with shape [miniBatchSize,nOut]
                    // instead of 3d output with shape [miniBatchSize,nOut,1]
                    outputs[i] = outputs[i].tensorAlongDimension(0, 1, 0);
                }
            }
        }

        graph.clearInputs();
        return outputs;
    }

    // -------------------------------------------------------------------------
    // rnnGetPreviousState / rnnGetPreviousStates
    // -------------------------------------------------------------------------

    /**
     * Get the state of the RNN layer, as used in {@link ComputationGraph#rnnTimeStep(INDArray...)}.
     *
     * @param layer Number/index of the layer.
     * @return Hidden state, or null if layer is not an RNN layer
     */
    Map<String, INDArray> rnnGetPreviousState(int layer) {
        return rnnGetPreviousState(graph.layers[layer].conf().getLayer().getLayerName());
    }

    /**
     * Get the state of the RNN layer, as used in {@link ComputationGraph#rnnTimeStep(INDArray...)}.
     *
     * @param layerName name of the layer
     * @return Hidden state, or null if layer is not an RNN layer
     */
    Map<String, INDArray> rnnGetPreviousState(String layerName) {
        Layer l = graph.verticesMap.get(layerName).getLayer();
        if (l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) {
            l = ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) l).getUnderlying();
        }
        if (l == null || !(l instanceof RecurrentLayer))
            return null;
        return ((RecurrentLayer) l).rnnGetPreviousState();
    }

    /**
     * Get a map of states for ALL RNN layers, as used in {@link ComputationGraph#rnnTimeStep(INDArray...)}.
     * Layers that are not RNN layers will not have an entry in the returned map
     *
     * @return Map of states (keyed by layer name) or null if layer is not an RNN layer
     * @see ComputationGraph#rnnSetPreviousStates(Map)
     */
    Map<String, Map<String, INDArray>> rnnGetPreviousStates() {
        Map<String, Map<String, INDArray>> states = new HashMap<>();
        for (Layer l : graph.layers) {
            if (l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) {
                l = ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) l).getUnderlying();
            }
            if (l instanceof RecurrentLayer) {
                states.put(l.conf().getLayer().getLayerName(), ((RecurrentLayer) l).rnnGetPreviousState());
            }
        }
        return states;
    }

    // -------------------------------------------------------------------------
    // rnnSetPreviousState / rnnSetPreviousStates
    // -------------------------------------------------------------------------

    /**
     * Set the state of the RNN layer, for use in {@link ComputationGraph#rnnTimeStep(INDArray...)}
     *
     * @param layer The number/index of the layer.
     * @param state The state to set the specified layer to
     */
    void rnnSetPreviousState(int layer, Map<String, INDArray> state) {
        rnnSetPreviousState(graph.layers[layer].conf().getLayer().getLayerName(), state);
    }

    /**
     * Set the state of the RNN layer, for use in {@link ComputationGraph#rnnTimeStep(INDArray...)}
     *
     * @param layerName The name of the layer.
     * @param state     The state to set the specified layer to
     */
    void rnnSetPreviousState(String layerName, Map<String, INDArray> state) {
        Layer l = graph.verticesMap.get(layerName).getLayer();
        if (l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) {
            l = ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) l).getUnderlying();
        }
        if (l == null || !(l instanceof RecurrentLayer)) {
            throw new UnsupportedOperationException(
                    "Layer \"" + layerName + "\" is not a recurrent layer. Cannot set state");
        }
        ((RecurrentLayer) l).rnnSetPreviousState(state);
    }

    /**
     * Set the states for all RNN layers, for use in {@link ComputationGraph#rnnTimeStep(INDArray...)}
     *
     * @param previousStates The previous time step states for all layers (key: layer name. Value: layer states)
     * @see ComputationGraph#rnnGetPreviousStates()
     */
    void rnnSetPreviousStates(Map<String, Map<String, INDArray>> previousStates) {
        for (Map.Entry<String, Map<String, INDArray>> entry : previousStates.entrySet()) {
            rnnSetPreviousState(entry.getKey(), entry.getValue());
        }
    }

    // -------------------------------------------------------------------------
    // rnnClearPreviousState
    // -------------------------------------------------------------------------

    /**
     * Clear the previous state of the RNN layers (if any), used in
     * {@link ComputationGraph#rnnTimeStep(INDArray...)}
     */
    void rnnClearPreviousState() {
        if (graph.layers == null)
            return;
        for (Layer layer : graph.layers) {
            if (layer instanceof RecurrentLayer)
                ((RecurrentLayer) layer).rnnClearPreviousState();
            else if (layer instanceof MultiLayerNetwork) {
                ((MultiLayerNetwork) layer).rnnClearPreviousState();
            }
        }
    }

    // -------------------------------------------------------------------------
    // rnnActivateUsingStoredState
    // -------------------------------------------------------------------------

    /**
     * Similar to rnnTimeStep and feedForward() methods. Difference here is that this method:<br>
     * (a) like rnnTimeStep does forward pass using stored state for RNN layers, and<br>
     * (b) unlike rnnTimeStep does not modify the RNN layer state<br>
     * Therefore multiple calls to this method with the same input should have the same output.<br>
     * Typically used during training only. Use rnnTimeStep for prediction/forward pass at test time.
     *
     * @param inputs            Input to network
     * @param training          Whether training or not
     * @param storeLastForTBPTT set to true if used as part of truncated BPTT training
     * @return Activations for each layer (including input, as per feedforward() etc)
     */
    Map<String, INDArray> rnnActivateUsingStoredState(INDArray[] inputs, boolean training,
                                                      boolean storeLastForTBPTT) {
        return graph.ffToLayerActivationsDetached(training, FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE,
                storeLastForTBPTT, graph.vertices.length - 1,
                null, inputs, graph.getInputMaskArrays(), graph.getLabelMaskArrays(), true);
    }

    // -------------------------------------------------------------------------
    // doTruncatedBPTT
    // -------------------------------------------------------------------------

    /**
     * Fit the network using truncated BPTT
     */
    void doTruncatedBPTT(INDArray[] inputs, INDArray[] labels, INDArray[] featureMasks,
                         INDArray[] labelMasks, LayerWorkspaceMgr workspaceMgr) {
        if (graph.flattenedGradients == null) {
            graph.initGradientsView();
        }

        //Approach used here to implement truncated BPTT: if input is 3d, split it. Otherwise: input is unmodified
        long timeSeriesLength = -1;
        for (INDArray in : inputs) {
            if (in.rank() != 3)
                continue;
            if (timeSeriesLength == -1)
                timeSeriesLength = in.size(2);
            else if (timeSeriesLength != in.size(2)) {
                log.warn("Cannot do TBPTT with time series of different lengths");
                return;
            }
        }
        for (INDArray out : labels) {
            if (out.rank() != 3)
                continue;
            if (timeSeriesLength == -1)
                timeSeriesLength = out.size(2);
            else if (timeSeriesLength != out.size(2)) {
                log.warn("Cannot do TBPTT with time series of different lengths");
                return;
            }
        }

        long fwdLen = graph.configuration.getTbpttFwdLength();
        long nSubsets = timeSeriesLength / fwdLen;
        if (timeSeriesLength % fwdLen != 0)
            nSubsets++;

        rnnClearPreviousState();

        for (int i = 0; i < nSubsets; i++) {
            long startTimeIdx = i * fwdLen;
            long endTimeIdx = startTimeIdx + fwdLen;
            if (endTimeIdx > timeSeriesLength)
                endTimeIdx = timeSeriesLength;

            if (startTimeIdx > Integer.MAX_VALUE)
                throw new ND4JArraySizeException();
            List<INDArray[]> list = getSubsetsForTbptt((int) startTimeIdx, endTimeIdx, inputs, labels,
                    featureMasks, labelMasks);

            graph.setInputs(list.get(0));
            graph.setLabels(list.get(1));
            graph.setLayerMaskArrays(list.get(2), list.get(3));

            if (graph.solver == null) {
                try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    graph.solver = new Solver.Builder().configure(graph.conf()).listeners(graph.getListeners())
                            .model(graph).build();
                }
            }
            graph.solver.optimize(workspaceMgr);

            //Finally, update the state of the RNN layers:
            rnnUpdateStateWithTBPTTState();
        }

        if (graph.clearTbpttState) {
            rnnClearPreviousState();
        }
        graph.clearLayerMaskArrays();
    }

    // -------------------------------------------------------------------------
    // getSubsetsForTbptt
    // -------------------------------------------------------------------------

    List<INDArray[]> getSubsetsForTbptt(int startTimeIdx, long endTimeIdx, INDArray[] inputs, INDArray[] labels,
                                        INDArray[] featureMasks, INDArray[] labelMasks) {
        INDArray[] newInputs = new INDArray[inputs.length];
        INDArray[] newLabels = new INDArray[labels.length];
        INDArray[] newFeatureMasks = (featureMasks != null ? new INDArray[featureMasks.length] : null);
        INDArray[] newLabelMasks = (labelMasks != null ? new INDArray[labelMasks.length] : null);

        for (int j = 0; j < inputs.length; j++) {
            if (inputs[j].rank() != 3)
                newInputs[j] = inputs[j];
            else {
                newInputs[j] = inputs[j].get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(startTimeIdx, endTimeIdx));
            }
        }
        for (int j = 0; j < labels.length; j++) {
            if (labels[j].rank() != 3)
                newLabels[j] = labels[j];
            else {
                newLabels[j] = labels[j].get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(startTimeIdx, endTimeIdx));
            }
        }
        if (featureMasks != null) {
            for (int j = 0; j < featureMasks.length; j++) {
                if (featureMasks[j] == null)
                    continue;
                newFeatureMasks[j] = featureMasks[j].get(NDArrayIndex.all(),
                        NDArrayIndex.interval(startTimeIdx, endTimeIdx));
            }
        }
        if (labelMasks != null) {
            for (int j = 0; j < labelMasks.length; j++) {
                if (labelMasks[j] == null)
                    continue;
                newLabelMasks[j] = labelMasks[j].get(NDArrayIndex.all(),
                        NDArrayIndex.interval(startTimeIdx, endTimeIdx));
            }
        }

        return Arrays.asList(newInputs, newLabels, newFeatureMasks, newLabelMasks);
    }

    // -------------------------------------------------------------------------
    // rnnUpdateStateWithTBPTTState
    // -------------------------------------------------------------------------

    /**
     * Update the internal state of RNN layers after a truncated BPTT fit call
     */
    void rnnUpdateStateWithTBPTTState() {
        for (int i = 0; i < graph.layers.length; i++) {
            if (graph.layers[i] instanceof RecurrentLayer) {
                RecurrentLayer l = ((RecurrentLayer) graph.layers[i]);
                l.rnnSetPreviousState(l.rnnGetTBPTTState());
            } else if (graph.layers[i] instanceof MultiLayerNetwork) {
                ((MultiLayerNetwork) graph.layers[i]).updateRnnStateWithTBPTTState();
            }
        }
    }
}
