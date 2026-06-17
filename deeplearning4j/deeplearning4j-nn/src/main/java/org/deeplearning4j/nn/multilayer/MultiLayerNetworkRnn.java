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

package org.deeplearning4j.nn.multilayer;

import lombok.val;
import org.deeplearning4j.nn.api.FwdPassType;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.util.CrashReportingUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;
import java.util.Map;

/**
 * Helper class containing all RNN-related methods extracted from {@link MultiLayerNetwork}.
 * Each method delegates to the corresponding implementation here; the public API on
 * {@link MultiLayerNetwork} is preserved unchanged.
 */
public class MultiLayerNetworkRnn {

    private final MultiLayerNetwork network;

    public MultiLayerNetworkRnn(MultiLayerNetwork network) {
        this.network = network;
    }

    /**
     * See {@link MultiLayerNetwork#rnnTimeStep(INDArray)}.
     */
    public INDArray rnnTimeStep(INDArray input) {
        return rnnTimeStep(input, null);
    }

    /**
     * See {@link MultiLayerNetwork#rnnTimeStep(INDArray, MemoryWorkspace)}.
     */
    public INDArray rnnTimeStep(INDArray input, MemoryWorkspace outputWorkspace) {
        try {
            boolean inputIs2d = input.rank() == 2;
            Layer[] layers = network.getLayers();
            INDArray out = network.outputOfLayerDetached(false, FwdPassType.RNN_TIMESTEP,
                    layers.length - 1, input, null, null, outputWorkspace);
            if (inputIs2d && out.rank() == 3
                    && layers[layers.length - 1].type() == Layer.Type.RECURRENT) {
                return out.tensorAlongDimension(0, 1, 0);
            }
            return out;
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(network, e);
            throw e;
        }
    }

    /**
     * See {@link MultiLayerNetwork#rnnGetPreviousState(int)}.
     */
    public Map<String, INDArray> rnnGetPreviousState(int layer) {
        Layer[] layers = network.getLayers();
        if (layer < 0 || layer >= layers.length)
            throw new IllegalArgumentException("Invalid layer number");
        Layer l = layers[layer];
        if (l instanceof BaseWrapperLayer) {
            l = ((BaseWrapperLayer) l).getUnderlying();
        }
        if (!(l instanceof RecurrentLayer))
            throw new IllegalArgumentException("Layer is not an RNN layer");
        return ((RecurrentLayer) l).rnnGetPreviousState();
    }

    /**
     * See {@link MultiLayerNetwork#rnnSetPreviousState(int, Map)}.
     */
    public void rnnSetPreviousState(int layer, Map<String, INDArray> state) {
        Layer[] layers = network.getLayers();
        if (layer < 0 || layer >= layers.length)
            throw new IllegalArgumentException("Invalid layer number");
        Layer l = layers[layer];
        if (l instanceof BaseWrapperLayer) {
            l = ((BaseWrapperLayer) l).getUnderlying();
        }
        if (!(l instanceof RecurrentLayer))
            throw new IllegalArgumentException("Layer is not an RNN layer");
        ((RecurrentLayer) l).rnnSetPreviousState(state);
    }

    /**
     * See {@link MultiLayerNetwork#rnnClearPreviousState()}.
     */
    public void rnnClearPreviousState() {
        Layer[] layers = network.getLayers();
        if (layers == null)
            return;
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] instanceof RecurrentLayer)
                ((RecurrentLayer) layers[i]).rnnClearPreviousState();
            else if (layers[i] instanceof MultiLayerNetwork)
                ((MultiLayerNetwork) layers[i]).rnnClearPreviousState();
            else if (layers[i] instanceof BaseWrapperLayer
                    && ((BaseWrapperLayer) layers[i]).getUnderlying() instanceof RecurrentLayer)
                ((RecurrentLayer) ((BaseWrapperLayer) layers[i]).getUnderlying()).rnnClearPreviousState();
        }
    }

    /**
     * See {@link MultiLayerNetwork#rnnActivateUsingStoredState(INDArray, boolean, boolean)}.
     */
    public List<INDArray> rnnActivateUsingStoredState(INDArray input, boolean training,
            boolean storeLastForTBPTT) {
        Layer[] layers = network.getLayers();
        return network.ffToLayerActivationsDetached(training,
                FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE, storeLastForTBPTT,
                layers.length - 1, input, network.getMask(), null, false);
    }

    /**
     * See {@link MultiLayerNetwork#doTruncatedBPTT(INDArray, INDArray, INDArray, INDArray, LayerWorkspaceMgr)}.
     */
    public void doTruncatedBPTT(INDArray input, INDArray labels, INDArray featuresMaskArray,
            INDArray labelsMaskArray, LayerWorkspaceMgr workspaceMgr) {
        int fwdLen = network.getLayerWiseConfigurations().getTbpttFwdLength();
        val timeSeriesLength = input.size(2);
        long nSubsets = timeSeriesLength / fwdLen;
        if (timeSeriesLength % fwdLen != 0)
            nSubsets++;

        network.rnnClearPreviousState();

        for (int i = 0; i < nSubsets; i++) {
            long startTimeIdx = i * fwdLen;
            long endTimeIdx = startTimeIdx + fwdLen;
            if (endTimeIdx > timeSeriesLength)
                endTimeIdx = timeSeriesLength;

            if (startTimeIdx > Integer.MAX_VALUE || endTimeIdx > Integer.MAX_VALUE)
                throw new ND4JArraySizeException();
            INDArray[] subsets = getSubsetsForTbptt((int) startTimeIdx, (int) endTimeIdx,
                    input, labels, featuresMaskArray, labelsMaskArray);

            network.setInput(subsets[0]);
            network.setLabels(subsets[1]);
            network.setLayerMaskArrays(subsets[2], subsets[3]);

            if (network.solver == null) {
                try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    network.solver = new Solver.Builder()
                            .configure(network.conf())
                            .listeners(network.getListeners())
                            .model(network)
                            .build();
                }
            }
            network.solver.optimize(workspaceMgr);

            network.updateRnnStateWithTBPTTState();
        }

        network.rnnClearPreviousState();
        network.clearLayerMaskArrays();
    }

    /**
     * See {@link MultiLayerNetwork#getSubsetsForTbptt(int, int, INDArray, INDArray, INDArray, INDArray)}.
     */
    public INDArray[] getSubsetsForTbptt(int startTimeIdx, int endTimeIdx, INDArray input,
            INDArray labels, INDArray fMask, INDArray lMask) {
        INDArray[] out = new INDArray[4];
        out[0] = input.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startTimeIdx, endTimeIdx));
        out[1] = labels.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startTimeIdx, endTimeIdx));
        if (fMask != null) {
            out[2] = fMask.get(NDArrayIndex.all(),
                    NDArrayIndex.interval(startTimeIdx, endTimeIdx));
        }
        if (lMask != null) {
            out[3] = lMask.get(NDArrayIndex.all(),
                    NDArrayIndex.interval(startTimeIdx, endTimeIdx));
        }
        return out;
    }

    /**
     * See {@link MultiLayerNetwork#updateRnnStateWithTBPTTState()}.
     */
    public void updateRnnStateWithTBPTTState() {
        Layer[] layers = network.getLayers();
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] instanceof RecurrentLayer) {
                RecurrentLayer l = (RecurrentLayer) layers[i];
                l.rnnSetPreviousState(l.rnnGetTBPTTState());
            } else if (layers[i] instanceof MultiLayerNetwork) {
                ((MultiLayerNetwork) layers[i]).updateRnnStateWithTBPTTState();
            }
        }
    }
}
