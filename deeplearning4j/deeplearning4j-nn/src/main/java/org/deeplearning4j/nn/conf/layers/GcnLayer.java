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

package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.Map;

/**
 * Graph Convolutional Network (GCN) layer (Kipf &amp; Welling 2017) for use inside a DL4J
 * {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork} or
 * {@link org.deeplearning4j.nn.graph.ComputationGraph}.
 *
 * <p>Delegates to {@code sd.gnn().gcnConv()} which computes:
 * <pre>  out = relu?( A_norm · X · W + bias )</pre>
 *
 * <p><b>Usage:</b>
 * <ol>
 *   <li>Build a {@link MultiLayerConfiguration} including this layer.</li>
 *   <li>Call {@link #setAdjacency(SparseNDArray)} on the conf-layer object (before or after
 *       initialising the network) to supply the CSR adjacency matrix.  The method
 *       automatically applies {@code addSelfLoops().normalizeSymmetric()} (GCN-style).</li>
 *   <li>Pass node features [numNodes, nIn] as the input batch.</li>
 * </ol>
 *
 * <p><b>Constraint:</b> the adjacency matrix is baked into the SameDiff graph as a constant
 * when the network is first initialized ({@code doInit}).  Calling {@code setAdjacency} after
 * network initialization has no effect; reinitialize the network if the graph topology changes.
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties({"adjValues", "adjColIdx", "adjRowPtr", "adjRows", "adjCols"})
public class GcnLayer extends SameDiffLayer {

    // -----------------------------------------------------------------------
    // Learnable-parameter keys
    // -----------------------------------------------------------------------
    private static final String W_KEY    = "W";
    private static final String BIAS_KEY = "b";

    // -----------------------------------------------------------------------
    // Configuration fields (serialized)
    // -----------------------------------------------------------------------
    private long    nIn;
    private long    nOut;
    private boolean applyRelu = true;
    private boolean hasBias   = true;

    // -----------------------------------------------------------------------
    // Adjacency (transient — NOT serialized, NOT in param table)
    // -----------------------------------------------------------------------
    /** Non-zero values of the symmetrically-normalised adjacency [nnz]. */
    private transient INDArray adjValues;
    /** CSR column indices of the normalised adjacency [nnz], INT32. */
    private transient INDArray adjColIdx;
    /** CSR row pointers of the normalised adjacency [N+1], INT32. */
    private transient INDArray adjRowPtr;
    /** Number of nodes (rows = cols of the square adjacency). */
    private transient long adjRows;
    /** Number of columns (equals adjRows for a square graph adjacency). */
    private transient long adjCols;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    protected GcnLayer(Builder builder) {
        super(builder);
        this.nIn       = builder.nIn;
        this.nOut      = builder.nOut;
        this.applyRelu = builder.applyRelu;
        this.hasBias   = builder.hasBias;
    }

    // -----------------------------------------------------------------------
    // Adjacency setter
    // -----------------------------------------------------------------------

    /**
     * Set the adjacency matrix for this GCN layer.
     *
     * <p>The provided CSR sparse matrix is pre-processed (GCN-style):
     * <ol>
     *   <li>{@code addSelfLoops()} — add the identity to ensure each node attends to itself</li>
     *   <li>{@code normalizeSymmetric()} — {@code D̃^{-1/2}(A+I)D̃^{-1/2}}</li>
     * </ol>
     *
     * @param adj CSR adjacency matrix of shape [numNodes, numNodes]
     */
    public void setAdjacency(SparseNDArray adj) {
        // GCN normalisation: Â = D̃^{-1/2}(A+I)D̃^{-1/2}
        SparseNDArray norm = adj.normalizeSymmetric();  // normalizeSymmetric() calls addSelfLoops internally
        this.adjValues = norm.getValues();
        this.adjColIdx = norm.getColIdx();
        this.adjRowPtr = norm.getRowPtr();
        this.adjRows   = norm.rows();
        this.adjCols   = norm.cols();
    }

    // -----------------------------------------------------------------------
    // SameDiffLayer contract
    // -----------------------------------------------------------------------

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();
        params.addWeightParam(W_KEY,    nIn, nOut);
        if (hasBias) {
            params.addBiasParam(BIAS_KEY, nOut);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        for (Map.Entry<String, INDArray> e : params.entrySet()) {
            if (BIAS_KEY.equals(e.getKey())) {
                e.getValue().assign(0.0);
            } else {
                // Xavier by default (weightInit inherited from SameDiffLayer.Builder)
                WeightInitUtil.initWeights(nIn, nOut, e.getValue().shape(),
                        weightInit, null, 'c', e.getValue());
            }
        }
    }

    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput,
                                  Map<String, SDVariable> paramTable, SDVariable mask) {
        if (adjValues == null) {
            throw new IllegalStateException(
                    "GcnLayer: adjacency not set. Call setAdjacency(SparseNDArray) before the network is initialized.");
        }
        // Inject adjacency as graph constants (not learnable, not serialized)
        SDVariable aVals   = sd.constant("gcn_adj_vals",    adjValues);
        SDVariable aColIdx = sd.constant("gcn_adj_col_idx", adjColIdx);
        SDVariable aRowPtr = sd.constant("gcn_adj_row_ptr", adjRowPtr);

        SDVariable W    = paramTable.get(W_KEY);
        SDVariable bias = hasBias ? paramTable.get(BIAS_KEY) : null;

        return sd.gnn().gcnConv(layerInput, W, bias, aVals, aColIdx, aRowPtr, adjRows, adjCols, applyRelu);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return InputType.feedForward(nOut);
    }

    // -----------------------------------------------------------------------
    // Builder
    // -----------------------------------------------------------------------

    public static class Builder extends SameDiffLayer.Builder<Builder> {

        private long    nIn;
        private long    nOut;
        private boolean applyRelu = true;
        private boolean hasBias   = true;

        /** Number of input features per node. */
        public Builder nIn(long nIn) {
            this.nIn = nIn;
            return this;
        }

        /** Number of output features per node (hidden dimension H). */
        public Builder nOut(long nOut) {
            this.nOut = nOut;
            return this;
        }

        /** Whether to apply ReLU to the GCN output (default: true). */
        public Builder applyRelu(boolean applyRelu) {
            this.applyRelu = applyRelu;
            return this;
        }

        /** Whether to include a bias term (default: true). */
        public Builder hasBias(boolean hasBias) {
            this.hasBias = hasBias;
            return this;
        }

        @Override
        public GcnLayer build() {
            return new GcnLayer(this);
        }
    }
}
