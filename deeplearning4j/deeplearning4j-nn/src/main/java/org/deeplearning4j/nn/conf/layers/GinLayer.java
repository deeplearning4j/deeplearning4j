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
 * Graph Isomorphism Network (GIN) layer (Xu et al. 2019) for use inside a DL4J
 * {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork} or
 * {@link org.deeplearning4j.nn.graph.ComputationGraph}.
 *
 * <p>Delegates to {@code sd.gnn().ginConv()} which computes:
 * <pre>
 *   agg = CsrEdgeAggregate(rowPtr, CsrEdgeGather(colIdx, X), SUM)   [rows, F]
 *   hi  = (1 + eps) * X + agg                                        [rows, F]
 *   h1  = relu( hi · w1 + b1 )                                       [rows, hiddenSize]
 *   out = h1 · w2 + b2                                               [rows, nOut]
 * </pre>
 *
 * <p>The epsilon {@code eps} is a learnable scalar parameter (shape [1]).
 *
 * <p><b>Usage:</b>
 * <ol>
 *   <li>Build a {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration} including this layer.</li>
 *   <li>Call {@link #setAdjacency(SparseNDArray)} before network initialization.
 *       GIN uses the raw adjacency (no normalization).</li>
 *   <li>Pass node features [numNodes, nIn] as the input batch.</li>
 * </ol>
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties({"adjColIdx", "adjRowPtr", "adjRows", "adjNnz"})
public class GinLayer extends SameDiffLayer {

    // -----------------------------------------------------------------------
    // Learnable-parameter keys
    // -----------------------------------------------------------------------
    private static final String W1_KEY   = "w1";
    private static final String B1_KEY   = "b1";
    private static final String W2_KEY   = "w2";
    private static final String B2_KEY   = "b2";
    private static final String EPS_KEY  = "eps";

    // -----------------------------------------------------------------------
    // Configuration fields (serialized)
    // -----------------------------------------------------------------------
    /** Number of input features per node (F). */
    private long nIn;
    /** Number of output features per node. */
    private long nOut;
    /** Hidden dimension of the two-layer MLP inside GIN. */
    private long hiddenSize;
    /** Initial value for the learnable epsilon scalar. */
    private double initEps = 0.0;

    // -----------------------------------------------------------------------
    // Adjacency (transient — NOT serialized, NOT in param table)
    // -----------------------------------------------------------------------
    /** CSR column indices [nnz], INT32 — no values needed for GIN (uniform agg). */
    private transient INDArray adjColIdx;
    /** CSR row pointers [N+1], INT32. */
    private transient INDArray adjRowPtr;
    /** Number of nodes. */
    private transient long adjRows;
    /** Number of edges. */
    private transient long adjNnz;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    protected GinLayer(Builder builder) {
        super(builder);
        this.nIn        = builder.nIn;
        this.nOut       = builder.nOut;
        this.hiddenSize = builder.hiddenSize;
        this.initEps    = builder.initEps;
    }

    // -----------------------------------------------------------------------
    // Adjacency setter
    // -----------------------------------------------------------------------

    /**
     * Set the adjacency matrix for this GIN layer.
     *
     * <p>GIN uses raw CSR structure (uniform edge weights — the aggregation is SUM over
     * neighbour features, unweighted).  No normalization is applied.
     *
     * @param adj CSR adjacency matrix of shape [numNodes, numNodes]
     */
    public void setAdjacency(SparseNDArray adj) {
        this.adjColIdx = adj.getColIdx();
        this.adjRowPtr = adj.getRowPtr();
        this.adjRows   = adj.rows();
        this.adjNnz    = adj.nnz();
    }

    // -----------------------------------------------------------------------
    // SameDiffLayer contract
    // -----------------------------------------------------------------------

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();
        params.addWeightParam(W1_KEY,  nIn,        hiddenSize);
        params.addBiasParam  (B1_KEY,  hiddenSize);
        params.addWeightParam(W2_KEY,  hiddenSize, nOut);
        params.addBiasParam  (B2_KEY,  nOut);
        // eps is a learnable scalar; declare as shape [1] (broadcasts correctly over [rows, F])
        params.addWeightParam(EPS_KEY, 1);
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        for (Map.Entry<String, INDArray> e : params.entrySet()) {
            String key = e.getKey();
            INDArray arr = e.getValue();
            if (B1_KEY.equals(key) || B2_KEY.equals(key)) {
                arr.assign(0.0);
            } else if (EPS_KEY.equals(key)) {
                arr.assign(initEps);
            } else if (W1_KEY.equals(key)) {
                WeightInitUtil.initWeights(nIn, hiddenSize, arr.shape(), weightInit, null, 'c', arr);
            } else if (W2_KEY.equals(key)) {
                WeightInitUtil.initWeights(hiddenSize, nOut, arr.shape(), weightInit, null, 'c', arr);
            }
        }
    }

    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput,
                                  Map<String, SDVariable> paramTable, SDVariable mask) {
        if (adjColIdx == null) {
            throw new IllegalStateException(
                    "GinLayer: adjacency not set. Call setAdjacency(SparseNDArray) before the network is initialized.");
        }
        // Inject adjacency structure as graph constants (INT32, skipped by grad-check)
        SDVariable colIdx = sd.constant("gin_adj_col_idx", adjColIdx);
        SDVariable rowPtr = sd.constant("gin_adj_row_ptr", adjRowPtr);

        SDVariable w1  = paramTable.get(W1_KEY);
        SDVariable b1  = paramTable.get(B1_KEY);
        SDVariable w2  = paramTable.get(W2_KEY);
        SDVariable b2  = paramTable.get(B2_KEY);
        SDVariable eps = paramTable.get(EPS_KEY);

        return sd.gnn().ginConv(layerInput, w1, b1, w2, b2, eps, colIdx, rowPtr, adjRows, adjRows);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return InputType.feedForward(nOut);
    }

    // -----------------------------------------------------------------------
    // Builder
    // -----------------------------------------------------------------------

    public static class Builder extends SameDiffLayer.Builder<Builder> {

        private long   nIn;
        private long   nOut;
        private long   hiddenSize;
        private double initEps = 0.0;

        /** Number of input features per node (F). */
        public Builder nIn(long nIn) {
            this.nIn = nIn;
            return this;
        }

        /** Number of output features per node. */
        public Builder nOut(long nOut) {
            this.nOut = nOut;
            return this;
        }

        /** Hidden dimension of the internal 2-layer MLP (default: nOut is a reasonable choice). */
        public Builder hiddenSize(long hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        /** Initial value for the learnable epsilon scalar (default: 0.0). */
        public Builder initEps(double initEps) {
            this.initEps = initEps;
            return this;
        }

        @Override
        public GinLayer build() {
            return new GinLayer(this);
        }
    }
}
