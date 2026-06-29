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
 * GraphSAGE layer (Hamilton et al. 2017) for use inside a DL4J
 * {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork} or
 * {@link org.deeplearning4j.nn.graph.ComputationGraph}.
 *
 * <p>Supports three aggregation modes ({@link Aggregation}):
 * <ul>
 *   <li>{@link Aggregation#MEAN} — mean-aggregated neighbourhood + self; delegates to
 *       {@code sd.gnn().sageMean(...)}</li>
 *   <li>{@link Aggregation#MAX}  — max-aggregated neighbourhood + self; delegates to
 *       {@code sd.gnn().sageMax(...)}</li>
 *   <li>{@link Aggregation#POOL} — per-neighbour MLP + max-aggregate + output transform;
 *       delegates to {@code sd.gnn().sagePool(...)}.  Requires {@code poolSize} to be set.</li>
 * </ul>
 *
 * <p>All variants concatenate the aggregated neighbourhood embedding with the self-embedding
 * before applying the output linear transform, so the weight matrix {@code W} has shape
 * {@code [2*nIn, nOut]} for MEAN/MAX and the output weight {@code wOut} has shape
 * {@code [nIn + poolSize, nOut]} for POOL.
 *
 * <p><b>Usage:</b>
 * <ol>
 *   <li>Build the layer via the {@link Builder}, choosing the aggregation mode.</li>
 *   <li>Call {@link #setAdjacency(SparseNDArray)} before network initialization.</li>
 *   <li>Pass node features [numNodes, nIn] as the input batch.</li>
 * </ol>
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties({"adjColIdx", "adjRowPtr", "adjRows"})
public class GraphSageLayer extends SameDiffLayer {

    // -----------------------------------------------------------------------
    // Aggregation mode enum
    // -----------------------------------------------------------------------

    /**
     * Neighbourhood aggregation strategy for GraphSAGE.
     */
    public enum Aggregation {
        /** Mean aggregation: {@code CsrEdgeAggregate(MEAN)}. */
        MEAN,
        /** Max aggregation: {@code CsrSegmentMax}. */
        MAX,
        /**
         * Pool aggregation: per-neighbour MLP ({@code wPool}/{@code bPool}) followed by
         * max-aggregation.  Requires {@link Builder#poolSize(long)} to be configured.
         */
        POOL
    }

    // -----------------------------------------------------------------------
    // Param keys (MEAN / MAX)
    // -----------------------------------------------------------------------
    private static final String W_KEY    = "W";
    private static final String BIAS_KEY = "b";

    // -----------------------------------------------------------------------
    // Param keys (POOL only)
    // -----------------------------------------------------------------------
    private static final String W_POOL_KEY    = "wPool";
    private static final String B_POOL_KEY    = "bPool";
    private static final String W_OUT_KEY     = "wOut";
    private static final String B_OUT_KEY     = "bOut";

    // -----------------------------------------------------------------------
    // Configuration fields (serialized)
    // -----------------------------------------------------------------------
    private long        nIn;
    private long        nOut;
    private Aggregation aggregation = Aggregation.MEAN;
    /** MLP hidden size used only for {@link Aggregation#POOL}. */
    private long        poolSize    = 0;
    private boolean     hasBias     = true;

    // -----------------------------------------------------------------------
    // Adjacency (transient — NOT serialized, NOT in param table)
    // -----------------------------------------------------------------------
    private transient INDArray adjColIdx;
    private transient INDArray adjRowPtr;
    private transient long     adjRows;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    protected GraphSageLayer(Builder builder) {
        super(builder);
        this.nIn         = builder.nIn;
        this.nOut        = builder.nOut;
        this.aggregation = builder.aggregation;
        this.poolSize    = builder.poolSize;
        this.hasBias     = builder.hasBias;
    }

    // -----------------------------------------------------------------------
    // Adjacency setter
    // -----------------------------------------------------------------------

    /**
     * Set the adjacency matrix for this GraphSAGE layer.
     *
     * <p>GraphSAGE uses the raw CSR structure (no normalization).
     *
     * @param adj CSR adjacency matrix of shape [numNodes, numNodes]
     */
    public void setAdjacency(SparseNDArray adj) {
        this.adjColIdx = adj.getColIdx();
        this.adjRowPtr = adj.getRowPtr();
        this.adjRows   = adj.rows();
    }

    // -----------------------------------------------------------------------
    // SameDiffLayer contract
    // -----------------------------------------------------------------------

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();
        switch (aggregation) {
            case MEAN:
            case MAX:
                // concat([X, agg], axis=1) → [rows, 2*nIn]; then W [2*nIn, nOut]
                params.addWeightParam(W_KEY, 2 * nIn, nOut);
                if (hasBias) params.addBiasParam(BIAS_KEY, nOut);
                break;
            case POOL:
                if (poolSize <= 0) throw new IllegalStateException(
                        "GraphSageLayer POOL mode requires poolSize > 0. Use Builder.poolSize(...).");
                // Neighbour MLP: [nIn, poolSize]
                params.addWeightParam(W_POOL_KEY, nIn,           poolSize);
                params.addBiasParam  (B_POOL_KEY, poolSize);
                // Output transform: concat([X, agg]) has width nIn + poolSize
                params.addWeightParam(W_OUT_KEY,  nIn + poolSize, nOut);
                if (hasBias) params.addBiasParam(B_OUT_KEY, nOut);
                break;
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        for (Map.Entry<String, INDArray> e : params.entrySet()) {
            String key = e.getKey();
            INDArray arr = e.getValue();
            if (BIAS_KEY.equals(key) || B_POOL_KEY.equals(key) || B_OUT_KEY.equals(key)) {
                arr.assign(0.0);
            } else if (W_KEY.equals(key)) {
                WeightInitUtil.initWeights(2 * nIn, nOut, arr.shape(), weightInit, null, 'c', arr);
            } else if (W_POOL_KEY.equals(key)) {
                WeightInitUtil.initWeights(nIn, poolSize, arr.shape(), weightInit, null, 'c', arr);
            } else if (W_OUT_KEY.equals(key)) {
                WeightInitUtil.initWeights(nIn + poolSize, nOut, arr.shape(), weightInit, null, 'c', arr);
            }
        }
    }

    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput,
                                  Map<String, SDVariable> paramTable, SDVariable mask) {
        if (adjColIdx == null) {
            throw new IllegalStateException(
                    "GraphSageLayer: adjacency not set. Call setAdjacency(SparseNDArray) before the network is initialized.");
        }
        SDVariable colIdx = sd.constant("sage_adj_col_idx", adjColIdx);
        SDVariable rowPtr = sd.constant("sage_adj_row_ptr", adjRowPtr);

        switch (aggregation) {
            case MEAN: {
                SDVariable W    = paramTable.get(W_KEY);
                SDVariable bias = hasBias ? paramTable.get(BIAS_KEY) : null;
                return sd.gnn().sageMean(layerInput, W, bias, colIdx, rowPtr, adjRows, adjRows);
            }
            case MAX: {
                SDVariable W    = paramTable.get(W_KEY);
                SDVariable bias = hasBias ? paramTable.get(BIAS_KEY) : null;
                return sd.gnn().sageMax(layerInput, W, bias, colIdx, rowPtr, adjRows);
            }
            case POOL: {
                SDVariable wPool = paramTable.get(W_POOL_KEY);
                SDVariable bPool = paramTable.get(B_POOL_KEY);
                SDVariable wOut  = paramTable.get(W_OUT_KEY);
                SDVariable bOut  = hasBias ? paramTable.get(B_OUT_KEY) : null;
                return sd.gnn().sagePool(layerInput, wPool, bPool, wOut, bOut, colIdx, rowPtr, adjRows, adjRows);
            }
            default:
                throw new IllegalStateException("Unknown aggregation mode: " + aggregation);
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return InputType.feedForward(nOut);
    }

    // -----------------------------------------------------------------------
    // Builder
    // -----------------------------------------------------------------------

    public static class Builder extends SameDiffLayer.Builder<Builder> {

        private long        nIn;
        private long        nOut;
        private Aggregation aggregation = Aggregation.MEAN;
        private long        poolSize    = 0;
        private boolean     hasBias     = true;

        /** Number of input features per node (F). */
        public Builder nIn(long nIn) {
            this.nIn = nIn;
            return this;
        }

        /** Number of output features per node (H). */
        public Builder nOut(long nOut) {
            this.nOut = nOut;
            return this;
        }

        /** Aggregation mode (default: {@link Aggregation#MEAN}). */
        public Builder aggregation(Aggregation aggregation) {
            this.aggregation = aggregation;
            return this;
        }

        /**
         * Hidden dimension of the per-neighbour MLP used in {@link Aggregation#POOL} mode.
         * Must be set when using POOL aggregation.
         */
        public Builder poolSize(long poolSize) {
            this.poolSize = poolSize;
            return this;
        }

        /** Whether to include a bias term (default: true). */
        public Builder hasBias(boolean hasBias) {
            this.hasBias = hasBias;
            return this;
        }

        @Override
        public GraphSageLayer build() {
            return new GraphSageLayer(this);
        }
    }
}
