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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Multi-head Graph Attention Network (GAT) layer (Veličković et al. 2018) for use inside a DL4J
 * {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork} or
 * {@link org.deeplearning4j.nn.graph.ComputationGraph}.
 *
 * <p>Each attention head {@code h} computes (via {@code sd.gnn().gatConvHead()}):
 * <pre>
 *   Wh_h      = X · W_h                              [N, headDim]
 *   srcFeat_h = CsrEdgeGather(colIdx, Wh_h)          [nnz, headDim]
 *   dstFeat_h = gather(Wh_h, rowIdx, 0)              [nnz, headDim]
 *   alpha_h   = CsrRowSoftmax(LeakyReLU(srcFeat_h·aSrc_h + dstFeat_h·aDst_h))   [nnz]
 *   out_h     = CsrEdgeAggregate(rowPtr, srcFeat_h * alpha_h[:,None], SUM)       [N, headDim]
 * </pre>
 *
 * <p>Head outputs are concatenated along axis=1: output shape is {@code [N, numHeads * headDim]}.
 *
 * <p><b>Parameters per head {@code h}</b> (named {@code "W_h"}, {@code "aSrc_h"}, {@code "aDst_h"}):
 * <ul>
 *   <li>{@code W_h}    — linear projection weight [nIn, headDim]</li>
 *   <li>{@code aSrc_h} — attention source vector [headDim, 1]</li>
 *   <li>{@code aDst_h} — attention destination vector [headDim, 1]</li>
 * </ul>
 *
 * <p><b>Usage:</b>
 * <ol>
 *   <li>Build the layer via the {@link Builder}.</li>
 *   <li>Call {@link #setAdjacency(SparseNDArray)} before network initialization.
 *       The method derives the per-edge {@code rowIdx} array from the CSR row pointers.</li>
 *   <li>Pass node features [numNodes, nIn] as the input batch.</li>
 * </ol>
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties({"adjColIdx", "adjRowPtr", "adjRowIdx", "adjRows", "adjNnz"})
public class GatLayer extends SameDiffLayer {

    // -----------------------------------------------------------------------
    // Configuration fields (serialized)
    // -----------------------------------------------------------------------
    private long   nIn;
    /** Feature dimension per attention head (H). Total output = numHeads * headDim. */
    private long   headDim;
    /** Number of parallel attention heads. */
    private int    numHeads   = 1;
    /** Negative slope for the LeakyReLU in the attention score. */
    private double leakyAlpha = 0.2;

    // -----------------------------------------------------------------------
    // Adjacency (transient — NOT serialized, NOT in param table)
    // -----------------------------------------------------------------------
    /** CSR column indices [nnz], INT32. */
    private transient INDArray adjColIdx;
    /** CSR row pointers [N+1], INT32. */
    private transient INDArray adjRowPtr;
    /** Per-edge destination node index [nnz], INT32.  Derived from rowPtr in setAdjacency. */
    private transient INDArray adjRowIdx;
    /** Number of nodes N. */
    private transient long adjRows;
    /** Number of edges nnz. */
    private transient long adjNnz;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    protected GatLayer(Builder builder) {
        super(builder);
        this.nIn        = builder.nIn;
        this.headDim    = builder.headDim;
        this.numHeads   = builder.numHeads;
        this.leakyAlpha = builder.leakyAlpha;
    }

    // -----------------------------------------------------------------------
    // Adjacency setter
    // -----------------------------------------------------------------------

    /**
     * Set the adjacency matrix for this GAT layer.
     *
     * <p>In addition to storing CSR {@code colIdx} and {@code rowPtr}, this method derives
     * the per-edge destination-node index array {@code rowIdx} required by
     * {@code gatConvHead}: for each edge at position {@code k} in CSR row {@code i},
     * {@code rowIdx[k] = i}.
     *
     * @param adj CSR adjacency matrix of shape [numNodes, numNodes]
     */
    public void setAdjacency(SparseNDArray adj) {
        this.adjColIdx = adj.getColIdx();
        this.adjRowPtr = adj.getRowPtr();
        this.adjRows   = adj.rows();
        this.adjNnz    = adj.nnz();

        // Expand rowPtr → rowIdx: for each row i, edges rowPtr[i]..rowPtr[i+1]-1 get rowIdx = i
        int N         = (int) adj.rows();
        int nnz       = (int) adj.nnz();
        int[] rpData  = adj.getRowPtr().toIntVector();
        int[] riData  = new int[nnz];
        for (int i = 0; i < N; i++) {
            for (int k = rpData[i]; k < rpData[i + 1]; k++) {
                riData[k] = i;
            }
        }
        this.adjRowIdx = Nd4j.createFromArray(riData);
    }

    // -----------------------------------------------------------------------
    // SameDiffLayer contract
    // -----------------------------------------------------------------------

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();
        for (int h = 0; h < numHeads; h++) {
            // Linear projection for head h: [nIn, headDim]
            params.addWeightParam(wKey(h),    nIn, headDim);
            // Attention vectors for head h: [headDim, 1]
            params.addWeightParam(aSrcKey(h), headDim, 1);
            params.addWeightParam(aDstKey(h), headDim, 1);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        for (Map.Entry<String, INDArray> e : params.entrySet()) {
            String key = e.getKey();
            INDArray arr = e.getValue();
            // All GAT params are weight-like; attention vectors init'd with Xavier too
            for (int h = 0; h < numHeads; h++) {
                if (wKey(h).equals(key)) {
                    WeightInitUtil.initWeights(nIn, headDim, arr.shape(), weightInit, null, 'c', arr);
                } else if (aSrcKey(h).equals(key) || aDstKey(h).equals(key)) {
                    WeightInitUtil.initWeights(headDim, 1, arr.shape(), weightInit, null, 'c', arr);
                }
            }
        }
    }

    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput,
                                  Map<String, SDVariable> paramTable, SDVariable mask) {
        if (adjColIdx == null) {
            throw new IllegalStateException(
                    "GatLayer: adjacency not set. Call setAdjacency(SparseNDArray) before the network is initialized.");
        }
        // Inject structural constants (INT32 — not differentiated)
        SDVariable colIdx = sd.constant("gat_adj_col_idx", adjColIdx);
        SDVariable rowPtr = sd.constant("gat_adj_row_ptr", adjRowPtr);
        SDVariable rowIdx = sd.constant("gat_adj_row_idx", adjRowIdx);

        // Compute each head and collect outputs for concatenation
        List<SDVariable> headOutputs = new ArrayList<>(numHeads);
        for (int h = 0; h < numHeads; h++) {
            SDVariable W    = paramTable.get(wKey(h));
            SDVariable aSrc = paramTable.get(aSrcKey(h));
            SDVariable aDst = paramTable.get(aDstKey(h));
            SDVariable headOut = sd.gnn().gatConvHead(
                    layerInput, W, aSrc, aDst, colIdx, rowPtr, rowIdx, adjNnz, adjRows, leakyAlpha);
            headOutputs.add(headOut);
        }

        // Concatenate head outputs along the feature axis: [N, numHeads * headDim]
        if (numHeads == 1) {
            return headOutputs.get(0);
        }
        return sd.concat(1, headOutputs.toArray(new SDVariable[0]));
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return InputType.feedForward((long) numHeads * headDim);
    }

    // -----------------------------------------------------------------------
    // Private helpers: stable, unique parameter key names
    // -----------------------------------------------------------------------

    private static String wKey(int head)    { return "W_"    + head; }
    private static String aSrcKey(int head) { return "aSrc_" + head; }
    private static String aDstKey(int head) { return "aDst_" + head; }

    // -----------------------------------------------------------------------
    // Builder
    // -----------------------------------------------------------------------

    public static class Builder extends SameDiffLayer.Builder<Builder> {

        private long   nIn;
        private long   headDim;
        private int    numHeads   = 1;
        private double leakyAlpha = 0.2;

        /** Number of input features per node (F). */
        public Builder nIn(long nIn) {
            this.nIn = nIn;
            return this;
        }

        /** Feature dimension per attention head.  Total output = {@code numHeads * headDim}. */
        public Builder headDim(long headDim) {
            this.headDim = headDim;
            return this;
        }

        /** Number of parallel attention heads (default: 1). */
        public Builder numHeads(int numHeads) {
            this.numHeads = numHeads;
            return this;
        }

        /** Negative slope for the LeakyReLU in the attention score (default: 0.2). */
        public Builder leakyAlpha(double leakyAlpha) {
            this.leakyAlpha = leakyAlpha;
            return this;
        }

        @Override
        public GatLayer build() {
            return new GatLayer(this);
        }
    }
}
