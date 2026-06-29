/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*

/**
 * GNN message-passing namespace. All methods are pure SameDiff compositions — no new native ops.
 * Gradients flow automatically. Generated via op-codegen from this file; do not hand-edit SDGNN/NDGNN.
 */
fun GNN() = Namespace("GNN") {

    // -------------------------------------------------------------------------
    // GCN
    // -------------------------------------------------------------------------

    Op("gcnConv") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        Input(FLOATING_POINT, "W") { description = "Weight matrix [F, H]" }
        Input(FLOATING_POINT, "bias") { description = "Optional bias [H]; pass null to omit" }
        Input(FLOATING_POINT, "aNormVals") { description = "CSR non-zero values of the normalised adjacency [nnz]" }
        Input(INT, "aNormColIdx") { description = "CSR column indices [nnz]" }
        Input(INT, "aNormRowPtr") { description = "CSR row pointers [rows+1]" }
        Arg(LONG, "rows") { description = "Number of nodes / rows in A_norm" }
        Arg(LONG, "cols") { description = "Number of columns in A_norm" }
        Arg(BOOL, "applyRelu") { description = "Whether to apply ReLU to the output" }
        Output(FLOATING_POINT, "out") { description = "Output node embeddings [rows, H]" }
        Composition("""
            SDVariable AX = sd.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, X, (int) rows, (int) cols, false);
            SDVariable AXW = sd.mmul(AX, W);
            if (bias != null) {
                AXW = AXW.add(bias);
            }
            SDVariable out = applyRelu ? sd.nn().relu(AXW, 0.0) : AXW;
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Graph Convolutional Network layer (Kipf & Welling 2017).
            out = relu?( A_norm · X · W + bias )
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // GAT
    // -------------------------------------------------------------------------

    Op("gatConvHead") {
        Input(FLOATING_POINT, "X") { description = "Node features [N, F]" }
        Input(FLOATING_POINT, "W") { description = "Linear weight [F, H]" }
        Input(FLOATING_POINT, "attSrc") { description = "Attention source vector [H, 1]" }
        Input(FLOATING_POINT, "attDst") { description = "Attention destination vector [H, 1]" }
        Input(INT, "colIdx") { description = "CSR column indices (source nodes per edge) [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [N+1] INT32" }
        Input(INT, "rowIdx") { description = "Per-edge destination-node index [nnz] INT32" }
        Arg(LONG, "nnz") { description = "Number of edges" }
        Arg(LONG, "N") { description = "Number of nodes" }
        Arg(FLOATING_POINT, "leakyAlpha") { description = "Negative slope for LeakyReLU" }
        Output(FLOATING_POINT, "out") { description = "Aggregated node embeddings [N, H]" }
        Composition("""
            SDVariable Wh = sd.mmul(X, W);
            SDVariable srcFeat = sd.sparse().csrEdgeGather(colIdx, Wh, (int) N);
            SDVariable dstFeat = sd.gather(Wh, rowIdx, 0);
            SDVariable eSrc = sd.reshape(sd.mmul(srcFeat, attSrc), nnz);
            SDVariable eDst = sd.reshape(sd.mmul(dstFeat, attDst), nnz);
            SDVariable eLogit = sd.nn().leakyRelu(eSrc.add(eDst), leakyAlpha);
            SDVariable alpha = sd.sparse().csrRowSoftmax(eLogit, rowPtr, (int) N);
            SDVariable weighted = srcFeat.mul(sd.reshape(alpha, nnz, 1));
            SDVariable out = sd.sparse().csrEdgeAggregate(rowPtr, weighted, (int) N, 0);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Single-head Graph Attention Network convolution (Veličković et al. 2018).
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // GraphSAGE
    // -------------------------------------------------------------------------

    Op("sageMean") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        Input(FLOATING_POINT, "W") { description = "Weight matrix [2F, H]" }
        Input(FLOATING_POINT, "bias") { description = "Optional bias [H]; pass null to omit" }
        Input(INT, "colIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "n") { description = "Number of nodes (= X.shape[0], for CsrEdgeGather doDiff)" }
        Output(FLOATING_POINT, "out") { description = "Node embeddings [rows, H]" }
        Composition("""
            SDVariable Xj = sd.sparse().csrEdgeGather(colIdx, X, (int) n);
            SDVariable agg = sd.sparse().csrEdgeAggregate(rowPtr, Xj, (int) rows, 1);
            SDVariable cat = sd.concat(1, X, agg);
            SDVariable h = sd.mmul(cat, W);
            if (bias != null) {
                h = h.add(bias);
            }
            SDVariable out = sd.nn().relu(h, 0.0);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            GraphSAGE mean aggregation (Hamilton et al. 2017).
            """.trimIndent()
        }
    }

    Op("sageMax") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        Input(FLOATING_POINT, "W") { description = "Weight matrix [2F, H]" }
        Input(FLOATING_POINT, "bias") { description = "Optional bias [H]; pass null to omit" }
        Input(INT, "colIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Output(FLOATING_POINT, "out") { description = "Node embeddings [rows, H]" }
        Composition("""
            SDVariable agg = sd.sparse().csrSegmentMax(colIdx, rowPtr, X, (int) rows);
            SDVariable cat = sd.concat(1, X, agg);
            SDVariable h = sd.mmul(cat, W);
            if (bias != null) {
                h = h.add(bias);
            }
            SDVariable out = sd.nn().relu(h, 0.0);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            GraphSAGE max aggregation.
            """.trimIndent()
        }
    }

    Op("sagePool") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        Input(FLOATING_POINT, "wPool") { description = "MLP weight for neighbours [F, H_pool]" }
        Input(FLOATING_POINT, "bPool") { description = "MLP bias [H_pool]" }
        Input(FLOATING_POINT, "wOut") { description = "Output weight [F + H_pool, H_out]" }
        Input(FLOATING_POINT, "bOut") { description = "Output bias [H_out]; pass null to omit" }
        Input(INT, "colIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "n") { description = "Number of nodes (for CsrEdgeGather doDiff)" }
        Output(FLOATING_POINT, "out") { description = "Node embeddings [rows, H_out]" }
        Composition("""
            SDVariable Xj = sd.sparse().csrEdgeGather(colIdx, X, (int) n);
            SDVariable hj = sd.nn().relu(sd.mmul(Xj, wPool).add(bPool), 0.0);
            SDVariable agg = sd.sparse().csrEdgeAggregate(rowPtr, hj, (int) rows, 2);
            SDVariable cat = sd.concat(1, X, agg);
            SDVariable out = sd.mmul(cat, wOut);
            if (bOut != null) {
                out = out.add(bOut);
            }
            out = sd.nn().relu(out, 0.0);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            GraphSAGE pool aggregation: apply an MLP to each neighbour, max-aggregate, then predict.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // GIN
    // -------------------------------------------------------------------------

    Op("ginConv") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        Input(FLOATING_POINT, "w1") { description = "First MLP weight [F, H]" }
        Input(FLOATING_POINT, "b1") { description = "First MLP bias [H]" }
        Input(FLOATING_POINT, "w2") { description = "Second MLP weight [H, out]" }
        Input(FLOATING_POINT, "b2") { description = "Second MLP bias [out]" }
        Input(FLOATING_POINT, "eps") { description = "Learnable epsilon scalar SDVariable (shape [])" }
        Input(INT, "colIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "n") { description = "Number of nodes (for CsrEdgeGather doDiff)" }
        Arg(BOOL, "layerNorm") { description = "Whether to apply parameter-free layer normalisation"; defaultValue = false }
        Output(FLOATING_POINT, "out") { description = "Node embeddings [rows, out], optionally layer-normalised" }
        Composition("""
            SDVariable Xj = sd.sparse().csrEdgeGather(colIdx, X, (int) n);
            SDVariable agg = sd.sparse().csrEdgeAggregate(rowPtr, Xj, (int) rows, 0);
            SDVariable hi = X.mul(eps.add(1.0)).add(agg);
            SDVariable h1 = sd.nn().relu(sd.mmul(hi, w1).add(b1), 0.0);
            SDVariable out = sd.mmul(h1, w2).add(b2);
            if (layerNorm) {
                SDVariable lnMean = sd.mean(out, true, 1);
                SDVariable d = out.sub(lnMean);
                SDVariable var = sd.mean(d.mul(d), true, 1);
                out = d.div(sd.math().sqrt(var.add(1e-5)));
            }
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Graph Isomorphism Network convolution (Xu et al. 2019) with optional Layer Normalisation.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // GATv2
    // -------------------------------------------------------------------------

    Op("gatV2ConvHead") {
        Input(FLOATING_POINT, "X") { description = "Node features [N, F]" }
        Input(FLOATING_POINT, "W") { description = "Linear weight [F, H]" }
        Input(FLOATING_POINT, "att") { description = "Shared attention vector [H, 1]" }
        Input(INT, "colIdx") { description = "CSR column indices (source nodes per edge) [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [N+1] INT32" }
        Input(INT, "rowIdx") { description = "Per-edge destination-node index [nnz] INT32" }
        Arg(LONG, "nnz") { description = "Number of edges" }
        Arg(LONG, "N") { description = "Number of nodes" }
        Arg(FLOATING_POINT, "leakyAlpha") { description = "Negative slope for LeakyReLU" }
        Output(FLOATING_POINT, "out") { description = "Aggregated node embeddings [N, H]" }
        Composition("""
            SDVariable Wh = sd.mmul(X, W);
            SDVariable srcFeat = sd.sparse().csrEdgeGather(colIdx, Wh, (int) N);
            SDVariable dstFeat = sd.gather(Wh, rowIdx, 0);
            SDVariable g = sd.nn().leakyRelu(srcFeat.add(dstFeat), leakyAlpha);
            SDVariable eLogit = sd.reshape(sd.mmul(g, att), nnz);
            SDVariable alpha = sd.sparse().csrRowSoftmax(eLogit, rowPtr, (int) N);
            SDVariable weighted = srcFeat.mul(sd.reshape(alpha, nnz, 1));
            SDVariable out = sd.sparse().csrEdgeAggregate(rowPtr, weighted, (int) N, 0);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Single-head GATv2 convolution (Brody et al. 2021). GATv2 applies the nonlinearity before
            the attention projection (dynamic attention), fixing the static attention limitation of GAT v1.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // APPNP
    // -------------------------------------------------------------------------

    Op("appnp") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        Input(FLOATING_POINT, "W") { description = "Prediction weight [F, H]" }
        Input(FLOATING_POINT, "bias") { description = "Optional bias [H]; pass null to omit" }
        Input(FLOATING_POINT, "aNormVals") { description = "CSR values of the normalised adjacency [nnz]" }
        Input(INT, "aNormColIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "aNormRowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "cols") { description = "Number of columns of A_norm" }
        Arg(INT, "k") { description = "Number of propagation steps" }
        Arg(FLOATING_POINT, "alpha") { description = "Teleport (restart) probability in [0,1)" }
        Output(FLOATING_POINT, "out") { description = "Propagated node predictions [rows, H]" }
        Composition("""
            SDVariable H0 = sd.mmul(X, W);
            if (bias != null) {
                H0 = H0.add(bias);
            }
            SDVariable H = H0;
            for (int i = 0; i < k; i++) {
                SDVariable AH = sd.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, H, (int) rows, (int) cols, false);
                H = AH.mul(1.0 - alpha).add(H0.mul(alpha));
            }
            SDVariable out = H;
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Approximate Personalized Propagation of Neural Predictions (Klicpera et al. 2019).
            Decouples prediction from propagation: k steps of personalized PageRank with teleport alpha.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // GCNII
    // -------------------------------------------------------------------------

    Op("gcniiConv") {
        Input(FLOATING_POINT, "H") { description = "Current layer representation [rows, F]" }
        Input(FLOATING_POINT, "H0") { description = "Initial (input-projected) representation [rows, F]" }
        Input(FLOATING_POINT, "W") { description = "Weight matrix [F, F]" }
        Input(FLOATING_POINT, "aNormVals") { description = "CSR values of the normalised adjacency [nnz]" }
        Input(INT, "aNormColIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "aNormRowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "cols") { description = "Number of columns of A_norm" }
        Arg(FLOATING_POINT, "alpha") { description = "Initial-residual strength in [0,1]" }
        Arg(FLOATING_POINT, "beta") { description = "Identity-mapping strength in [0,1]" }
        Arg(BOOL, "applyRelu") { description = "Whether to apply ReLU to the output" }
        Output(FLOATING_POINT, "out") { description = "Updated node representation [rows, F]" }
        Composition("""
            SDVariable AH = sd.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, H, (int) rows, (int) cols, false);
            SDVariable M = AH.mul(1.0 - alpha).add(H0.mul(alpha));
            SDVariable MW = sd.mmul(M, W);
            SDVariable out = M.mul(1.0 - beta).add(MW.mul(beta));
            if (applyRelu) {
                out = sd.nn().relu(out, 0.0);
            }
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            GCNII convolution (Chen et al. 2020): deep GCN layer combining initial residual connection
            with identity mapping. M = (1-alpha)*(A_norm*H) + alpha*H0; out = sigma((1-beta)*M + beta*(M*W))
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // Jumping Knowledge
    // -------------------------------------------------------------------------

    Op("jkNetConcat") {
        val layerOutputs = Input(FLOATING_POINT, "layerOutputs") { count = AtLeast(1); description = "Per-layer node representations, each [rows, H_l]" }
        Output(FLOATING_POINT, "out") { description = "Concatenated representation [rows, sum(H_l)]" }
        Composition("""
            SDVariable out = sd.concat(1, layerOutputs);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Jumping-Knowledge concatenation aggregator (Xu et al. 2018).
            Concatenates per-layer representations along the feature dimension.
            """.trimIndent()
        }
    }

    Op("jkNetMax") {
        val layerOutputs = Input(FLOATING_POINT, "layerOutputs") { count = AtLeast(1); description = "Per-layer node representations, each [rows, H]" }
        Output(FLOATING_POINT, "out") { description = "Element-wise max across layers [rows, H]" }
        Composition("""
            SDVariable out = layerOutputs[0];
            for (int i = 1; i < layerOutputs.length; i++) {
                out = sd.math().max(out, layerOutputs[i]);
            }
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Jumping-Knowledge max-pooling aggregator (Xu et al. 2018).
            Element-wise maximum across per-layer node representations.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // PairNorm
    // -------------------------------------------------------------------------

    Op("pairNorm") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        Arg(FLOATING_POINT, "scale") { description = "Target row-norm scale s (typically ~1.0)" }
        Output(FLOATING_POINT, "out") { description = "Normalised node features [rows, F]" }
        Composition("""
            SDVariable mean = sd.mean(X, true, 0);
            SDVariable Xc = X.sub(mean);
            SDVariable rowSq = sd.sum(Xc.mul(Xc), true, 1);
            SDVariable denom = sd.math().sqrt(sd.mean(rowSq, false).add(1e-6));
            SDVariable out = Xc.mul(scale).div(denom);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            PairNorm (Zhao & Akoglu 2020): parameter-free normalisation that keeps total pairwise feature distance constant.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // GraphNorm
    // -------------------------------------------------------------------------

    Op("graphNorm") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        Input(FLOATING_POINT, "gamma") { description = "Learnable scale [F]" }
        Input(FLOATING_POINT, "beta") { description = "Learnable shift [F]" }
        Input(FLOATING_POINT, "alpha") { description = "Learnable mean-retention coefficient [F]" }
        Output(FLOATING_POINT, "out") { description = "Normalised node features [rows, F]" }
        Composition("""
            SDVariable Xt = sd.transpose(X);
            SDVariable meanT = sd.mean(Xt, true, 1);
            SDVariable alphaCol = sd.reshape(alpha, -1, 1);
            SDVariable shiftedT = Xt.sub(meanT.mul(alphaCol));
            SDVariable varT = sd.mean(shiftedT.mul(shiftedT), true, 1);
            SDVariable normT = shiftedT.div(sd.math().sqrt(varT.add(1e-5)));
            SDVariable gammaCol = sd.reshape(gamma, -1, 1);
            SDVariable betaCol = sd.reshape(beta, -1, 1);
            SDVariable outT = normT.mul(gammaCol).add(betaCol);
            SDVariable out = sd.transpose(outT);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            GraphNorm (Cai et al. 2021): learnable graph-level normalisation. Implemented in transposed
            [F, rows] layout for correct CUDA gradient flow on per-feature alpha scaling.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // ChebConv
    // -------------------------------------------------------------------------

    Op("chebConv") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        val weights = Input(FLOATING_POINT, "weights") { count = AtLeast(1); description = "K Chebyshev-coefficient matrices, each [F, H]" }
        Input(FLOATING_POINT, "lapVals") { description = "CSR values of the scaled Laplacian L_hat [nnz]" }
        Input(INT, "colIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "cols") { description = "Number of columns of L_hat" }
        Output(FLOATING_POINT, "out") { description = "Filtered node embeddings [rows, H]" }
        Composition("""
            SDVariable tPrev2 = X;
            SDVariable out = sd.mmul(X, weights[0]);
            if (weights.length >= 2) {
                SDVariable tPrev1 = sd.sparse().csrSpmm(lapVals, colIdx, rowPtr, X, (int) rows, (int) cols, false);
                out = out.add(sd.mmul(tPrev1, weights[1]));
                for (int k = 2; k < weights.length; k++) {
                    SDVariable lTk = sd.sparse().csrSpmm(lapVals, colIdx, rowPtr, tPrev1, (int) rows, (int) cols, false);
                    SDVariable tk = lTk.mul(2.0).sub(tPrev2);
                    out = out.add(sd.mmul(tk, weights[k]));
                    tPrev2 = tPrev1;
                    tPrev1 = tk;
                }
            }
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Chebyshev spectral graph convolution (Defferrard et al. 2016).
            T_0=X, T_1=L_hat*X, T_k=2*L_hat*T_{k-1}-T_{k-2}; out=sum_k T_k*W_k
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // RGCN
    // -------------------------------------------------------------------------

    Op("rgcnConv") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        val relVals = Input(FLOATING_POINT, "relVals") { count = AtLeast(1); description = "Per-relation CSR values, relVals[r]=[nnz_r]" }
        val relColIdx = Input(INT, "relColIdx") { count = AtLeast(1); description = "Per-relation CSR column indices [nnz_r] INT32" }
        val relRowPtr = Input(INT, "relRowPtr") { count = AtLeast(1); description = "Per-relation CSR row pointers [rows+1] INT32" }
        val relW = Input(FLOATING_POINT, "relW") { count = AtLeast(1); description = "Per-relation weights relW[r]=[F,H]" }
        Input(FLOATING_POINT, "selfW") { description = "Self-loop weight [F, H]" }
        Input(FLOATING_POINT, "bias") { description = "Optional bias [H]; pass null to omit" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "cols") { description = "Number of columns of each A_r" }
        Arg(BOOL, "applyRelu") { description = "Whether to apply ReLU to the output" }
        Output(FLOATING_POINT, "out") { description = "Node embeddings [rows, H]" }
        Composition("""
            SDVariable out = sd.mmul(X, selfW);
            for (int r = 0; r < relW.length; r++) {
                SDVariable aX = sd.sparse().csrSpmm(relVals[r], relColIdx[r], relRowPtr[r], X, (int) rows, (int) cols, false);
                out = out.add(sd.mmul(aX, relW[r]));
            }
            if (bias != null) {
                out = out.add(bias);
            }
            if (applyRelu) {
                out = sd.nn().relu(out, 0.0);
            }
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Relational Graph Convolutional Network layer (Schlichtkrull et al. 2018).
            out = X*W_self + sum_r(A_r*X*W_r) + bias
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // PNA
    // -------------------------------------------------------------------------

    Op("pnaConv") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, F]" }
        Input(FLOATING_POINT, "W") { description = "Weight matrix [4F, H]" }
        Input(FLOATING_POINT, "bias") { description = "Optional bias [H]; pass null to omit" }
        Input(INT, "colIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "n") { description = "Number of nodes (for CsrEdgeGather doDiff)" }
        Arg(BOOL, "applyRelu") { description = "Whether to apply ReLU to the output" }
        Output(FLOATING_POINT, "out") { description = "Node embeddings [rows, H]" }
        Composition("""
            SDVariable Xj = sd.sparse().csrEdgeGather(colIdx, X, (int) n);
            SDVariable mean = sd.sparse().csrEdgeAggregate(rowPtr, Xj, (int) rows, 1);
            SDVariable max = sd.sparse().csrEdgeAggregate(rowPtr, Xj, (int) rows, 2);
            SDVariable min = sd.sparse().csrEdgeAggregate(rowPtr, Xj.mul(-1.0), (int) rows, 2).mul(-1.0);
            SDVariable meanSq = sd.sparse().csrEdgeAggregate(rowPtr, Xj.mul(Xj), (int) rows, 1);
            SDVariable std = sd.math().sqrt(meanSq.sub(mean.mul(mean)).add(1e-6));
            SDVariable agg = sd.concat(1, mean, max, min, std);
            SDVariable out = sd.mmul(agg, W);
            if (bias != null) {
                out = out.add(bias);
            }
            if (applyRelu) {
                out = sd.nn().relu(out, 0.0);
            }
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Principal Neighbourhood Aggregation convolution (Corso et al. 2020).
            Combines mean, max, min and std aggregators, concatenated and linearly projected.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // VGAE
    // -------------------------------------------------------------------------

    Op("vgaeReparam") {
        Input(FLOATING_POINT, "mu") { description = "Latent mean [N, d]" }
        Input(FLOATING_POINT, "logvar") { description = "Latent log-variance [N, d]" }
        Input(FLOATING_POINT, "noise") { description = "Standard-normal sample [N, d]" }
        Output(FLOATING_POINT, "out") { description = "Latent sample z [N, d]" }
        Composition("""
            SDVariable out = mu.add(sd.math().exp(logvar.mul(0.5)).mul(noise));
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            VGAE reparameterisation trick (Kipf & Welling 2016): z = mu + exp(0.5*logvar) * noise
            """.trimIndent()
        }
    }

    Op("vgaeKlLoss") {
        Input(FLOATING_POINT, "mu") { description = "Latent mean [N, d]" }
        Input(FLOATING_POINT, "logvar") { description = "Latent log-variance [N, d]" }
        Output(FLOATING_POINT, "out") { description = "Scalar KL divergence" }
        Composition("""
            SDVariable kl = sd.math().exp(logvar).add(mu.mul(mu)).sub(logvar).sub(1.0);
            SDVariable out = sd.mean(kl, false).mul(0.5);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            VGAE KL-divergence regulariser (Kipf & Welling 2016): 0.5 * mean(exp(logvar) + mu^2 - logvar - 1)
            """.trimIndent()
        }
    }

    Op("innerProductDecoder") {
        Input(FLOATING_POINT, "z") { description = "Node latents [N, d]" }
        Output(FLOATING_POINT, "out") { description = "Edge logits [N, N]" }
        Composition("""
            SDVariable out = sd.mmul(z, z, false, true, false);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Inner-product link decoder (Kipf & Welling 2016): Z*Z^T. Apply sigmoid downstream for edge probabilities.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // GGNN
    // -------------------------------------------------------------------------

    Op("ggnn") {
        Input(FLOATING_POINT, "X") { description = "Initial node states [rows, H]" }
        Input(FLOATING_POINT, "aggW") { description = "Message transform [H, H]" }
        Input(FLOATING_POINT, "wz") { description = "Update-gate weight [H, H]" }
        Input(FLOATING_POINT, "uz") { description = "Update-gate recurrent weight [H, H]" }
        Input(FLOATING_POINT, "wr") { description = "Reset-gate weight [H, H]" }
        Input(FLOATING_POINT, "ur") { description = "Reset-gate recurrent weight [H, H]" }
        Input(FLOATING_POINT, "wh") { description = "Candidate weight [H, H]" }
        Input(FLOATING_POINT, "uh") { description = "Candidate recurrent weight [H, H]" }
        Input(FLOATING_POINT, "aNormVals") { description = "CSR values of the normalised adjacency [nnz]" }
        Input(INT, "aNormColIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "aNormRowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "cols") { description = "Columns of A_norm" }
        Arg(INT, "steps") { description = "Number of propagation / GRU steps" }
        Output(FLOATING_POINT, "out") { description = "Final node states [rows, H]" }
        Composition("""
            SDVariable h = X;
            for (int t = 0; t < steps; t++) {
                SDVariable agg = sd.sparse().csrSpmm(aNormVals, aNormColIdx, aNormRowPtr, h, (int) rows, (int) cols, false);
                SDVariable a = sd.mmul(agg, aggW);
                SDVariable ah = sd.concat(1, a, h);
                SDVariable z = sd.nn().sigmoid(sd.mmul(ah, sd.concat(0, wz, uz)));
                SDVariable r = sd.nn().sigmoid(sd.mmul(ah, sd.concat(0, wr, ur)));
                SDVariable arh = sd.concat(1, a, r.mul(h));
                SDVariable hh = sd.math().tanh(sd.mmul(arh, sd.concat(0, wh, uh)));
                h = h.mul(z.mul(-1.0).add(1.0)).add(z.mul(hh));
            }
            SDVariable out = h;
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Gated Graph Neural Network (Li et al. 2016): steps rounds of neighbour aggregation + GRU update.
            Uses concat([in1,in2])*concat([W1,W2]) gate form for correct CUDA backward.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // Graph Transformer
    // -------------------------------------------------------------------------

    Op("graphTransformer") {
        Input(FLOATING_POINT, "X") { description = "Node features [N, F]" }
        Input(FLOATING_POINT, "wq") { description = "Query weight [F, d]" }
        Input(FLOATING_POINT, "wk") { description = "Key weight [F, d]" }
        Input(FLOATING_POINT, "wv") { description = "Value weight [F, d]" }
        Input(FLOATING_POINT, "wo") { description = "Output weight [d, d_out]" }
        Input(FLOATING_POINT, "adjMask") { description = "Optional additive attention mask [N, N]; pass null for full attention" }
        Arg(FLOATING_POINT, "scale") { description = "Score scaling (typically 1/sqrt(d))" }
        Output(FLOATING_POINT, "out") { description = "Node embeddings [N, d_out]" }
        Composition("""
            SDVariable q = sd.mmul(X, wq);
            SDVariable k = sd.mmul(X, wk);
            SDVariable v = sd.mmul(X, wv);
            SDVariable scores = sd.mmul(q, k, false, true, false).mul(scale);
            if (adjMask != null) {
                scores = scores.add(adjMask);
            }
            SDVariable attn = sd.nn().softmax(scores, 1);
            SDVariable ctx = sd.mmul(attn, v);
            SDVariable out = sd.mmul(ctx, wo);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Single-head Graph Transformer layer (Dwivedi & Bresson 2021, simplified):
            scaled dot-product self-attention, optionally restricted via additive mask.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // NNConv / ECC
    // -------------------------------------------------------------------------

    Op("nnConv") {
        Input(FLOATING_POINT, "X") { description = "Node features [rows, Fin]" }
        Input(FLOATING_POINT, "edgeFeatures") { description = "Per-edge features [nnz, edgeF]" }
        Input(FLOATING_POINT, "edgeNetW") { description = "Edge-network weight [edgeF, Fin*Fout]" }
        Input(FLOATING_POINT, "edgeNetB") { description = "Edge-network bias [Fin*Fout]" }
        Input(FLOATING_POINT, "rootW") { description = "Root/self transform [Fin, Fout]" }
        Input(FLOATING_POINT, "bias") { description = "Optional output bias [Fout]; pass null to omit" }
        Input(INT, "colIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "n") { description = "Number of nodes (for CsrEdgeGather doDiff)" }
        Arg(LONG, "fin") { description = "Input feature dimension" }
        Arg(LONG, "fout") { description = "Output feature dimension" }
        Output(FLOATING_POINT, "out") { description = "Node embeddings [rows, Fout]" }
        Composition("""
            SDVariable Xj = sd.sparse().csrEdgeGather(colIdx, X, (int) n);
            SDVariable eW = sd.mmul(edgeFeatures, edgeNetW).add(edgeNetB);
            SDVariable eWr = sd.reshape(eW, -1, fin, fout);
            SDVariable XjE = sd.reshape(Xj, -1, fin, 1);
            SDVariable prod = XjE.mul(eWr);
            SDVariable msg = sd.sum(prod, false, 1);
            SDVariable agg = sd.sparse().csrEdgeAggregate(rowPtr, msg, (int) rows, 0);
            SDVariable out = agg.add(sd.mmul(X, rootW));
            if (bias != null) {
                out = out.add(bias);
            }
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Edge-conditioned convolution / NNConv (Simonovsky & Komodakis 2017; Gilmer et al. MPNN 2017).
            An edge network maps edge features to [Fin, Fout] weight matrices applied to neighbour features.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // CompGCN
    // -------------------------------------------------------------------------

    Op("compGcnConv") {
        Input(FLOATING_POINT, "X") { description = "Entity embeddings [n, dim]" }
        Input(FLOATING_POINT, "relEmb") { description = "Relation embeddings [numRelations, dim]" }
        Input(INT, "edgeRelIdx") { description = "Per-edge relation index [nnz] INT32" }
        Input(FLOATING_POINT, "W") { description = "Output weight [dim, dim_out]" }
        Input(INT, "colIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Arg(LONG, "rows") { description = "Number of entities (nodes)" }
        Arg(LONG, "n") { description = "Number of entities (for CsrEdgeGather doDiff)" }
        Arg(INT, "compOp") { description = "Composition: 0=subtraction (TransE-style), else element-wise mult (DistMult-style)" }
        Arg(BOOL, "applyRelu") { description = "Whether to apply ReLU to the output" }
        Output(FLOATING_POINT, "out") { description = "Updated entity embeddings [rows, dim_out]" }
        Composition("""
            SDVariable Xj = sd.sparse().csrEdgeGather(colIdx, X, (int) n);
            SDVariable Rj = sd.gather(relEmb, edgeRelIdx, 0);
            SDVariable phi = (compOp == 0) ? Xj.sub(Rj) : Xj.mul(Rj);
            SDVariable agg = sd.sparse().csrEdgeAggregate(rowPtr, phi, (int) rows, 1);
            SDVariable out = sd.mmul(agg, W);
            if (applyRelu) {
                out = sd.nn().relu(out, 0.0);
            }
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            CompGCN convolution (Vashishth et al. 2020): multi-relational GNN composing entity+relation embeddings.
            compOp=0: sub (TransE-style); else: elementwise mult (DistMult-style).
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // R-GAT
    // -------------------------------------------------------------------------

    Op("rgatConvHead") {
        Input(FLOATING_POINT, "X") { description = "Node features [N, F]" }
        Input(FLOATING_POINT, "W") { description = "Linear weight [F, H]" }
        Input(FLOATING_POINT, "relEmb") { description = "Relation embeddings [numRelations, H]" }
        Input(INT, "edgeRelIdx") { description = "Per-edge relation index [nnz] INT32" }
        Input(FLOATING_POINT, "att") { description = "Shared attention vector [H, 1]" }
        Input(INT, "colIdx") { description = "CSR column indices (source nodes) [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [N+1] INT32" }
        Input(INT, "rowIdx") { description = "Per-edge destination index [nnz] INT32" }
        Arg(LONG, "nnz") { description = "Number of edges" }
        Arg(LONG, "N") { description = "Number of nodes" }
        Arg(FLOATING_POINT, "leakyAlpha") { description = "LeakyReLU negative slope" }
        Output(FLOATING_POINT, "out") { description = "Aggregated node embeddings [N, H]" }
        Composition("""
            SDVariable Wh = sd.mmul(X, W);
            SDVariable src = sd.sparse().csrEdgeGather(colIdx, Wh, (int) N);
            SDVariable dst = sd.gather(Wh, rowIdx, 0);
            SDVariable rel = sd.gather(relEmb, edgeRelIdx, 0);
            SDVariable msg = src.add(rel);
            SDVariable g = sd.nn().leakyRelu(msg.add(dst), leakyAlpha);
            SDVariable e = sd.reshape(sd.mmul(g, att), nnz);
            SDVariable alpha = sd.sparse().csrRowSoftmax(e, rowPtr, (int) N);
            SDVariable weighted = msg.mul(sd.reshape(alpha, nnz, 1));
            SDVariable out = sd.sparse().csrEdgeAggregate(rowPtr, weighted, (int) N, 0);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Single-head relational GAT: graph attention with relation embedding modulating each edge's message.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // HGT
    // -------------------------------------------------------------------------

    Op("hgtConvHead") {
        Input(FLOATING_POINT, "X") { description = "Node features [N, F]" }
        Input(FLOATING_POINT, "wq") { description = "Query weight [F, d]" }
        Input(FLOATING_POINT, "wk") { description = "Key weight [F, d]" }
        Input(FLOATING_POINT, "wv") { description = "Value weight [F, d]" }
        Input(FLOATING_POINT, "relEmb") { description = "Relation embeddings [numRelations, d]" }
        Input(INT, "edgeRelIdx") { description = "Per-edge relation index [nnz] INT32" }
        Input(INT, "colIdx") { description = "CSR column indices (source nodes) [nnz] INT32" }
        Input(INT, "rowPtr") { description = "CSR row pointers [N+1] INT32" }
        Input(INT, "rowIdx") { description = "Per-edge destination index [nnz] INT32" }
        Arg(LONG, "nnz") { description = "Number of edges" }
        Arg(LONG, "N") { description = "Number of nodes" }
        Arg(FLOATING_POINT, "scale") { description = "Attention scaling (typically 1/sqrt(d))" }
        Output(FLOATING_POINT, "out") { description = "Aggregated node embeddings [N, d]" }
        Composition("""
            SDVariable q = sd.mmul(X, wq);
            SDVariable k = sd.mmul(X, wk);
            SDVariable v = sd.mmul(X, wv);
            SDVariable qDst = sd.gather(q, rowIdx, 0);
            SDVariable kSrc = sd.sparse().csrEdgeGather(colIdx, k, (int) N);
            SDVariable vSrc = sd.sparse().csrEdgeGather(colIdx, v, (int) N);
            SDVariable rel = sd.gather(relEmb, edgeRelIdx, 0);
            SDVariable e = sd.sum(qDst.mul(kSrc.add(rel)), false, 1).mul(scale);
            SDVariable alpha = sd.sparse().csrRowSoftmax(e, rowPtr, (int) N);
            SDVariable weighted = vSrc.mul(sd.reshape(alpha, nnz, 1));
            SDVariable out = sd.sparse().csrEdgeAggregate(rowPtr, weighted, (int) N, 0);
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Simplified single-head Heterogeneous Graph Transformer (Hu et al. 2020):
            scaled dot-product attention where relation embedding modulates the key.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // HAN (Heterogeneous Attention Network)
    // -------------------------------------------------------------------------

    Op("han") {
        Input(FLOATING_POINT, "X") { description = "Node features [N, F]" }
        val metaW  = Input(FLOATING_POINT, "metaW")  { count = AtLeast(1); description = "Per-meta-path linear weight [F, H], one per meta-path" }
        val attSrc = Input(FLOATING_POINT, "attSrc") { count = AtLeast(1); description = "Per-meta-path attention source vector [H, 1]" }
        val attDst = Input(FLOATING_POINT, "attDst") { count = AtLeast(1); description = "Per-meta-path attention destination vector [H, 1]" }
        val colIdx = Input(INT, "colIdx") { count = AtLeast(1); description = "Per-meta-path CSR column indices [nnz_p] INT32" }
        val rowPtr = Input(INT, "rowPtr") { count = AtLeast(1); description = "Per-meta-path CSR row pointers [N+1] INT32" }
        val rowIdx = Input(INT, "rowIdx") { count = AtLeast(1); description = "Per-meta-path per-edge destination indices [nnz_p] INT32" }
        Input(FLOATING_POINT, "semW") { description = "Semantic-attention weight [H, A]" }
        Input(FLOATING_POINT, "semB") { description = "Semantic-attention bias [A]" }
        Input(FLOATING_POINT, "semQ") { description = "Semantic-attention query [A, 1]" }
        Arg(LONG, "nnz") { count = AtLeast(1); description = "Per-meta-path edge counts" }
        Arg(LONG, "N") { description = "Number of nodes" }
        Arg(FLOATING_POINT, "leakyAlpha") { description = "LeakyReLU negative slope for node-level GAT" }
        Output(FLOATING_POINT, "out") { description = "Fused node embeddings [N, H]" }
        Composition("""
            final int P = metaW.length;
            SDVariable[] z = new SDVariable[P];
            SDVariable[] expw = new SDVariable[P];
            SDVariable sumExp = null;
            for (int p = 0; p < P; p++) {
                z[p] = sd.gnn().gatConvHead(X, metaW[p], attSrc[p], attDst[p],
                        colIdx[p], rowPtr[p], rowIdx[p], nnz[p], N, leakyAlpha);
                SDVariable proj  = sd.math().tanh(sd.mmul(z[p], semW).add(semB));
                SDVariable score = sd.mean(sd.mmul(proj, semQ), false);
                expw[p] = sd.math().exp(score);
                sumExp = (p == 0) ? expw[p] : sumExp.add(expw[p]);
            }
            SDVariable out = null;
            for (int p = 0; p < P; p++) {
                SDVariable betaP = expw[p].div(sumExp);
                SDVariable term  = z[p].mul(betaP);
                out = (p == 0) ? term : out.add(term);
            }
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Heterogeneous Attention Network (Wang et al. 2019).
            For each meta-path: run a single-head node-level GAT (gatConvHead), then compute a semantic
            attention score via tanh(Z*semW + semB)*semQ. Softmax over meta-path scores gives mixing
            weights beta_p; final output = sum_p(beta_p * Z_p).
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // TemporalGCN (temporal graph convolutional network)
    // -------------------------------------------------------------------------

    Op("temporalGcn") {
        val Xt = Input(FLOATING_POINT, "Xt") { count = AtLeast(1); description = "Per-timestep node features, each [N, F]" }
        Input(FLOATING_POINT, "W") { description = "Shared spatial GCN weight [F, H]" }
        Input(FLOATING_POINT, "bias") { description = "Shared spatial GCN bias [H]; pass null to omit" }
        Input(FLOATING_POINT, "aNormVals") { description = "CSR values of the normalised adjacency [nnz]" }
        Input(INT, "aNormColIdx") { description = "CSR column indices [nnz] INT32" }
        Input(INT, "aNormRowPtr") { description = "CSR row pointers [rows+1] INT32" }
        Input(FLOATING_POINT, "tempW") { description = "Temporal-attention weight [H, A]" }
        Input(FLOATING_POINT, "tempQ") { description = "Temporal-attention query [A, 1]" }
        Arg(LONG, "rows") { description = "Number of nodes" }
        Arg(LONG, "cols") { description = "Columns of A_norm (= rows for square graphs)" }
        Arg(BOOL, "applyRelu") { description = "Whether the spatial GCN applies ReLU" }
        Output(FLOATING_POINT, "out") { description = "Temporally-fused node embeddings [N, H]" }
        Composition("""
            final int T = Xt.length;
            SDVariable[] h = new SDVariable[T];
            SDVariable[] expw = new SDVariable[T];
            SDVariable sumExp = null;
            for (int t = 0; t < T; t++) {
                h[t] = sd.gnn().gcnConv(Xt[t], W, bias, aNormVals, aNormColIdx, aNormRowPtr, rows, cols, applyRelu);
                SDVariable score = sd.mean(sd.mmul(sd.math().tanh(sd.mmul(h[t], tempW)), tempQ), false);
                expw[t] = sd.math().exp(score);
                sumExp = (t == 0) ? expw[t] : sumExp.add(expw[t]);
            }
            SDVariable out = null;
            for (int t = 0; t < T; t++) {
                SDVariable betaT = expw[t].div(sumExp);
                SDVariable term  = h[t].mul(betaT);
                out = (t == 0) ? term : out.add(term);
            }
        """.trimIndent())
        Doc(Language.ANY, DocScope.ALL) {
            """
            Temporal Graph Convolutional Network: weight-shared spatial GCN applied at each timestep,
            fused by temporal attention. For each timestep t: h[t] = gcnConv(X_t, W, bias, A_norm).
            Temporal attention score = mean(tanh(h[t]*tempW)*tempQ). Softmax gives beta_t;
            out = sum_t(beta_t * h[t]).
            """.trimIndent()
        }
    }
}
