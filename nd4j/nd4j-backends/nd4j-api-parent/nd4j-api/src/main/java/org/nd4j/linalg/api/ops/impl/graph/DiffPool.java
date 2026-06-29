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

package org.nd4j.linalg.api.ops.impl.graph;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmm;

/**
 * DiffPool: differentiable hierarchical graph pooling (Ying et al. 2018).
 *
 * <p>Given a graph with {@code N} nodes, a dense adjacency stored in CSR format,
 * node features {@code H[N,F]}, and raw cluster-assignment logits produced by a
 * GNN (e.g. via {@code sd.gnn().gcnConv}), coarsens the graph from {@code N} nodes
 * down to {@code K} clusters:
 *
 * <pre>
 *   S       = softmax(assignLogits, axis=1)           [N, K]  soft assignment (rows sum to 1)
 *   Xc      = S^T · H                                  [K, F]  coarsened node features
 *   AS      = A · S        (sparse × dense SpMM)       [N, K]
 *   Ac      = S^T · AS                                 [K, K]  coarsened adjacency
 *   entropy = -mean( sum( S * log(S + 1e-12), axis=1 ) )       auxiliary entropy regulariser
 * </pre>
 *
 * <p>No new native ops are introduced.  Gradients flow automatically through
 * the component ops' {@code doDiff} implementations ({@link CsrSpmm},
 * {@code SoftMax}, {@code Mmul}, {@code Transpose}, {@code Log}, …).
 *
 * <p>Typical usage:
 * <pre>{@code
 *   SameDiff sd = SameDiff.create();
 *   // ... build GNN to produce assignLogits[N,K] ...
 *   SDVariable[] pooled = DiffPool.apply(sd, "pool",
 *           aValues, aColIdx, aRowPtr, H, assignLogits, N, K);
 *   SDVariable Xc          = pooled[0];   // [K, F] coarsened features
 *   SDVariable Ac          = pooled[1];   // [K, K] coarsened adjacency
 *   SDVariable entropyLoss = pooled[2];   // scalar auxiliary regulariser
 * }</pre>
 *
 * <p>Verified API signatures used here (all exist as of the current codebase):
 * <ul>
 *   <li>{@code sd.nn().softmax(SDVariable x, int dimension)} — SDNN line ~5917</li>
 *   <li>{@code sd.transpose(SDVariable x)} — SDBaseOps line ~5477</li>
 *   <li>{@code sd.mmul(SDVariable x, SDVariable y)} — SDBaseOps line ~2414</li>
 *   <li>{@code new CsrSpmm(SameDiff, SDVariable values, SDVariable colIdx, SDVariable rowPtr,
 *         SDVariable B, long rows, long cols, boolean transposeA)} — CsrSpmm.java line ~91</li>
 *   <li>{@code sd.math().add(SDVariable x, double value)} — SDMath line ~273</li>
 *   <li>{@code sd.math().log(SDVariable x)} — SDMath line ~2573</li>
 *   <li>{@code sd.math().mul(SDVariable x, SDVariable y)} — SDMath line ~3264</li>
 *   <li>{@code sd.math().mul(SDVariable x, double value)} — SDMath line ~3296</li>
 *   <li>{@code sd.math().sum(SDVariable in, boolean keepDims, long... dimensions)} — SDMath line ~5349</li>
 *   <li>{@code sd.math().mean(SDVariable in, long... dimensions)} — SDMath line ~2936</li>
 * </ul>
 */
public final class DiffPool {

    private DiffPool() {
        // static utility class — not instantiable
    }

    /**
     * Apply DiffPool coarsening to a single graph.
     *
     * <p>The adjacency matrix is assumed square ({@code N × N}), symmetric and stored
     * in CSR format via separate {@code aValues}, {@code aColIdx}, {@code aRowPtr}
     * tensors — exactly the representation produced by {@link Nd4j#toSparse} and the
     * other sparse ops in this package.
     *
     * @param sd            the SameDiff computation graph
     * @param name          optional name prefix for the three output variables;
     *                      if non-null the outputs are named
     *                      {@code name + "_Xc"}, {@code name + "_Ac"},
     *                      {@code name + "_entropyLoss"}; pass {@code null} for
     *                      auto-generated names
     * @param aValues       SDVariable {@code [nnz]} non-zero values of the CSR adjacency
     *                      (floating dtype, typically DOUBLE or FLOAT)
     * @param aColIdx       SDVariable {@code [nnz]} INT32 column indices of the CSR
     *                      adjacency
     * @param aRowPtr       SDVariable {@code [N+1]} INT32 row-pointer array of the CSR
     *                      adjacency
     * @param H             SDVariable {@code [N, F]} node-feature matrix
     * @param assignLogits  SDVariable {@code [N, K]} raw cluster-assignment logits,
     *                      typically the output of a GNN layer applied to the graph
     * @param N             number of nodes (= number of rows = number of columns of the
     *                      square adjacency matrix)
     * @param K             number of target clusters (coarsened nodes)
     * @return array of three SDVariables:
     *         <ul>
     *           <li>[0] {@code Xc} — coarsened node features {@code [K, F]}</li>
     *           <li>[1] {@code Ac} — coarsened adjacency {@code [K, K]}</li>
     *           <li>[2] {@code entropyLoss} — scalar entropy regulariser
     *               (add to the task loss with a small weight, e.g. 0.01)</li>
     *         </ul>
     */
    public static SDVariable[] apply(SameDiff sd, String name,
                                     SDVariable aValues, SDVariable aColIdx, SDVariable aRowPtr,
                                     SDVariable H, SDVariable assignLogits,
                                     long N, long K) {

        // ------------------------------------------------------------------
        // S[N, K] = softmax(assignLogits, axis=1)
        //   Each row of S sums to 1 — it is a soft assignment of node i to
        //   cluster k.  sd.nn() is the SDNN accessor on SameDiff.
        // ------------------------------------------------------------------
        SDVariable S = sd.nn().softmax(assignLogits, 1);

        // ------------------------------------------------------------------
        // St[K, N] = S^T
        //   Reused in both the Xc and Ac computations, so compute once.
        // ------------------------------------------------------------------
        SDVariable St = sd.transpose(S);

        // ------------------------------------------------------------------
        // Xc[K, F] = S^T · H[N, F]
        //   Dense matmul: cluster-aggregated node features.
        // ------------------------------------------------------------------
        SDVariable Xc = sd.mmul(St, H);

        // ------------------------------------------------------------------
        // AS[N, K] = A · S   (sparse × dense matrix multiply)
        //   A is the N×N adjacency in CSR format.  CsrSpmm with
        //   transposeA=false computes A · B, output shape = [N, K].
        // ------------------------------------------------------------------
        SDVariable AS = sd.sparse().csrSpmm(aValues, aColIdx, aRowPtr, S, (int) N, (int) N, false);

        // ------------------------------------------------------------------
        // Ac[K, K] = S^T · AS
        //   Dense matmul: coarsened adjacency.
        // ------------------------------------------------------------------
        SDVariable Ac = sd.mmul(St, AS);

        // ------------------------------------------------------------------
        // entropyLoss = -mean( sum( S * log(S + ε), axis=1 ) )
        //
        //   The entropy regulariser encourages crisp (near-one-hot)
        //   assignment matrices.  Adding ε=1e-12 before log avoids NaN at
        //   S_ik = 0.
        //
        //   Step by step:
        //     Seps   = S + 1e-12                  [N, K]
        //     logSeps = log(Seps)                  [N, K]
        //     SlogS  = S * log(S + ε)              [N, K]  (element-wise)
        //     rowEnt = sum(SlogS, axis=1)           [N]    (per-node entropy, ≤ 0)
        //     meanEnt = mean(rowEnt)               scalar
        //     entropyLoss = -meanEnt               scalar (≥ 0)
        // ------------------------------------------------------------------
        SDVariable Seps     = sd.math().add(S, 1e-12);
        SDVariable logSeps  = sd.math().log(Seps);
        SDVariable SlogS    = sd.math().mul(S, logSeps);
        SDVariable rowEnt   = sd.math().sum(SlogS, false, 1L);
        SDVariable meanEnt  = sd.math().mean(rowEnt);
        SDVariable entropyLoss = sd.math().mul(meanEnt, -1.0);

        // ------------------------------------------------------------------
        // Optionally assign caller-provided names to the three outputs.
        // Following the pattern used in GraphPooling.java.
        // ------------------------------------------------------------------
        if (name != null) {
            Xc          = sd.updateVariableNameAndReference(Xc,          name + "_Xc");
            Ac          = sd.updateVariableNameAndReference(Ac,          name + "_Ac");
            entropyLoss = sd.updateVariableNameAndReference(entropyLoss, name + "_entropyLoss");
        }

        return new SDVariable[]{Xc, Ac, entropyLoss};
    }
}
