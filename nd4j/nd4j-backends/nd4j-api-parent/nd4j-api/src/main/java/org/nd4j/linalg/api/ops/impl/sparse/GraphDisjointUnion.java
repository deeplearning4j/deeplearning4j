/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.sparse;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Builds a block-diagonal (disjoint union) graph from K variable-size graphs.
 *
 * <p>Input layout (4*K tensors, where K is an IArg):
 * <ul>
 *   <li>Inputs [0..K-1]:       X_k       [N_k, F]  float  — node features per graph</li>
 *   <li>Inputs [K..2K-1]:      vals_k    [nnz_k]   float  — CSR edge weights per graph</li>
 *   <li>Inputs [2K..3K-1]:     colIdx_k  [nnz_k]   int    — CSR column indices per graph</li>
 *   <li>Inputs [3K..4K-1]:     rowPtr_k  [N_k+1]   int    — CSR row pointers per graph</li>
 * </ul>
 *
 * <p>Outputs:
 * <ul>
 *   <li>[0] X_combined      [sumN, F]     float</li>
 *   <li>[1] vals_combined   [sumNnz]      float</li>
 *   <li>[2] colIdx_combined [sumNnz]      int (shifted by cumulative node count)</li>
 *   <li>[3] rowPtr_combined [sumN+1]      int (offset by cumulative nnz)</li>
 *   <li>[4] batchVec        [sumN]        INT32 — node → graph index (0-based)</li>
 * </ul>
 *
 * <p>C++ op name: {@code graph_disjoint_union}
 */
public class GraphDisjointUnion extends DynamicCustomOp {

    private int K;

    /** No-arg constructor for op-registry reflection. */
    public GraphDisjointUnion() { this.K = 0; }

    /** Returns K, falling back to iArguments[0] when the no-arg constructor was used (post-deserialization). */
    private int effectiveK() {
        if (K != 0) return K;
        return iArguments != null && !iArguments.isEmpty() ? iArguments.get(0).intValue() : 0;
    }

    /**
     * SameDiff constructor.
     *
     * @param sd      the SameDiff graph
     * @param Xs      K node-feature SDVariables [N_k, F]
     * @param vals    K edge-weight SDVariables [nnz_k]
     * @param colIdxs K column-index SDVariables [nnz_k] (int)
     * @param rowPtrs K row-pointer SDVariables [N_k+1] (int)
     */
    public GraphDisjointUnion(SameDiff sd,
                               SDVariable[] Xs,
                               SDVariable[] vals,
                               SDVariable[] colIdxs,
                               SDVariable[] rowPtrs) {
        super(sd, concat(Xs, vals, colIdxs, rowPtrs), false);
        this.K = Xs.length;
        addIArgument(K);
    }

    /**
     * Eager (INDArray) constructor.
     *
     * @param Xs      K node-feature arrays [N_k, F]
     * @param vals    K edge-weight arrays [nnz_k]
     * @param colIdxs K column-index arrays [nnz_k]
     * @param rowPtrs K row-pointer arrays [N_k+1]
     */
    public GraphDisjointUnion(INDArray[] Xs, INDArray[] vals, INDArray[] colIdxs, INDArray[] rowPtrs) {
        super(concat(Xs, vals, colIdxs, rowPtrs), null);
        this.K = Xs.length;
        addIArgument(K);
    }

    private static SDVariable[] concat(SDVariable[] a, SDVariable[] b, SDVariable[] c, SDVariable[] d) {
        List<SDVariable> list = new ArrayList<>();
        for (SDVariable v : a) list.add(v);
        for (SDVariable v : b) list.add(v);
        for (SDVariable v : c) list.add(v);
        for (SDVariable v : d) list.add(v);
        return list.toArray(new SDVariable[0]);
    }

    private static INDArray[] concat(INDArray[] a, INDArray[] b, INDArray[] c, INDArray[] d) {
        List<INDArray> list = new ArrayList<>();
        for (INDArray v : a) list.add(v);
        for (INDArray v : b) list.add(v);
        for (INDArray v : c) list.add(v);
        for (INDArray v : d) list.add(v);
        return list.toArray(new INDArray[0]);
    }

    @Override
    public String opName() { return "graph_disjoint_union"; }

    @Override
    public int getNumOutputs() { return 5; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // dataTypes[0] = X_0 dtype (float)
        // dataTypes[K] = vals_0 dtype (float)
        // dataTypes[2K] = colIdx_0 dtype (int)
        int k = effectiveK();
        DataType floatDt = dataTypes.get(0);
        DataType idxDt   = dataTypes.get(2 * k);
        return Arrays.asList(floatDt, floatDt, idxDt, idxDt, DataType.INT32);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        // grads.get(0) = dX_combined [sumN, F]
        // grads.get(1) = dVals_combined [sumNnz]
        // grads.get(2..4) are structural (no gradient for colIdx, rowPtr, batchVec)
        SDVariable dXcombined    = grads.get(0);
        SDVariable dValsCombined = grads.get(1);

        int k = effectiveK();

        // Collect forward inputs
        SDVariable[] Xs      = new SDVariable[k];
        SDVariable[] vals    = new SDVariable[k];
        SDVariable[] colIdxs = new SDVariable[k];
        SDVariable[] rowPtrs = new SDVariable[k];
        for (int i = 0; i < k; i++) {
            Xs[i]      = arg(i);
            vals[i]    = arg(k + i);
            colIdxs[i] = arg(2 * k + i);
            rowPtrs[i] = arg(3 * k + i);
        }

        // Backward op outputs 2K gradients (K dXs + K dVals)
        GraphDisjointUnionBp bp = new GraphDisjointUnionBp(sameDiff, Xs, vals, colIdxs, rowPtrs,
                                                            dXcombined, dValsCombined);
        SDVariable[] bpOuts = bp.outputVariables();

        // Build result: K dX, K dVals, then zeros for colIdx/rowPtr
        List<SDVariable> result = new ArrayList<>();
        for (int i = 0; i < k; i++) result.add(bpOuts[i]);          // dXs
        for (int i = 0; i < k; i++) result.add(bpOuts[k + i]);      // dVals
        for (int i = 0; i < k; i++) result.add(sameDiff.zerosLike(arg(2 * k + i)));  // dColIdx = 0
        for (int i = 0; i < k; i++) result.add(sameDiff.zerosLike(arg(3 * k + i)));  // dRowPtr = 0
        return result;
    }
}
