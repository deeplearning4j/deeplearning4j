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
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.List;

/**
 * Backward pass for {@link GraphDisjointUnion}.
 *
 * <p>Splits the combined gradient tensors back into K per-graph gradients.
 *
 * <p>C++ op name: {@code graph_disjoint_union_bp}
 */
public class GraphDisjointUnionBp extends DynamicCustomOp {

    private int K;

    public GraphDisjointUnionBp() { this.K = 0; }

    /** Returns K, falling back to iArguments[0] when the no-arg constructor was used (post-deserialization). */
    private int effectiveK() {
        if (K != 0) return K;
        return iArguments != null && !iArguments.isEmpty() ? iArguments.get(0).intValue() : 0;
    }

    public GraphDisjointUnionBp(SameDiff sd,
                                 SDVariable[] Xs, SDVariable[] vals,
                                 SDVariable[] colIdxs, SDVariable[] rowPtrs,
                                 SDVariable dXcombined, SDVariable dValsCombined) {
        super(sd, concat(Xs, vals, colIdxs, rowPtrs, dXcombined, dValsCombined), false);
        this.K = Xs.length;
        addIArgument(K);
    }

    private static SDVariable[] concat(SDVariable[] Xs, SDVariable[] vals,
                                        SDVariable[] colIdxs, SDVariable[] rowPtrs,
                                        SDVariable dX, SDVariable dV) {
        List<SDVariable> list = new ArrayList<>();
        for (SDVariable v : Xs)      list.add(v);
        for (SDVariable v : vals)    list.add(v);
        for (SDVariable v : colIdxs) list.add(v);
        for (SDVariable v : rowPtrs) list.add(v);
        list.add(dX);
        list.add(dV);
        return list.toArray(new SDVariable[0]);
    }

    @Override
    public String opName() { return "graph_disjoint_union_bp"; }

    @Override
    public int getNumOutputs() {
        int k = effectiveK();
        return k == 0 ? -1 : 2 * k;
    }

    @Override
    public int numOutputArguments() { return 2 * effectiveK(); }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // K dX dtypes (same as X_k) + K dVals dtypes (same as vals_k)
        int k = effectiveK();
        List<DataType> out = new ArrayList<>();
        for (int i = 0; i < k; i++) out.add(dataTypes.get(i));
        for (int i = 0; i < k; i++) out.add(dataTypes.get(k + i));
        return out;
    }
}
