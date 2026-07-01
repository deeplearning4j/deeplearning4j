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

package org.nd4j.linalg.api.ops.impl.transforms.custom.segment;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Segment-wise softmax: softmax over each contiguous run of elements sharing a segment ID.
 *
 * <p>Used in batched GAT attention to normalize attention scores within each graph in a
 * block-diagonal batch (where segmentIds = batchVec from {@link org.nd4j.linalg.api.ops.impl.sparse.GraphDisjointUnion}).
 *
 * <p>For 1D logits:
 * <pre>
 *   out[i] = exp(logits[i]) / sum_{j in seg(i)} exp(logits[j])
 * </pre>
 *
 * <p>C++ op name: {@code segment_softmax}
 * <ul>
 *   <li>Inputs: logits[N,...] (float), segmentIds[N] (INT32, sorted)</li>
 *   <li>IArgs: K (number of segments)</li>
 *   <li>Output: out[N,...] (same shape and dtype as logits)</li>
 * </ul>
 */
public class SegmentSoftmax extends DynamicCustomOp {

    private final long K;

    public SegmentSoftmax() { this.K = 0; }

    /**
     * SameDiff constructor.
     *
     * @param sd         the SameDiff graph
     * @param logits     [N] or [N,...] float attention logits
     * @param segmentIds [N] INT32 sorted segment IDs in [0, K)
     * @param K          number of segments
     */
    public SegmentSoftmax(SameDiff sd, SDVariable logits, SDVariable segmentIds, long K) {
        super(sd, new SDVariable[]{logits, segmentIds}, false);
        this.K = K;
        addIArgument(K);
    }

    /**
     * Eager (INDArray) constructor.
     */
    public SegmentSoftmax(INDArray logits, INDArray segmentIds, long K) {
        super(new INDArray[]{logits, segmentIds}, null);
        this.K = K;
        addIArgument(K);
    }

    @Override
    public String opName() { return "segment_softmax"; }

    @Override
    public int getNumOutputs() { return 1; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradOut    = grads.get(0);
        SDVariable fwdOut     = outputVariables()[0];
        SDVariable logitsArg  = arg(0);
        SDVariable segIdsArg  = arg(1);

        SDVariable dLogits = new SegmentSoftmaxBp(sameDiff, logitsArg, segIdsArg, fwdOut, gradOut, K)
                .outputVariable();
        return Arrays.asList(dLogits, sameDiff.zerosLike(segIdsArg));
    }
}
