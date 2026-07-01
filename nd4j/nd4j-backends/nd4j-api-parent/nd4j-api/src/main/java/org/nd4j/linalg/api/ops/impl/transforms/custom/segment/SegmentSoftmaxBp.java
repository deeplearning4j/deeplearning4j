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
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Backward pass for {@link SegmentSoftmax}.
 *
 * <p>C++ op name: {@code segment_softmax_bp}
 */
public class SegmentSoftmaxBp extends DynamicCustomOp {

    public SegmentSoftmaxBp() {}

    public SegmentSoftmaxBp(SameDiff sd,
                              SDVariable logits,
                              SDVariable segmentIds,
                              SDVariable fwdOut,
                              SDVariable gradOut,
                              long K) {
        super(sd, new SDVariable[]{logits, segmentIds, fwdOut, gradOut}, false);
        addIArgument(K);
    }

    @Override
    public String opName() { return "segment_softmax_bp"; }

    @Override
    public int getNumOutputs() { return 1; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(0));
    }
}
