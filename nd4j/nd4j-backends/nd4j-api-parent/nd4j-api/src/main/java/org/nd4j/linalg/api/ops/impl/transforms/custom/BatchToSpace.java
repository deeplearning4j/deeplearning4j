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

package org.nd4j.linalg.api.ops.impl.transforms.custom;


import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * N-dimensional batch to space operation. Transforms data from a tensor from batch dimension into M spatial dimensions
 * according to the "blocks" specified (a vector of length M). Afterwards the spatial dimensions are optionally cropped,
 * as specified in "crops", a tensor of dim (M, 2), denoting the crop range.
 * <p>
 * Example:
 * input:        [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
 * input shape:  [4, 1, 1, 1]
 * blocks:       [2, 2]
 * crops:        [[0, 0], [0, 0]]
 * <p>
 * output:       [[[[1], [2]], [[3], [4]]]]
 * output shape: [1, 2, 2, 1]
 *
 * @author Max Pumperla
 */
public class BatchToSpace extends DynamicCustomOp {

    private int[] blocks;
    private int[][] crops;

    public BatchToSpace() {
    }

    public BatchToSpace(SameDiff sameDiff, SDVariable x, int[] blocks, int[] croppingTop, int... croppingBottom) {
        this(sameDiff, x, blocks, new int[][]{croppingTop, croppingBottom}, false);
    }

    public BatchToSpace(SameDiff sameDiff, SDVariable x, int[] blocks, int[][] crops, boolean inPlace) {
        this(sameDiff, new SDVariable[]{x}, blocks, crops, inPlace);
    }

    public BatchToSpace(SameDiff sameDiff, SDVariable[] args, int[] blocks, int[][] crops, boolean inPlace) {
        super(null, sameDiff, new SDVariable[]{args[0], sameDiff.constant(Nd4j.createFromArray(crops))}, inPlace);

        this.blocks = blocks;
        this.crops = crops;

        for (val b : blocks)
            addIArgument(b);
    }

    public BatchToSpace(INDArray x, int[] blocks, int[] croppingTop, int... croppingBottom) {
        addInputArgument(x);
        int[][] crops = new int[][]{croppingTop, croppingBottom};
        this.blocks = blocks;
        this.crops = crops;

        for (val b : blocks)
            addIArgument(b);
    }


    @Override
    public String opName() {
        return "batch_to_space";
    }

    @Override
    public String onnxName() {
        return "batch_to_space";
    }

    @Override
    public String tensorflowName() {
        return "BatchToSpace";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // Inverse of batch to space is space to batch with same blocks and padding as crops
        SDVariable gradient = sameDiff.setupFunction(i_v.get(0));
        return Arrays.asList(sameDiff.cnn().spaceToBatch(gradient, blocks, crops[0], crops[1]));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        return Collections.singletonList(dataTypes.get(0));
    }
}
