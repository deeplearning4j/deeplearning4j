/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.FloorModBpOp;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Collections;
import java.util.List;

/**
 * Floor mod
 *
 * @author raver119@gmail.com
 */
public class FloorModOp extends BaseDynamicTransformOp {
    public FloorModOp() {}

    public FloorModOp(SameDiff sameDiff, SDVariable x, SDVariable y) {
        super(sameDiff, new SDVariable[]{x, y}, false);
    }

    public FloorModOp(@NonNull INDArray x, @NonNull INDArray y) {
        this(new INDArray[]{x, y}, null);
    }

    public FloorModOp(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    @Override
    public String opName() {
        return "floormod";
    }

    @Override
    public String onnxName() {
        return "FloorMod";
    }

    @Override
    public String tensorflowName() {
        return "FloorMod";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return new FloorModBpOp(sameDiff, larg(), rarg(), f1.get(0)).outputs();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes for %s, got input %s", getClass(), dataTypes);

        DataType z = Shape.pickPairwiseDataType(dataTypes.get(0), dataTypes.get(1));
        return Collections.singletonList(z);
    }
}
