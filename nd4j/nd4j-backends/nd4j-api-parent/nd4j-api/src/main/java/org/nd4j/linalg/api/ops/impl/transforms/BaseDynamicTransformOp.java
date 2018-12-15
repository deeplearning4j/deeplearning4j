/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@Slf4j
public abstract class BaseDynamicTransformOp extends DynamicCustomOp {

    public BaseDynamicTransformOp() {}

    public BaseDynamicTransformOp(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public BaseDynamicTransformOp(INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }


    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        val args = args();
        if(args.length < 2) {
            if(args[0] == null || args[0].getShape() == null) {
                return Collections.emptyList();
            }
            val dtypeX = args[0].getArr() != null ? args[0].getArr().dataType() : args[0].dataType();

            return Collections.singletonList(LongShapeDescriptor.fromShape(args[0].getShape(), dtypeX));
        }

        val firstArgShape = args[0].getShape();
        val secondArgShape = args[1].getShape();
        if(args[0] == null || args[0].getShape() == null) {
            return Collections.emptyList();
        }

        if(args[1] == null || args[1].getShape() == null) {
            return Collections.emptyList();
        }

        // detecting datatype based on both args
        val dtypeX = args[0].getArr() != null ? args[0].getArr().dataType() : args[0].dataType();
        val dtypeY = args[1].getArr() != null ? args[1].getArr().dataType() : args[1].dataType();

        val dtypeZ = Shape.pickPairwiseDataType(dtypeX, dtypeY);

        if(Arrays.equals(firstArgShape, secondArgShape)){
            try {
                return Collections.singletonList(LongShapeDescriptor.fromShape(firstArgShape, dtypeZ));
            } catch (Throwable e) {
                throw new RuntimeException("calculateOutputShape() failed for [" + this.opName() + "]", e);
            }
        } else {
            //Handle broadcast shape: [1,4]+[3,1] = [3,4]
            Shape.assertBroadcastable(firstArgShape, secondArgShape, this.getClass());
            val outShape = Shape.broadcastOutputShape(firstArgShape, secondArgShape);

            return Collections.singletonList(LongShapeDescriptor.fromShape(outShape, dtypeZ));
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes, got input %s", dataTypes);

        DataType z = Shape.pickPairwiseDataType(dataTypes.get(0), dataTypes.get(1));
        return Collections.singletonList(z);
    }
}
