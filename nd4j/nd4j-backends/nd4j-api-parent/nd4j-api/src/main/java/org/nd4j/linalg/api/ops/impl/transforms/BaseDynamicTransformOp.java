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

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public abstract class BaseDynamicTransformOp extends DynamicCustomOp {

    public BaseDynamicTransformOp() {}

    public BaseDynamicTransformOp(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public BaseDynamicTransformOp(INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }


    @Override
    public List<long[]> calculateOutputShape() {
        val args = args();
        if(args.length < 2) {
            if(args[0] == null || args[0].getShape() == null) {
                return Collections.emptyList();
            }

            return Arrays.asList(args[0].getShape());
        }

        val firstArgShape = args[0].getShape();
        val secondArgShape = args[1].getShape();
        if(args[0] == null || args[0].getShape() == null) {
            return Collections.emptyList();
        }

        if(args[1] == null || args[1].getShape() == null) {
            return Collections.emptyList();
        }

        if(Arrays.equals(firstArgShape, secondArgShape)){
            return Collections.singletonList(firstArgShape);
        }
        //Handle broadcast shape: [1,4]+[3,1] = [3,4]
        Shape.assertBroadcastable(firstArgShape, secondArgShape, this.getClass());
        val outShape = Shape.broadcastOutputShape(firstArgShape, secondArgShape);

        return Collections.singletonList(outShape);
    }
}
