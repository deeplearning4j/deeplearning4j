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

package org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Base arithmetic backprop operation
 *
 * @author Alex Black
 */
public abstract class BaseArithmeticBackpropOp extends BaseDynamicTransformOp {

    public BaseArithmeticBackpropOp() {}

    public BaseArithmeticBackpropOp(SameDiff sameDiff, SDVariable x, SDVariable y, SDVariable eps) {
        super(sameDiff, new SDVariable[]{x,y,eps}, false);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(){
        return Nd4j.getExecutioner().calculateOutputShape(this);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 3, "Expected exactly 3 input datatype, got input %s", dataTypes);
        //Gradient types: same as input
        return Arrays.asList(arg(0).dataType(), arg(1).dataType());
    }
}
