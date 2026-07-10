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

package org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;

import java.util.Arrays;
import java.util.List;

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
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 3, "Expected exactly 3 input datatypes for %s, got input %s", getClass(), dataTypes);
        DataType gradType = dataTypes.contains(DataType.DOUBLE) ? DataType.DOUBLE : DataType.FLOAT;
        return Arrays.asList(gradType, gradType);
    }

    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        // Backprop ops produce 2 outputs (gradient for each input of the forward op).
        // The parent BaseDynamicTransformOp.calculateOutputShapeFromInputs() returns
        // only 1 shape (correct for forward binary ops, wrong for backprop).
        // Return null to fall through to the C++ shape function which correctly
        // returns 2 shapes via SHAPELIST(CONSTANT(x), CONSTANT(y)).
        return null;
    }
}
