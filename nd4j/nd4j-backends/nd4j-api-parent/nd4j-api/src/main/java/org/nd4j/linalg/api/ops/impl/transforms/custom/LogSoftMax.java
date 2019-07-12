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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.OldSoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Log(softmax(X))
 *
 * @author Alex Black
 */

public class LogSoftMax extends DynamicCustomOp {

    private Integer dimension = null;

    public LogSoftMax(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, i_v);
    }

    public LogSoftMax() {
    }

    public LogSoftMax(INDArray x, INDArray z) {
        super(null, x, z, null, null);
    }

    public LogSoftMax(INDArray x) {
        this(x, x);
    }

    public LogSoftMax(SameDiff sameDiff, SDVariable i_v, int dimension) {
        this(sameDiff, i_v);
        this.dimension = dimension;
        addIArgument(dimension);
    }


    @Override
    public String opName() {
        return "log_softmax";
    }
    @Override
    public String tensorflowName() {
        return "LogSoftmax";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        if(dimension == null) {
            SDVariable ret = f().logSoftmaxDerivative(arg(), i_v.get(0));
            return Collections.singletonList(ret);
        } else {
            SDVariable ret = f().logSoftmaxDerivative(arg(), i_v.get(0), dimension);
            return Collections.singletonList(ret);
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inTypes){
        Preconditions.checkState(inTypes != null && inTypes.size() == 1, "Expected 1 input datatype for %s, got %s",
                getClass(), inTypes);
        if(inTypes.get(0).isFPType())
            return Collections.singletonList(inTypes.get(0));
        return Collections.singletonList(Nd4j.defaultFloatingPointType());
    }
}
