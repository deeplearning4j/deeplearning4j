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

package org.nd4j.linalg.api.ops.impl.transforms.gradient;


import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 *
 */
public class LogSoftMaxDerivative extends DynamicCustomOp {
    public LogSoftMaxDerivative(SameDiff sameDiff, SDVariable in, SDVariable gradO) {
        super(sameDiff, new SDVariable[]{in, gradO});
    }

    public LogSoftMaxDerivative() {
    }

    public LogSoftMaxDerivative(INDArray in, INDArray gradO, INDArray out) {
        super(null, new INDArray[]{in, gradO}, new INDArray[]{out});
    }

    public LogSoftMaxDerivative(SameDiff sameDiff, SDVariable arg, SDVariable wrt, int dimension) {
        this(sameDiff, arg, wrt);
        this.addIArgument(dimension);
    }

    /**
     * The opName of this operation
     *
     * @return the opName of this operation
     */
    @Override
    public String opName() {
        return "log_softmax_bp";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Differentation of op not supported: " + getClass().getName());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inTypes){
        Preconditions.checkState(inTypes != null && inTypes.size() == 2, "Expected 2 input datatypes for %s, got %s",
                getClass(), inTypes);
        return Collections.singletonList(inTypes.get(0));
    }
}
