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

package org.nd4j.linalg.api.ops.random.custom;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

@Slf4j
public class RandomBernoulli extends DynamicCustomOp {
    private double p = 0.0;

    public RandomBernoulli() {
        //
    }

    public RandomBernoulli(SameDiff sd, SDVariable shape, double p){
        super(null, sd, new SDVariable[]{shape});
        Preconditions.checkState(p >= 0 && p <= 1.0, "Probability must be between 0 and 1 - got %s", p);
        this.p = p;
        addTArgument(p);
    }

    public RandomBernoulli(INDArray shape, INDArray out, double p){
        super(null, new INDArray[]{shape}, new INDArray[]{out}, Collections.singletonList(p), (List<Integer>)null);
        Preconditions.checkState(p >= 0 && p <= 1.0, "Probability must be between 0 and 1 - got %s", p);
    }

    @Override
    public String opName() {
        return "random_bernoulli";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got %s", getClass(), inputDataTypes);
        //Input data type specifies the shape; output data type should be any float
        //TODO MAKE CONFIGUREABLE - https://github.com/eclipse/deeplearning4j/issues/6854
        return Collections.singletonList(DataType.FLOAT);
    }
}
