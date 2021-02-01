/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.linalg.api.ops.random.custom;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

/**
 * Random exponential distribution: p(x) = lambda * exp(-lambda * x)
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class RandomExponential extends DynamicCustomOp {
    private double lambda = 0.0;
    private DataType dataType = DataType.DOUBLE;

    public RandomExponential() {
        //
    }

    public RandomExponential(SameDiff sd, SDVariable shape, double lambda){
        super(null, sd, new SDVariable[]{shape});
        Preconditions.checkState(lambda >= 0, "Lambda parameter must be > 0 - got %s", lambda);
        this.lambda = lambda;
        addTArgument(lambda);
    }

    public RandomExponential(SameDiff sd, double lambda, DataType dataType, long... shape){
        super(null, sd, new SDVariable[]{sd.constant(Nd4j.createFromArray(shape))});
        this.lambda = lambda;
        addTArgument(lambda);
        this.dataType = dataType;
        addDArgument(dataType);
        addIArgument(shape);
    }

    public RandomExponential(double lambda, DataType datatype, long... shape){
        this(Nd4j.createFromArray(shape), Nd4j.createUninitialized(datatype, shape), lambda);
    }

    public RandomExponential(INDArray shape,INDArray out, double lambda){
        super(null, new INDArray[]{shape}, new INDArray[]{out}, Collections.singletonList(lambda), (List<Integer>)null);
        this.lambda = lambda;
    }

    @Override
    public String opName() {
        return "random_exponential";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got %s", getClass(), inputDataTypes);
        //Input data type specifies the shape; output data type should be any float
        //TODO MAKE CONFIGUREABLE - https://github.com/deeplearning4j/deeplearning4j/issues/6854
        return Collections.singletonList(DataType.FLOAT);
    }
}
