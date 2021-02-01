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

package org.nd4j.linalg.api.ops.impl.loss.bp;

import lombok.NonNull;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

public abstract class BaseLossBp extends DynamicCustomOp {

    protected LossReduce lossReduce;

    public BaseLossBp(@NonNull SameDiff sameDiff, @NonNull LossReduce lossReduce, @NonNull SDVariable predictions, @NonNull SDVariable weights,
                      @NonNull SDVariable labels){
        super(null, sameDiff, new SDVariable[]{predictions, weights, labels});
        this.lossReduce = lossReduce;
        addArgs();
    }

    protected BaseLossBp(){ }

    protected void addArgs(){
        iArguments.clear();
        tArguments.clear();
        addIArgument(lossReduce.ordinal()); //Ops: 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
    }

    public abstract String opName();

    @Override
    public int getNumOutputs(){
        return 3;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes.get(0).isFPType(), "Input 0 (predictions) must be a floating point type; inputs datatypes are %s for %s",
                inputDataTypes, getClass());
        DataType dt0 = inputDataTypes.get(0);
        DataType dt1 = arg(1).dataType();
        DataType dt2 = arg(2).dataType();
        if(!dt1.isFPType())
            dt1 = dt0;
        if(!dt2.isFPType())
            dt2 = dt0;
        return Arrays.asList(dt0, dt1, dt2);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }
}
