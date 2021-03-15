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

package org.nd4j.linalg.api.ops.impl.shape.bp;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.*;

@Slf4j
public class SliceBp extends DynamicCustomOp {

    private int[] begin;
    private int[] size;

    public SliceBp() {}

    public SliceBp(SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable gradient, @NonNull int[] begin, @NonNull int[] size){
        super(null, sameDiff, new SDVariable[]{input, gradient});
        this.begin = begin;
        this.size = size;
        addIArgument(begin);
        addIArgument(size);
    }

    public SliceBp(SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable gradient, @NonNull SDVariable begin, @NonNull SDVariable size){
        super(null, sameDiff, new SDVariable[]{input, begin, size, gradient});
    }


    @Override
    public String opName() {
        return "slice_bp";
    }


    @Override
    public void assertValidForExecution() {
        if (numInputArguments() != 2 && numInputArguments() != 4) {
            throw new ND4JIllegalStateException("Num input arguments must be 2 or 4.");
        }
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Differentiation not supported for backprop op: " + getClass().getSimpleName());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes.size() == 2 || dataTypes.size() == 4, "Expected list with exactly 2 or 4 datatypes for %s, got %s", getClass(), dataTypes);
        //Output type is same as (original) input type
//        if(args().length == 1){
            //Static begin/size
        SDVariable[] args = args();
            return Collections.singletonList(arg().dataType());
//        } else {
//            //Dynamic begin/size
//            return Arrays.asList(arg(0).dataType(), arg(1).dataType(), arg(2).dataType());
//        }
    }
}
