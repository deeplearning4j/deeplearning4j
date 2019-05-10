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

package org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.nd4j.linalg.api.shape.Shape;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Addition operation for n operands, called "mergeadd" in libnd4j
 *
 * @author Max Pumperla
 */
public class MergeAddOp extends BaseDynamicTransformOp {

    public MergeAddOp() {}

    public MergeAddOp(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(sameDiff, args, inPlace);
    }

    public MergeAddOp(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    @Override
    public String opName() {
        return "mergeadd";
    }

    @Override
    public String onnxName() {
        return "mergeadd";
    }

    @Override
    public String[] tensorflowNames(){
        return new String[]{"add_n", "AccumulateNV2"};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable gradient = sameDiff.setupFunction(i_v.get(0));
        List<SDVariable> ret = new ArrayList<>();
        for (int i = 0; i < args().length; i++)
            ret.add(gradient);
        return ret;
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        DataType first = dataTypes.get(0);
        for( int i=1; i<dataTypes.size(); i++ ){
            Preconditions.checkState(first == dataTypes.get(i), "Expected all input datatypes to be the same: first input is %s, input %s is %s", f(), i, dataTypes.get(i));
        }
        return Collections.singletonList(first);
    }

}
