/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.transforms.arithmetic;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;

import java.util.ArrayList;
import java.util.List;

/**
 * Truncated division operation
 *
 * @author Adam Gibson
 */
public class FloorDivOp extends BaseDynamicTransformOp {
    public FloorDivOp() {}

    public FloorDivOp(SameDiff sameDiff, SDVariable x, SDVariable y){
        this(sameDiff, new SDVariable[]{x,y}, false);
    }

    public FloorDivOp( SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(sameDiff, args, inPlace);
    }

    public FloorDivOp( INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }


    @Override
    public String opName() {
        return "floordiv";
    }

    @Override
    public String onnxName() {
        return "FloorDiv";
    }

    @Override
    public String tensorflowName() {
        return "FloorDiv";
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return f().floorDivBp(larg(), rarg(), i_v.get(0));
    }
}
