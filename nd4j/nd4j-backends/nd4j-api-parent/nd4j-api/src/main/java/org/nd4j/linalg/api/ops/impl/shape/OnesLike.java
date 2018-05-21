/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * OnesLike function - gives an output array with all values/entries being 1, with the same shape as the input.
 *
 * @author Alex Black
 */
@Slf4j
public class OnesLike extends DynamicCustomOp {


    public OnesLike() {
    }

    public OnesLike(String name, SameDiff sameDiff, SDVariable input) {
        this(name, sameDiff, input, false);
    }

    public OnesLike(String name, SameDiff sameDiff, SDVariable input, boolean inPlace) {
        super(name, sameDiff, new SDVariable[]{input}, inPlace);
    }


    @Override
    public String opName() {
        return "ones_as";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No op found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "OnesLike";
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = sameDiff.zerosLike(outputVariables()[0]);
        return Arrays.asList(ret);
    }

}
