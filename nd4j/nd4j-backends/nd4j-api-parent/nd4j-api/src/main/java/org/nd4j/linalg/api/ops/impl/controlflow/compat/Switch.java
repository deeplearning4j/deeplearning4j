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

package org.nd4j.linalg.api.ops.impl.controlflow.compat;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Switch op forwards input to one of two outputs based on the value of a predicate
 */
public class Switch extends BaseCompatOp {

    public Switch(SameDiff sameDiff, SDVariable input, SDVariable predicate){
        super(sameDiff, new SDVariable[]{input, predicate});
    }

    public Switch(){ }

    @Override
    public String opName() {
        return "switch";
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        if(args()[0].getArr() != null) {
            val arg0 = args()[0];
            val arr0 = arg0.getArr();
            val dtype = arr0.dataType();
            return Arrays.asList(LongShapeDescriptor.fromShape(arg0.getShape(), dtype),LongShapeDescriptor.fromShape(arg0.getShape(), dtype));
        }
        else
            return Collections.emptyList();
    }

    @Override
    public SDVariable[] outputVariables() {
        return super.outputVariables();
    }

    @Override
    public String tensorflowName() {
        return "Switch";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.IF;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public int getNumOutputs(){
        return 2;   //2 outputs - 2 branches
    }
}
