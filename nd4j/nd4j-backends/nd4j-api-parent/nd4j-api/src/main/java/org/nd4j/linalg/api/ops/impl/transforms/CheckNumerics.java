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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * CheckNumerics op wrapper
 * @author raver119@gmail.com
 */
public class CheckNumerics extends DynamicCustomOp {

    public CheckNumerics(SameDiff sd, SDVariable input, SDVariable message){
        super(sd, new SDVariable[]{input, message});
    }

    public CheckNumerics(){ }

    @Override
    public String opName() {
        return "check_numerics";
    }

    @Override
    public String tensorflowName() {
        return "CheckNumerics";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(f1.get(0));
    }

    @Override
    public int numOutputArguments(){
        return 1;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        String str = attributesForNode.get("message").getS().toStringUtf8();
        //No "string args" support in libnd4j custom ops -> make it a constant instead
        String name = nodeDef.getName();
        SDVariable msg = initWith.constant(name + "/message", Nd4j.scalar(str));
        List<String> newInputs = new ArrayList<>(2);
        newInputs.addAll(initWith.getOps().get(name).getInputsToOp());
        newInputs.add(msg.name());
        initWith.getOps().get(name).setInputsToOp(newInputs);
        initWith.getVariables().get(msg.name()).setInputsForOp(Collections.singletonList(getOwnName()));    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        //input data types may be less than 2 for import, only first one matters anyways
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() <= 2, "Expected 2 datatype in, got %s", inputDataTypes);
        Preconditions.checkState(inputDataTypes.get(0).isFPType(), "Input datatype must be a floating point type, got %s", inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
