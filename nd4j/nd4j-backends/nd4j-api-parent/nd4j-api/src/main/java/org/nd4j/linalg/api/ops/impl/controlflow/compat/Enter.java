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

package org.nd4j.linalg.api.ops.impl.controlflow.compat;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.Op.Type;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

@Data
public class Enter extends BaseCompatOp {

    protected boolean isConstant;

    public Enter() {
        System.out.println();
    }

    public Enter(SameDiff sameDiff, SDVariable[] inputs){
        super(sameDiff, inputs);
    }

    public Enter(SameDiff sameDiff, String frameName, SDVariable input) {
        super(sameDiff, new SDVariable[]{input});
        this.frameName = frameName;
        isConstant = input.isConstant();
    }

    public Enter(SameDiff sameDiff, String frameName, SDVariable input, boolean isConstant) {
        super(sameDiff, new SDVariable[]{input});
        this.frameName = frameName;
        this.isConstant = isConstant;
    }

    /**
     * WARNING: do not change without changing serialization methods
     * See {@link org.nd4j.autodiff.samediff.serde.FlatBuffersMapper#getOpNum(String, Type)}
     *  and {@link org.nd4j.imports.converters.DifferentialFunctionClassHolder#customOpClassForHashAndName(long, String)}
     */
    public static final String OP_NAME = "enter";
    public static final int OP_NUM = 100;

    @Override
    public String opName() {
        return OP_NAME;
    }



    @Override
    public String tensorflowName() {
        return "Enter";
    }

    @Override
    public Type opType() {
        return Type.LOGIC;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
        isConstant = attributesForNode.get("is_constant").getB();
    }

    @Override
    public void configureFromArguments() {
        if(!bArguments.isEmpty()) {
            this.isConstant = bArguments.get(0);
        }
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("frameName")) {
            String frameName = getStringFromProperty("frameName",properties);
            this.frameName = frameName;
        }

        if(properties.containsKey("isConstant")) {
            Boolean isConstant = getBooleanFromProperty("isConstant",properties);
            this.isConstant = isConstant;
        }

    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return super.doDiff(f1);
    }

    @Override
    public int getNumOutputs(){
        return 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected 1 input datatype for %s, got %s", getClass(), inputDataTypes);
        return inputDataTypes;
    }
}
