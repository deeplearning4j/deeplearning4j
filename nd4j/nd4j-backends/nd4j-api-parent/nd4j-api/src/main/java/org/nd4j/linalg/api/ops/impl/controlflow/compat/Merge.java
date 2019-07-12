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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.Op.Type;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class Merge extends BaseCompatOp {

    public Merge(SameDiff sd, SDVariable[] inputs){
        super(sd, inputs);
    }

    public Merge(){
        
    }

    /**
     * WARNING: do not change without changing serialization methods
     * See {@link org.nd4j.autodiff.samediff.serde.FlatBuffersMapper#getOpNum(String, Type)}
     *  and {@link org.nd4j.imports.converters.DifferentialFunctionClassHolder#customOpClassForHashAndName(long, String)}
     */
    public static final String OP_NAME = "merge";
    public static final int OP_NUM = 60;

    public Merge(SameDiff sd, SDVariable a, SDVariable b){
        this(sd, new SDVariable[]{a, b});
    }

    @Override
    public String opName() {
        return OP_NAME;
    }

    @Override
    public long opHash() {
        return 60L;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        if(arg().getArr() != null) {
            return Collections.singletonList(arg().getShapeDescriptor());
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
        return "Merge";
    }

    @Override
    public Op.Type opType() {
        return Type.LOGIC;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public int getNumOutputs(){
        return 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 2, "Expected 2 input dataypes for %s, got %s", getClass(), inputDataTypes);
        Preconditions.checkState(inputDataTypes.get(0) == inputDataTypes.get(1), "Input datatypes must be the same for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
