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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Unstack op conversion
 *
 * @author raver119@gmail.com
 */
public class Unstack extends DynamicCustomOp {

    // TODO: libnd4j currently doesn't support "num", number of outputs is inferred.
    private int num = -1;
    private int jaxis;

    public Unstack() {
    }

    public Unstack(SameDiff sameDiff, SDVariable value, int axis) {
        super(null, sameDiff, new SDVariable[]{value}, false);
        this.jaxis = axis;
        if (value.getShape() != null){
            if (value.getShape()[axis] != -1){
                num = (int)value.getShape()[axis];
            }
        }
        if (num <= 0){
            throw new ND4JIllegalStateException("Unstack: Unable to infer number of outputs from input. Provide number of outputs explicitly.");
        }
        addArgs();
    }

    public Unstack(SameDiff sameDiff, SDVariable value, int axis, int num) {
        super(null, sameDiff, new SDVariable[]{value}, false);
        this.jaxis = axis;
        this.num = num;
        addArgs();
    }

    public Unstack(INDArray in, INDArray[] out, int axis){
        super(null, new INDArray[]{in}, out, null, (int[])null);
        this.jaxis = axis;
        addArgs();
    }

    public void addArgs() {
        addIArgument(jaxis);
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Unstack", "Unpack"};
    }

    @Override
    public String tensorflowName() {
        return "Unstack";
    }


    @Override
    public String opName() {
        return "unstack";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val attrAxis = nodeDef.getAttrOrThrow("axis");
        int axis = (int) attrAxis.getI();
        this.jaxis = axis;
        val attrNum = nodeDef.getAttrOrDefault("num", null);
        if(attrNum != null){
            this.num = (int) attrNum.getI();
        }
        addArgs();
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();

        val axisMapping = PropertyMapping.builder()
                .onnxAttrName("axis")
                .tfInputPosition(-1)
                .propertyNames(new String[]{"axis"})
                .build();

        map.put("axis", axisMapping);

        ret.put(tensorflowName(), map);

        return ret;
    }


    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        throw new UnsupportedOperationException("No analog found for onnx for " + opName());
    }

    @Override
    public int getNumOutputs(){
        return num;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.stack(jaxis, f1.toArray(new SDVariable[f1.size()])));
    }

    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        Preconditions.checkState(dataTypes.size() == 1, "Expected list with exactly 1 datatype, got %s", dataTypes);
        //Output types are same as input type - i.e., just unpack rank R array into N rank R-1 arrays
        List<DataType> out = new ArrayList<>();
        for( int i=0; i<num; i++ ){
            out.add(dataTypes.get(0));
        }
        return out;
    }

}
