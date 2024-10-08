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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

public class Stack extends DynamicCustomOp {
    protected int jaxis;

    public Stack() {
    }

    public Stack(INDArray[] inputs, INDArray output, int axis){
        super(null, inputs, output == null ? null : new INDArray[]{output}, null, (List<Long>)null);
        this.jaxis = axis;
        addArgs();
    }

    public Stack(INDArray[] input, int axis) {
        addInputArgument(input);
        this.jaxis = axis;
        addArgs();
    }

    public Stack(SameDiff sameDiff, SDVariable values, int axis) {
        this(sameDiff, new SDVariable[]{values}, axis);
    }

    public Stack(SameDiff sameDiff, SDVariable[] values, int axis) {
        super(null, sameDiff, values, false);
        this.jaxis = axis;
        addArgs();
    }

    public void addArgs() {
        addIArgument(jaxis);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "stack";
    }


    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Pack", "Stack"};
    }

    @Override
    public String opName() {
        return "stack";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        throw new UnsupportedOperationException("No analog found for onnx for " + opName());
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();

        val axisMapping = PropertyMapping.builder()
                .onnxAttrName("axis")
                .tfAttrName("axis")
                .propertyNames(new String[]{"jaxis"})
                .build();

        map.put("jaxis", axisMapping);

        for (val name : tensorflowNames())
            ret.put(name, map);

        return ret;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList(sameDiff.unstack(f1.get(0), jaxis, args().length));
    }


    @Override
    public void configureFromArguments() {
       if(!iArguments.isEmpty()) {
           this.jaxis = iArguments.get(0).intValue();
       }
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("dimensions")) {
            Long dimension = (Long) properties.get("dimensions");
            this.jaxis = dimension.intValue();
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        DataType first = dataTypes.get(0);
        for( int i = 1; i < dataTypes.size(); i++) {
            DataType dt = dataTypes.get(i);
            Preconditions.checkState(first == dt, "All inputs must have same datatype - got %s and %s for inputs 0 and %s respectively", first, dt, i);
        }
        //Output type is same as input types
        return Collections.singletonList(first);
    }
}
