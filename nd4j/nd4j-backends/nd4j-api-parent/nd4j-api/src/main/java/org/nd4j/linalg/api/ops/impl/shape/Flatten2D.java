/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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

import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Flatten 2D function
 *
 * @author Adam Gibson
 */
@Slf4j
@NoArgsConstructor
public class Flatten2D extends DynamicCustomOp {

    private long flattenDimension;

    public Flatten2D(SameDiff sameDiff, SDVariable i_v, long axis) {
        super(null, sameDiff, new SDVariable[]{i_v});
        this.flattenDimension = axis;
        addIArgument(axis);
    }



    public Flatten2D(INDArray in, long axis) {
        super(new INDArray[]{in}, null);
        this.flattenDimension = axis;
        addIArgument(axis);
    }



    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();

        val axisMapping = PropertyMapping.builder()
                .onnxAttrName("axis")
                .propertyNames(new String[]{"axis"})
                .build();

        map.put("axis", axisMapping);

        ret.put(onnxName(), map);

        return ret;
    }


    @Override
    public String opName() {
        return "flatten_2d";
    }

    @Override
    public String onnxName() {
        return "Flatten";
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No op name found for tensorflow!");
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Arrays.asList(new Flatten2D(sameDiff,i_v.get(0),flattenDimension).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        if(!dArguments.isEmpty())
            return Collections.singletonList(dArguments.get(0));
        //Output type is always same as input type
        return Collections.singletonList(dataTypes.get(0));
    }

}
