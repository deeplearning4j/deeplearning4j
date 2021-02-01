/*
 *  ******************************************************************************
 *  *
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
import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Gather op
 */
@NoArgsConstructor
public class Gather extends DynamicCustomOp {

    protected int[] indices;
    protected int jaxis = 0;

    public Gather(SameDiff sameDiff, SDVariable df, SDVariable indices, int axis) {
        this(sameDiff, df, indices, axis, false);
    }

    public Gather(SameDiff sameDiff, SDVariable df, int[] indices, int axis) {
        this(sameDiff, df, indices, axis, false);
    }

    public Gather(SameDiff sameDiff, SDVariable input, int[] indices, int axis, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input}, inPlace);

        addIArgument(axis);
        addIArgument(indices);
        this.jaxis = axis;
        this.indices = indices;
    }

    public Gather(SameDiff sameDiff, SDVariable input, SDVariable indices, int axis, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input, indices}, inPlace);
        addIArgument(axis);
        this.jaxis = axis;
    }

    public Gather(INDArray df, int[] indexes, int axis) {
        addInputArgument(df);
        addIArgument(axis);
        addIArgument(indexes);
        this.jaxis = axis;
        this.indices = indices;
    }

    public Gather(INDArray df, INDArray indexes, int axis) {
        addInputArgument(df, indexes);
        addIArgument(axis);
        this.jaxis = axis;
        this.indices = indices;
    }

    @Override
    public String onnxName() {
        return "Gather";
    }


    @Override
    public String[] tensorflowNames() {
        return new String[]{"Gather", "GatherV2"};
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();
        val broadcast = PropertyMapping.builder()
                .onnxAttrName("indices")
                .tfInputPosition(1)
                .propertyNames(new String[]{"indices"}).build();

        map.put("indices", broadcast);

        ret.put(tensorflowNames()[0], map);
        ret.put(onnxName(), map);

        Map<String, PropertyMapping> map2 = new HashMap<>();
        val broadcast2 = PropertyMapping.builder()
                .tfInputPosition(1)
                .propertyNames(new String[]{"indices"}).build();
        map2.put("indices", broadcast2);

        val axis2 = PropertyMapping.builder()
                .tfInputPosition(2)
                .propertyNames(new String[]{"axis"}).build();
        map2.put("axis", axis2);

        ret.put("GatherV2", map2);


        return ret;
    }

    @Override
    public String opName() {
        return "gather";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v){
        //2 args: input and indices. Plus integer dimension arg
        //Gather backprop is just scatter add

        SDVariable indicesGrad = sameDiff.zerosLike(arg(1));
        SDVariable inputGrad = sameDiff.zerosLike(arg(0));

        SDVariable[] inputs = args();
        SDVariable axis;
        SDVariable rank = inputs[0].rank();
        if(inputs.length == 2){
            axis = sameDiff.constant(jaxis);
            if(jaxis < 0)
                axis = axis.add(rank);
        } else {
            axis = inputs[2];
        }

        //Use scatter add plus permute
        SDVariable dimsExAxis = sameDiff.range(null, sameDiff.constant(0), rank, sameDiff.constant(1), DataType.INT);
        SDVariable axisRank1 = axis.reshape(1);
        dimsExAxis = sameDiff.math().listDiff(dimsExAxis, axisRank1)[0];  //Don't need indices
        SDVariable permuteDims = sameDiff.concat(0, axisRank1, dimsExAxis);
        SDVariable invertDims = sameDiff.invertPermutation(permuteDims);


        //Permute gradients so original axis is at position 0... then scatter add, and reverse
        SDVariable gradAtOut = i_v.get(0);
        SDVariable permuteGrad = gradAtOut.permute(permuteDims);
        SDVariable inputGradPermute = inputGrad.permute(permuteDims);
        inputGrad = sameDiff.scatterAdd(inputGradPermute, arg(1), permuteGrad);

        //Now, invert the permutation so axis is back where it was
        inputGrad = inputGrad.permute(invertDims);

        return Arrays.asList(inputGrad, indicesGrad);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //Output type is same as (first) input type
        return Collections.singletonList(dataTypes.get(0));
    }
}
