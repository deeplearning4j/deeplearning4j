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

package org.nd4j.linalg.api.ops.impl.reduce;

import lombok.EqualsAndHashCode;
import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;

@EqualsAndHashCode
public class EinSum extends DynamicCustomOp {
    /**
     *
     * @param sameDiff
     * @param inputs
     * @param equation the einsum equation
     */
    public EinSum(SameDiff sameDiff,
                  SDVariable[] inputs,
                  String equation) {
        super(null,sameDiff,inputs);
        addSArgument(equation);
    }




    public EinSum(INDArray[] inputs, INDArray z, String equation) {
        super(null, inputs, z == null ? null : new INDArray[]{z});
        addSArgument(equation);
    }





    public EinSum() {
    }

    public EinSum(INDArray[] a, String equation) {
        this(a, null, equation);
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        return Collections.emptyMap();
    }

    @Override
    public void configureFromArguments() {

    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }

    @Override
    public String configFieldName() {
        return "mt";
    }

    public void setPropertiesForFunction(Map<String,Object> properties) {

    }

    /**
     * For a 2D matrix of shape (M, N) we return (N, M).
     * For a 3D matrix with leading mini-batch dimension (mb, M, N)
     * we return (mb, N, M)
     *
     * @param shape input shape array
     * @return
     */
    public long[] transposeShapeArray(long[] shape) {
        if (shape.length == 2) {
            return ArrayUtil.reverseCopy(shape);
        } else if (shape.length == 3) {
            return new long[] {shape[0], shape[2], shape[1]};
        } else {
            throw new IllegalArgumentException("Matrix input has to be of length 2 or 3, got: " + shape.length );
        }

    }

    @Override
    public String onnxName() {
        return "Einsum";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Einsum"};
    }



    @Override
    public String opName() {
        return "einsum";
    }



    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
    }





    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        throw new UnsupportedOperationException("Not yet implemented");
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();

        return ret;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        if(!dArguments.isEmpty())
            return Collections.singletonList(dArguments.get(0));

        return Collections.singletonList(dataTypes.get(0));
    }
}
