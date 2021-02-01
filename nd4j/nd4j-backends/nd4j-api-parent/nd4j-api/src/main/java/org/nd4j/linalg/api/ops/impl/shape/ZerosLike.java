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
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Reshape function
 *
 * @author Adam Gibson
 */
@Slf4j
@NoArgsConstructor
public class ZerosLike extends DynamicCustomOp {

    protected DataType outputType;    //Allow customizing dtype for TF import

    public ZerosLike(SameDiff sameDiff, SDVariable input) {
        this(null, sameDiff, input, false, input.dataType());
    }

    public ZerosLike(String name, SameDiff sameDiff, SDVariable input) {
        this(name, sameDiff, input, false, input.dataType());
    }

    public ZerosLike(String name, SameDiff sameDiff, SDVariable input, DataType dataType) {
        this(name, sameDiff, input, false, dataType);
    }

    public ZerosLike(String name, SameDiff sameDiff, SDVariable input, boolean inPlace) {
        this(name, sameDiff, input, inPlace, input.dataType());
    }

    public ZerosLike(String name, SameDiff sameDiff, SDVariable input, boolean inPlace, DataType dataType) {
        super(name, sameDiff, new SDVariable[]{input}, inPlace);
        addDArgument(dataType);
    }

    public ZerosLike(INDArray in, INDArray out){
        this(in, out, in.dataType());
    }

    public ZerosLike(INDArray in){
        addInputArgument(in);
    }

    public ZerosLike(INDArray in, INDArray out, DataType dataType) {
        super(null, in, out, null, null);
        if (dataType != null) {
            addDArgument(dataType);
        }
    }


    @Override
    public String opName() {
        return "zeroslike";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No op found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "ZerosLike";
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(attributesForNode.containsKey("T")) {
            outputType = TFGraphMapper.convertType(attributesForNode.get("T").getType());
        }
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = sameDiff.zerosLike(outputVariables()[0]);
        return Collections.singletonList(ret);
    }

    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        Preconditions.checkState(dataTypes.size() == 1, "Expected list with exactly 1 datatype for %s, got %s", getClass(), dataTypes);
        if(outputType != null){
            return Collections.singletonList(outputType);
        } else {
            //Output type is same as input type
            return dataTypes;
        }
    }

}
