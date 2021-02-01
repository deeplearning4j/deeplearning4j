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

import lombok.NonNull;
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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * OnesLike function - gives an output array with all values/entries being 1, with the same shape as the input.
 *
 * @author Alex Black
 */
@Slf4j
public class OnesLike extends DynamicCustomOp {

    protected DataType outputType;    //Allow customizing dtype for TF import

    public OnesLike() {
    }

    public OnesLike(SameDiff sameDiff, SDVariable input) {
        this(null, sameDiff, input);
    }

    public OnesLike(String name, SameDiff sameDiff, SDVariable input) {
        this(name, sameDiff, input, input.dataType());
    }

    public OnesLike(SameDiff sameDiff, SDVariable input, DataType dataType) {
        this(null, sameDiff, input, dataType);
    }

    public OnesLike(String name, SameDiff sameDiff, SDVariable input, DataType dataType) {
        super(name, sameDiff, new SDVariable[]{input}, false);
        this.outputType = dataType;
        addArgs();
    }

    public OnesLike(@NonNull INDArray input, DataType dataType) {
        this.addInputArgument(input);
        this.outputType = dataType;
        addArgs();
    }

    public OnesLike(@NonNull INDArray input) {
        this(input, input.dataType());
    }

    public void addArgs() {
        if (outputType != null)
            addDArgument(outputType);
    }


    @Override
    public String opName() {
        return "ones_as";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No op found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "OnesLike";
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(attributesForNode.containsKey("T")) {
            outputType = TFGraphMapper.convertType(attributesForNode.get("T").getType());
        }

        addArgs();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = sameDiff.zerosLike(outputVariables()[0]);
        return Arrays.asList(ret);
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
