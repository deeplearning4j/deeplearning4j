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
 * This operation creates a new, optionally nullified, array with a given shape, order and data type
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class Create extends DynamicCustomOp {

    protected boolean initialize = false;
    protected char order = 'c';
    protected DataType outputType = DataType.FLOAT;    //Allow customizing dtype for TF import

    public Create() {
    }

    public Create(String name, SameDiff sameDiff, SDVariable input, boolean initialize) {
        this(name, sameDiff, input, 'c', initialize, input.dataType());
    }

    public Create(String name, SameDiff sameDiff, SDVariable input, char order, boolean initialize, DataType dataType) {
        super(name, sameDiff, new SDVariable[]{input}, false);
        this.outputType = dataType;
        this.initialize = initialize;
        this.order = order;

        addArgs();
    }

    public Create(INDArray shape, DataType dataType) {
        this(shape, 'c', false, dataType);
    }

    public Create(INDArray shape, boolean initialize, DataType dataType) {
        this(shape, 'c', initialize, dataType);
    }

    public Create(@NonNull INDArray shape, char order, boolean initialize, DataType dataType) {
        super(new INDArray[]{shape}, new INDArray[0]);
        this.order = order;
        this.initialize = initialize;
        this.outputType = dataType;

        addArgs();
    }

    protected void addArgs() {
        addBArgument(initialize);
        addIArgument((int) order,outputType.toInt());
    }

    @Override
    public String opName() {
        return "create";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No op found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Empty";
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        // convert output data type
        if(attributesForNode.containsKey("dtype")) {
            outputType = TFGraphMapper.convertType(attributesForNode.get("dtype").getType());
        }

        // get init field
        if(attributesForNode.containsKey("init")) {
            initialize = attributesForNode.get("init").getB();
        }

        // there's no order in TF, just plain C
        this.order = 'c';
        addArgs();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = sameDiff.zerosLike(outputVariables()[0]);
        return Arrays.asList(ret);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes.size() == 1, "Expected list with exactly 1 datatype for %s, got %s", getClass(), dataTypes);
        if(outputType != null){
            return Collections.singletonList(outputType);
        } else {
            //Output type is same as input type
            return dataTypes;
        }
    }
}
