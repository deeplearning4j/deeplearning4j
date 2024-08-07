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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
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

    public Create(SameDiff sd, SDVariable shape, DataType dataType) {
        super(sd,new SDVariable[]{shape});
        addDArgument(dataType);
        addBArgument(false);
        addIArgument('c', dataType.toInt());
        this.outputType = dataType;
    }

    public Create(SameDiff sd, SDVariable shape, DataType dataType, String order, boolean initialize) {
        this(sd,shape,dataType);
        addIArgument(order.charAt(0),dataType.toInt());
        addBArgument(initialize);
        this.outputType = dataType;
    }

    public Create(INDArray shape, DataType dataType, String order, boolean initialize) {
        super(new INDArray[]{shape},null);
        addBArgument(initialize);
        addIArgument(order.charAt(0),dataType.toInt());
    }

    protected void addArgs() {
        addBArgument(initialize);
        addIArgument(order,outputType.toInt());
    }

    @Override
    public void configureFromArguments() {
        if(!iArguments.isEmpty()) {
            this.outputType = DataType.fromInt(iArguments.size() > 1 ? iArguments.get(1).intValue(): iArguments.get(0).intValue());
        }
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
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

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
