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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
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
 * Linspace op - with dynamic (SDVariable) args
 * @author Alex Black
 */
public class Linspace extends DynamicCustomOp {

    private DataType dataType;
    private double start;
    private double stop;
    private long elements;

    public Linspace(SameDiff sameDiff, DataType dataType, double start, double stop, long number) {
        this(sameDiff, sameDiff.constant(start), sameDiff.constant(stop), sameDiff.constant(number), dataType);
    }

    public Linspace(SameDiff sameDiff, SDVariable from, SDVariable to, SDVariable length, DataType dataType){
        super(sameDiff, new SDVariable[]{from, to, length});
        this.dataType = dataType;
        addDArgument(dataType);
    }

    public Linspace(DataType dataType, double start, double stop, long number) {
        this(start, stop, number, dataType);
    }

    public Linspace(DataType dataType, INDArray start, INDArray stop, INDArray number) {
        this(start, stop, number, dataType);
    }

    public Linspace(@NonNull INDArray start, @NonNull INDArray stop, @NonNull INDArray number, @NonNull DataType dataType) {
        super(new INDArray[]{start, stop, number}, null);
        this.dataType = dataType;
        addDArgument(dataType);
    }

    public Linspace(double start, double stop, long number, @NonNull DataType dataType) {
        super(new INDArray[]{}, null);
        this.dataType = dataType;
        addDArgument(dataType);

        this.start = start;
        this.stop = stop;
        this.elements = number;

        addTArgument(this.start, this.stop);
        addIArgument(elements);
    }

    public Linspace(){ }

    @Override
    public String opName(){
        return "lin_space";
    }

    @Override
    public int getNumOutputs(){
        return 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        return Collections.singletonList(dataType);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "LinSpace";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        dataType = TFGraphMapper.convertType(attributesForNode.get("T").getType());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Arrays.asList(sameDiff.zerosLike(arg(0)), sameDiff.zerosLike(arg(1)), sameDiff.zerosLike(arg(2)));
    }
}
