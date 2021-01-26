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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 *
 */
public class ConfusionMatrix extends DynamicCustomOp {
    public static final DataType DEFAULT_DTYPE = DataType.INT;

    private DataType outputType = DEFAULT_DTYPE;

    public ConfusionMatrix(){
    }

    public ConfusionMatrix(@NonNull INDArray labels, @NonNull INDArray predicted, @NonNull DataType dataType){
        super(new INDArray[]{labels, predicted}, null);
        this.outputType = dataType;
        addDArgument(dataType);
    }

    public ConfusionMatrix(@NonNull INDArray labels, @NonNull INDArray predicted, int numClasses){
        this(labels, predicted, numClasses, DEFAULT_DTYPE);
    }

    public ConfusionMatrix(@NonNull INDArray labels, @NonNull INDArray predicted, INDArray weights) {
        this(labels, predicted, weights, null);
    }

    public ConfusionMatrix(@NonNull INDArray labels, @NonNull INDArray predicted, INDArray weights, Integer numClasses) {
        this(labels, predicted, weights, numClasses, DEFAULT_DTYPE);
    }

    public ConfusionMatrix(@NonNull INDArray labels, @NonNull INDArray predicted, Integer numClasses, @NonNull DataType dataType) {
        this(labels, predicted, null, numClasses, dataType);
    }

    public ConfusionMatrix(@NonNull INDArray labels, @NonNull INDArray predicted, INDArray weights, Integer numClasses, @NonNull DataType dataType) {
        super(wrapFilterNull(labels, predicted, weights), null);
        this.outputType = dataType;
        if(numClasses != null) {
            addIArgument(numClasses);
        }
        addDArgument(dataType);
    }


    public ConfusionMatrix(SameDiff sameDiff, SDVariable labels, SDVariable pred, SDVariable weights, DataType dataType){
        this(sameDiff, labels, pred, weights);
        this.outputType = dataType;
    }

    public ConfusionMatrix(SameDiff sameDiff, SDVariable labels, SDVariable pred, DataType dataType){
        super(null, sameDiff, new SDVariable[]{labels, pred});
        this.outputType = dataType;
        addDArgument(dataType);
    }

    public ConfusionMatrix(SameDiff sameDiff, SDVariable labels, SDVariable pred, SDVariable weights){
        super(null, sameDiff, new SDVariable[]{labels, pred, weights});
    }

    public ConfusionMatrix(SameDiff sameDiff, SDVariable labels, SDVariable pred, Integer numClasses){
        super(null, sameDiff, new SDVariable[]{labels, pred});
        addIArgument(numClasses);
    }

    public ConfusionMatrix(SameDiff sameDiff, SDVariable labels, SDVariable pred, SDVariable weights, Integer numClasses){
        super(null, sameDiff, new SDVariable[]{labels, pred, weights});
        addIArgument(numClasses);
    }

    public ConfusionMatrix(SameDiff sameDiff, SDVariable labels, SDVariable pred, Integer numClasses, SDVariable weights){
        super(null, sameDiff, new SDVariable[]{labels, pred, weights});
        if(numClasses != null) {
            addIArgument(numClasses);
        }
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
        //Looks like this is implemented in practice using a large collection of discrete ops - not single TF import op?
    }

    @Override
    public String opName() {
        return "confusion_matrix";
    }

    @Override
    public String tensorflowName() {
        return "ConfusionMatrix";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v){
        return Arrays.asList(sameDiff.zerosLike(arg(0)), sameDiff.zerosLike(arg(1)));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        return Collections.singletonList(outputType);
    }
}
