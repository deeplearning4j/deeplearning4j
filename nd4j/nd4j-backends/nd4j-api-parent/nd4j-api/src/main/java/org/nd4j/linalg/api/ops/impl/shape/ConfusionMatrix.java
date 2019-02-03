/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import org.apache.commons.lang3.NotImplementedException;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 *
 */
public class ConfusionMatrix extends DynamicCustomOp {

    public ConfusionMatrix(){
    }

    public ConfusionMatrix(SameDiff sameDiff, SDVariable labels, SDVariable pred){
        super(null, sameDiff, new SDVariable[]{labels, pred});
    }

    public ConfusionMatrix(SameDiff sameDiff, SDVariable labels, SDVariable pred, SDVariable weights){
        super(null, sameDiff, new SDVariable[]{labels, pred, weights});
    }

    public ConfusionMatrix(SameDiff sameDiff, SDVariable labels, SDVariable pred, Integer numClasses){
        super(null, sameDiff, new SDVariable[]{labels, pred});
        addIArgument(numClasses);
    }

    public ConfusionMatrix(SameDiff sameDiff, SDVariable labels, SDVariable pred, Integer numClasses, SDVariable weights){
        super(null, sameDiff, new SDVariable[]{labels, pred, weights});
        addIArgument(numClasses);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        super.resolvePropertiesFromSameDiffBeforeExecution();
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
        //TODO make this configurable: https://github.com/deeplearning4j/deeplearning4j/issues/6854
        return Collections.singletonList(DataType.FLOAT);
    }
}
