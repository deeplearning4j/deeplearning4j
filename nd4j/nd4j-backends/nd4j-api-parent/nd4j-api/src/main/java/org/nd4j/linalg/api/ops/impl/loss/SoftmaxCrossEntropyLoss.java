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

package org.nd4j.linalg.api.ops.impl.loss;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;


/**
 * Softmax cross entropy loss
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class SoftmaxCrossEntropyLoss extends BaseLoss {
    public static final double DEFAULT_LABEL_SMOOTHING = 0.0;

    private double labelSmoothing = 0.0;

    public SoftmaxCrossEntropyLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable logits, SDVariable weights, SDVariable labels,
                                   double labelSmoothing) {
        super(sameDiff, lossReduce, logits, weights, labels);
        this.labelSmoothing = labelSmoothing;

        addArgs();
    }

    public SoftmaxCrossEntropyLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable logits, SDVariable weights, SDVariable labels) {
        this(sameDiff, lossReduce, logits, weights, labels, 0.0);
    }

    public SoftmaxCrossEntropyLoss(INDArray labels, INDArray predictions, INDArray weights, LossReduce lossReduce, double labelSmoothing){
        super(lossReduce, predictions, weights, labels);
        this.labelSmoothing = labelSmoothing;
        addArgs();
    }

    public void addArgs() {
        super.addArgs();
        addTArgument(labelSmoothing);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    @Override
    public String opName() {
        return "softmax_cross_entropy_loss";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "SoftmaxCrossEntropy";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args are: predictions, weights, label
        SDVariable[] grads = f().lossSoftmaxCrossEntropyBp(arg(2), arg(0), arg(1), lossReduce, labelSmoothing);
        return Arrays.asList(grads);
    }
}
