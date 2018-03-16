/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.accum;

import lombok.NoArgsConstructor;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;

import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


/**
 * Sigmoid cross entropy loss with logits
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class SigmoidCrossEntropyLoss extends DynamicCustomOp {

    private int reductionMode = 0;
    private double labelSmoothing = 0.0;

    public SigmoidCrossEntropyLoss(SameDiff sameDiff, SDVariable logits, SDVariable weights, SDVariable labels,
                                   int reductionMode, double labelSmoothing) {
        super(null, sameDiff, new SDVariable[]{logits, weights, labels}, false);
        this.reductionMode = reductionMode;
        this.labelSmoothing = labelSmoothing;
        this.sameDiff = sameDiff;

        addArgs();
    }

    public SigmoidCrossEntropyLoss(SameDiff sameDiff, SDVariable logits, SDVariable weights, SDVariable labels,
                            int reductionMode) {
        this(sameDiff, logits, weights, labels, reductionMode, 0.0);
    }


    public void addArgs() {
        addIArgument(reductionMode);
        addTArgument(labelSmoothing);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    @Override
    public String opName() {
        return "sigm_cross_entropy_loss";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "sigmoid_cross_entropy";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }
}
