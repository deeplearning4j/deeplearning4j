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

import java.util.Map;


/**
 * Softmax cross entropy loss with Logits
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class SoftmaxCrossEntropyLossWithLogits extends DynamicCustomOp {

    public SoftmaxCrossEntropyLossWithLogits(SameDiff sameDiff, SDVariable logits, SDVariable weights, SDVariable labels,
                                             int reductionMode, double labelSmoothing) {
        super(null, sameDiff, new SDVariable[]{logits, weights, labels}, false);
    }

    public SoftmaxCrossEntropyLossWithLogits(SameDiff sameDiff, SDVariable logits, SDVariable weights, SDVariable labels,
                                             int reductionMode) {
        this(sameDiff, logits, weights, labels, reductionMode, 0.0);
    }


    public void addArgs() {
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
    }

    @Override
    public String opName() {
        return "softmax_cross_entropy_loss_with_logits";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "SoftmaxCrossEntropyWithLogits";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }
}
