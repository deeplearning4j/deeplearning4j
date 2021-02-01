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

package org.nd4j.linalg.api.ops.impl.layers.recurrent;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.SRUWeights;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

/**
 * Simple recurrent unit
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class SRU extends DynamicCustomOp {

    @Getter
    private SRUWeights weights;

    @Getter
    private SDVariable mask;

    public SRU(@NonNull SameDiff sameDiff, @NonNull SDVariable x, @NonNull SDVariable initialC, SDVariable mask, @NonNull SRUWeights weights) {
        super(null, sameDiff, wrapFilterNull(x, weights.getWeights(), weights.getBias(), initialC, mask));
        this.mask = mask;
        this.weights = weights;
    }

    public SRU(INDArray x, INDArray initialC, INDArray mask, SRUWeights sruWeights) {
        super(wrapFilterNull(x, sruWeights.getIWeights(), sruWeights.getIBias(), initialC, mask), null);
        this.mask = (SDVariable) mask;
        this.weights = sruWeights;
    }

    public SRU(INDArray x, INDArray initialC, SRUWeights sruWeights) {
        super(wrapFilterNull(x, sruWeights.getIWeights(), sruWeights.getIBias(), initialC), null);
        this.weights = sruWeights;
    }

    @Override
    public String opName() {
        return "sru";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op name for " + opName());
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }
}
