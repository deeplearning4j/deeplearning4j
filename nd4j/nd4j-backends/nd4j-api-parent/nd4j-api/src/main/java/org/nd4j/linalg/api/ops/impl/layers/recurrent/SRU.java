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

package org.nd4j.linalg.api.ops.impl.layers.recurrent;

import lombok.Getter;
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
public class SRU extends DynamicCustomOp {

    @Getter
    private SRUWeights weights;

    @Getter
    private INDArray ndarrayMask;
    private SDVariable mask;


    public SRU() { }

    public SRU(@NonNull SameDiff sameDiff, @NonNull SDVariable x, @NonNull SDVariable initialC, INDArray mask, @NonNull SRUWeights weights) {
        super(null, sameDiff, new INDArray[]{x, weights.getWeights(), weights.getBias(), initialC, mask});
        this.ndarrayMask = mask;
        this.weights = weights;
    }

    public SRU(@NonNull INDArray x,@NonNull INDArray initialC,@NonNull INDArray mask,@NonNull SRUWeights sruWeights) {
        super(null, null, new INDArray[]{x, sruWeights.getWeights(), sruWeights.getBias(), initialC, mask});
        this.ndarrayMask = mask;
        this.weights = sruWeights;
    }

    public SRU(@NonNull INDArray x,@NonNull INDArray initialC,@NonNull SRUWeights sruWeights) {
        super(null, null, new INDArray[]{x, sruWeights.getWeights(), sruWeights.getBias(), initialC});
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
