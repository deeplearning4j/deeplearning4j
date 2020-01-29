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

import java.util.Map;
import lombok.Getter;
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

/**
 * A simple recurrent unit cell.
 *
 * @author Adam Gibson
 */
public class SRUCell extends DynamicCustomOp {

    @Getter
    private SRUWeights weights;

    public SRUCell() {
    }

    public SRUCell(SameDiff sameDiff, SDVariable x, SDVariable cLast, SRUWeights weights) {
        super(null, sameDiff, weights.argsWithInputs(x, cLast));
        this.weights = weights;
    }



    public SRUCell(INDArray x, INDArray cLast, SRUWeights sruWeights) {
        super(null, null, sruWeights.argsWithInputs(x, cLast));
        this.weights = sruWeights;
    }


    @Override
    public String opName() {
        return "sruCell";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow name found for " + opName());
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
