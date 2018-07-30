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

import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.GRUCellConfiguration;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;


/**
 * GRU cell for RNNs
 *
 *
 */
public class GRUCell extends DynamicCustomOp {

    private GRUCellConfiguration configuration;

    public GRUCell() {
    }

    public GRUCell(SameDiff sameDiff, GRUCellConfiguration configuration) {
        super(null, sameDiff, configuration.args());
        this.configuration = configuration;
        addIArgument(configuration.iArgs());
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        return configuration.toProperties();
    }

    @Override
    public String opName() {
        return "gruCell";
    }


    @Override
    public String onnxName() {
        return "GRU";
    }

    @Override
    public String tensorflowName() {
        return super.tensorflowName();
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public String[] onnxNames() {
        return super.onnxNames();
    }
}
