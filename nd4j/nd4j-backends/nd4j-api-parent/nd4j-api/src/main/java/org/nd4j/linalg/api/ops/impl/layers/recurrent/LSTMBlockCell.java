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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMBlockCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMCellConfiguration;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * LSTM Block cell
 *
 * @author Alex Black
 */
public class LSTMBlockCell extends DynamicCustomOp {

    private LSTMBlockCellConfiguration configuration;

    public LSTMBlockCell() {
    }

    public LSTMBlockCell(SameDiff sameDiff, LSTMBlockCellConfiguration configuration) {
        super(null, sameDiff, configuration.args());
        this.configuration = configuration;
        addIArgument(configuration.iArgs());
        addTArgument(configuration.tArgs());
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return configuration.toProperties();
    }

    @Override
    public String opName() {
        return "lstmBlockCell";
    }

    @Override
    public String tensorflowName() {
        return "LSTMBlockCell";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        configuration = LSTMBlockCellConfiguration.builder()
                .forgetBias(attributesForNode.get("forget_bias").getF())
                .clippingCellValue(attributesForNode.get("cell_clip").getF())
                .peepHole(attributesForNode.get("use_peephole").getB())
                .build();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 8, "Expected exactly 8 inputs to LSTMBlockCell, got %s", inputDataTypes);
        //7 outputs, all of same type as input
        DataType dt = inputDataTypes.get(0);
        Preconditions.checkState(dt.isFPType(), "Input type 0 must be a floating point type, got %s", dt);
        return Arrays.asList(dt, dt, dt, dt, dt, dt, dt);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads){
        throw new UnsupportedOperationException("Not yet implemented");
    }

}
