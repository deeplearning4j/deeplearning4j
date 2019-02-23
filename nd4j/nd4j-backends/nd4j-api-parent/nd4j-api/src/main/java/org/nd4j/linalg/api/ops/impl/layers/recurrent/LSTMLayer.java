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

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMBlockCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.RnnDataFormat;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * LSTM layer implemented as a single operation.
 * Implementation of operation for LSTM layer with optional peep hole connections.
 * See lstmBlockCell for details. lstmBlockCell is used internally for computation.
 * This method expects as input (and returns as output) sequences in time major order: i.e., shape
 * [seqLength,batchSize,inOutSize] where inOutSize is either the input size or the output size, depending
 * on the array.
 *
 * @author Alex Black
 */
public class LSTMLayer extends DynamicCustomOp {

    private LSTMConfiguration configuration;

    public LSTMLayer() {
    }

    public LSTMLayer(@NonNull SameDiff sameDiff, @NonNull LSTMConfiguration configuration) {
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
        return "lstmBlock";
    }

    @Override
    public String tensorflowName() {
        return "BlockLSTM";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        configuration = LSTMConfiguration.builder()
                .forgetBias(attributesForNode.get("forget_bias").getF())
                .clippingCellValue(attributesForNode.get("cell_clip").getF())
                .peepHole(attributesForNode.get("use_peephole").getB())
                .dataFormat(RnnDataFormat.TNS)  //Always time major for TF BlockLSTM
                .build();
        addIArgument(configuration.iArgs());
        addTArgument(configuration.tArgs());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 9, "Expected exactly 9 inputs to LSTMBlock, got %s", inputDataTypes);
        //7 outputs, all of same type as input. Note that input 0 is max sequence length (int64), input 1 is actual input
        DataType dt = inputDataTypes.get(1);
        Preconditions.checkState(dt.isFPType(), "Input type 1 must be a floating point type, got %s", dt);
        return Arrays.asList(dt, dt, dt, dt, dt, dt, dt);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads){
        throw new UnsupportedOperationException("Not yet implemented");
    }

}
