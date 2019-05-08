/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.GRUCellConfiguration;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
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
        return "GRUBlockCell";
    }

    @Override
    public String[] onnxNames() {
        return super.onnxNames();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 6, "Expected exactly 6 inputs to GRUCell, got %s", inputDataTypes);
        //4 outputs, all of same type as input
        DataType dt = inputDataTypes.get(0);
        Preconditions.checkState(dt.isFPType(), "Input type 0 must be a floating point type, got %s", dt);
        return Arrays.asList(dt, dt, dt, dt);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads){
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
