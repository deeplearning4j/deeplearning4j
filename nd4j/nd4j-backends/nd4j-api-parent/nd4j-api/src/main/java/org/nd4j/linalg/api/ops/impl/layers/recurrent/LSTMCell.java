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

import lombok.NoArgsConstructor;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMCellConfiguration;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

/**
 * LSTM cell
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class LSTMCell extends DynamicCustomOp {

    private LSTMCellConfiguration configuration;

    public LSTMCell(SameDiff sameDiff, LSTMCellConfiguration configuration) {
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
        return "lstmCell";
    }


    @Override
    public String onnxName() {
        return  "LSTM";
    }

    @Override
    public String tensorflowName() {
        return super.tensorflowName();
    }
}
