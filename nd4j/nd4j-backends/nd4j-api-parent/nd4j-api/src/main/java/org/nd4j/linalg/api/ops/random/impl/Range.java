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

package org.nd4j.linalg.api.ops.random.impl;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Range Op implementation, generates from..to distribution within Z
 *
 * @author raver119@gmail.com
 */
public class Range extends DynamicCustomOp {
    public static final DataType DEFAULT_DTYPE = DataType.FLOAT;

    private Double from;
    private Double to;
    private Double delta;
    private DataType dataType;

    public Range() {
        // no-op
    }

    public Range(SameDiff sd, double from, double to, double step, DataType dataType){
        super(null, sd, new SDVariable[0]);
        addTArgument(from, to, step);
        addDArgument(dataType);
        this.from = from;
        this.to = to;
        this.delta = step;
        this.dataType = dataType;
    }

    public Range(double from, double to, double step, DataType dataType){
        addTArgument(from, to, step);
        this.from = from;
        this.to = to;
        this.delta = step;
        this.dataType = dataType;
        addDArgument(dataType);
    }

    public Range(SameDiff sd, SDVariable from, SDVariable to, SDVariable step, DataType dataType){
        super(null, sd, new SDVariable[]{from, to, step});
        this.dataType = dataType;
        addDArgument(dataType);
    }


    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "range";
    }

    @Override
    public String onnxName() {
        return "Range";
    }

    @Override
    public String tensorflowName() {
        return "Range";
    }



    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
        if(attributesForNode.containsKey("Tidx")){
            dataType = TFGraphMapper.convertType(attributesForNode.get("Tidx").getType());
        }
        addDArgument(dataType);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.emptyList();
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes == null || inputDataTypes.isEmpty() || inputDataTypes.size() == 3,
                "Expected no input datatypes (no args) or 3 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(dataType == null ? DEFAULT_DTYPE : dataType);
    }
}
