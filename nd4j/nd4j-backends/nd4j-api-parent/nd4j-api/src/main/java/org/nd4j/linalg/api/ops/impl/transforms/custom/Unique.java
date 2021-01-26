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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

@Slf4j
public class Unique extends DynamicCustomOp {
    public static final DataType DEFAULT_IDX_DTYPE = DataType.INT;

    private DataType idxDataType;

    public Unique(){ }

    public Unique(SameDiff sd, SDVariable in){
        super(sd, new SDVariable[]{in}, false);
    }

    @Override
    public String opName(){
        return "unique";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Unique","UniqueV2"};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public int numOutputArguments(){
        return 2;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        idxDataType = TFGraphMapper.convertType(nodeDef.getAttrOrThrow("out_idx").getType());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() >= 1, "Expected exactly 1 or more input datatypes for %s, got %s", getClass(), dataTypes);
        if(dataTypes.size() > 1)
            log.warn("Using returning first data type of type " + dataTypes.get(0) + " for input");
        return Arrays.asList(dataTypes.get(0), (idxDataType == null ? DEFAULT_IDX_DTYPE : idxDataType));
    }
}
