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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Created by farizrahman4u on 3/28/18.
 */
@NoArgsConstructor
public class SequenceMask extends DynamicCustomOp {
    public static final DataType DEFAULT_DTYPE = DataType.BOOL;

    private int maxLen;
    private boolean is_static_maxlen = false;
    private DataType dataType;

    public SequenceMask(SameDiff sameDiff, SDVariable input, SDVariable maxLen, DataType dataType) {
        super(null, sameDiff, new SDVariable[] {input, maxLen}, false);
        this.dataType = dataType;
        addDArgument(dataType);
    }

    public SequenceMask(SameDiff sameDiff, SDVariable input, int maxLen, DataType dataType) {
        super(null, sameDiff, new SDVariable[] {input}, false);
        this.maxLen = maxLen;
        this.is_static_maxlen = true;
        addIArgument(maxLen);
        this.dataType = dataType;
        addDArgument(dataType);
    }

    public SequenceMask(SameDiff sameDiff, SDVariable input, DataType dataType) {
        super(null, sameDiff, new SDVariable[] {input}, false);
        this.dataType = dataType;
        addDArgument(dataType);
    }

    public SequenceMask(INDArray input, int maxLen, DataType dataType) {
        addInputArgument(input);
        addIArgument(maxLen);
        this.dataType = dataType;
        addDArgument(dataType);
    }

    public SequenceMask(INDArray input, DataType dataType) {
        addInputArgument(input);
        this.dataType = dataType;
        addDArgument(dataType);
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val targetNode = TFGraphMapper.getNodeWithNameFromGraph(graph, nodeDef.getInput(1));
        val maxlen = TFGraphMapper.getNDArrayFromTensor(targetNode);
        if (maxlen == null){
            // No 2nd input
            this.is_static_maxlen = true;
        }
        TFGraphMapper.initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        if (is_static_maxlen) {
            addIArgument(this.maxLen);
        }

    }
    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> attrs = new LinkedHashMap<>();
        if (is_static_maxlen) {
            val maxLen = PropertyMapping.builder()
                    .propertyNames(new String[]{"maxLen"})
                    .tfAttrName("maxlen")
                    .build();
            attrs.put("maxLen", maxLen);
        }
        ret.put(tensorflowName(), attrs);
        return ret;
    }

    @Override
    public String opName() {
        return "sequence_mask";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //Input is integer indices
        return Collections.singletonList(f().zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        SDVariable[] args = args();
        Preconditions.checkState(dataTypes.size() == args.length, "Expected list with exactly %s datatypes for %s, got %s", args.length, getClass(), dataTypes);
        //Output type is same as input by default
        return Collections.singletonList(dataType == null ? DEFAULT_DTYPE : dataType);
    }
}
