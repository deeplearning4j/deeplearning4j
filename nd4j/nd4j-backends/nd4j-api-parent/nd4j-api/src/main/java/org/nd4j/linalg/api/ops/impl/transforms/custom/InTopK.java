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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * In Top K op
 *
 * @author Alex Black
 */
public class InTopK extends DynamicCustomOp {

    private boolean sorted;
    private int k;

    public InTopK(){ }

    public InTopK(SameDiff sd, SDVariable predictions, SDVariable targets, int k){
        super(sd, new SDVariable[]{predictions, targets}, false);
        this.k = k;
        addIArgument(k);
    }

    @Override
    public String opName(){
        return "in_top_k";
    }

    @Override
    public String tensorflowName() {
        return "InTopKV2";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

        String thisName = nodeDef.getName();
        String inputName = thisName + "/k";
        NodeDef kNode = null;
        for(int i = 0; i < graph.getNodeCount(); i++) {
            if(graph.getNode(i).getName().equals(inputName)){
                kNode = graph.getNode(i);
                break;
            }
        }
        Preconditions.checkState(kNode != null, "Could not find 'k' parameter node for op: %s", thisName);

        INDArray arr = TFGraphMapper.getInstance().getNDArrayFromTensor(inputName, kNode, graph);
        this.k = arr.getInt(0);
        addIArgument(k);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes, got %s", dataTypes);
        Preconditions.checkState(dataTypes.get(0) == dataTypes.get(1), "Input datatypes must be same type: got %s", dataTypes);
        return Collections.singletonList(DataType.BOOL);
    }
}
