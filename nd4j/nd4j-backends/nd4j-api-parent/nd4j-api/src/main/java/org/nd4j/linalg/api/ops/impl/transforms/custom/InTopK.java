/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

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
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //3rd input: dynamic K value
        Preconditions.checkState(dataTypes != null && !dataTypes.isEmpty(), "Expected at  least 1 input data types. for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(DataType.BOOL);
    }
}
