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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class BinCount extends DynamicCustomOp {

    private Integer minLength;
    private Integer maxLength;
    private DataType outputType;

    public BinCount(){ }

    public BinCount(SameDiff sd, SDVariable in, SDVariable weights, Integer minLength, Integer maxLength, DataType outputType){
        super(sd, weights == null ? new SDVariable[]{in} : new SDVariable[]{in, weights}, false);
        Preconditions.checkState((minLength == null) != (maxLength == null), "Cannot have only one of minLength and maxLength" +
                "non-null: both must be simultaneously null or non-null. minLength=%s, maxLength=%s", minLength, maxLength);
        this.minLength = minLength;
        this.maxLength = maxLength;
        this.outputType = outputType;
        addArgs();
    }

    private void addArgs(){
        if(minLength != null)
            addIArgument(minLength);
        if(maxLength != null)
            addIArgument(maxLength);
    }

    @Override
    public String opName(){
        return "bincount";
    }

    @Override
    public String tensorflowName() {
        return "Bincount";
    }



    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(attributesForNode.containsKey("T")) {
            outputType = TFGraphMapper.convertType(attributesForNode.get("T").getType());
        }
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputTypes){
        Preconditions.checkState(inputTypes != null && (inputTypes.size() >= 1 && inputTypes.size() <= 4), "Expected 1 to 4 input types, got %s for op %s",
                inputTypes, getClass());

        //If weights present, same type as weights. Otherwise specified dtype
        if(inputTypes.size() == 2 || inputTypes.size() == 4) {
            //weights available case or TF import case (args 2/3 are min/max)
            return Collections.singletonList(inputTypes.get(1));
        } else {
            Preconditions.checkNotNull(outputType, "No output type available - output type must be set unless weights input is available");
            return Collections.singletonList(outputType);
        }
    }
}
