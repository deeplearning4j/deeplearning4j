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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

/**
 * BinCount: counts the number of times each value appears in an integer array.
 *
 * @author Alex Black
 */
public class BinCount extends DynamicCustomOp {

    private Integer minLength;
    private Integer maxLength;

    public BinCount(){ }

    public BinCount(SameDiff sd, SDVariable in, SDVariable weights, Integer minLength, Integer maxLength){
        super(sd, new SDVariable[]{in}, false);
        Preconditions.checkState((minLength == null) != (maxLength == null), "Cannot have only one of minLength and maxLength" +
                "non-null: both must be simultaneously null or non-null. minLength=%s, maxLength=%s", minLength, maxLength);
        this.minLength = minLength;
        this.maxLength = maxLength;
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
        System.out.println();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not supported");
    }
}
