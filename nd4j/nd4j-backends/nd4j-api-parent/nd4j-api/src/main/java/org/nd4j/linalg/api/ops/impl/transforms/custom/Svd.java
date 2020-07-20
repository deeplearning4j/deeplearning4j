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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.common.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * SVD - singular value decomposition
 *
 * @author Alex Black
 */
public class Svd extends DynamicCustomOp {
    public static final int DEFAULT_SWITCHNUM = 16;

    private boolean fullUV;
    private boolean computeUv;
    private int switchNum;

    public Svd(){ }

    public Svd(INDArray input, boolean full_matrices, INDArray s, INDArray u, INDArray v) {
        inputArguments.add(input);
        fullUV = full_matrices;
        computeUv = true;
        switchNum = DEFAULT_SWITCHNUM;


        outputArguments.add(s);
        outputArguments.add(u);
        outputArguments.add(v);

        addIArgument(ArrayUtil.fromBoolean(fullUV), ArrayUtil.fromBoolean(computeUv), switchNum);
    }

    public Svd(SameDiff sd, SDVariable input, boolean fullUV, boolean computeUv){
        this(sd, input, fullUV, computeUv, DEFAULT_SWITCHNUM);
    }

    public Svd(SameDiff sd, SDVariable input, boolean fullUV, boolean computeUv, int switchNum){
        super(sd, new SDVariable[]{input}, false);
        this.fullUV = fullUV;
        this.computeUv = computeUv;
        this.switchNum = switchNum;
        addIArgument(ArrayUtil.fromBoolean(fullUV), ArrayUtil.fromBoolean(computeUv), switchNum);
    }

    public Svd(INDArray input, boolean fullUV, boolean computeUV, int switchNum) {
        addInputArgument(input);
        addBArgument(fullUV, computeUV);
        addIArgument(switchNum);
    }

    @Override
    public String opName(){
        return "svd";
    }

    @Override
    public String tensorflowName() {
        return "Svd";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        this.fullUV = attributesForNode.get("full_matrices").getB();
        this.computeUv = attributesForNode.get("compute_uv").getB();
        this.switchNum = 16;
        addIArgument(ArrayUtil.fromBoolean(fullUV), ArrayUtil.fromBoolean(computeUv), switchNum);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int getNumOutputs(){
        return computeUv ? 3 : 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        if(computeUv){
            DataType d = dataTypes.get(0);
            return Arrays.asList(d, d, d);
        } else {
            return Collections.singletonList(dataTypes.get(0));
        }
    }
}
