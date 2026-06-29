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
import org.nd4j.common.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class TopK extends DynamicCustomOp {

    private boolean sorted;
    private int k;

    public TopK(){ }

    public TopK(SameDiff sd, SDVariable in, int k, boolean sorted) {
        super(sd, new SDVariable[]{in}, false);
        this.k = k;
        this.sorted = sorted;
        // Native top_k contract: k = INT_ARG(0), sorted = B_ARG(0). (The old encoding put sorted in
        // INT_ARG(0), so the native op read k=0 whenever the op was serialized to its args — e.g. via
        // the DynamicShapePlan path — even though the non-DSP path happened to use the this.k field.)
        addIArgument(k);
        addBArgument(sorted);
    }


    public TopK(INDArray input, double k, boolean sorted) {
        super(null,new INDArray[]{input},null);
        this.k = (int) k;
        this.sorted = sorted;
        // Native top_k contract: k = INT_ARG(0), sorted = B_ARG(0). (See note above.)
        addIArgument(this.k);
        addBArgument(sorted);
    }

    public TopK(SameDiff sd, SDVariable input, double k, boolean sorted) {
        this(sd,input,(int) k,sorted);
    }

    @Override
    public String opName(){
        return "top_k";
    }

    @Override
    public String tensorflowName() {
        return "TopKV2";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // Forward: TopK(x) -> (values[...,k], indices[...,k])
        // i_v.get(0) = gradient wrt values output, shape [..., k]
        // i_v.get(1) = gradient wrt indices (integers, no gradient needed)
        SDVariable x = arg(0);                           // shape [..., n]
        SDVariable gradValues = i_v.get(0);              // shape [..., k]
        SDVariable topkIndices = outputVariables()[1];   // shape [..., k], INT type

        // Need n = last dimension of x to build the one-hot matrix
        long[] xShape = x.getShape();
        Preconditions.checkState(xShape != null && xShape[xShape.length - 1] > 0,
                "TopK doDiff requires a statically known last dimension of the input, got shape: %s",
                Arrays.toString(xShape));
        int n = (int) xShape[xShape.length - 1];

        // Build one-hot matrix: indices [..., k] -> [..., k, n]  (axis=-1 appends depth dim at end)
        SDVariable oneHotMatrix = sameDiff.oneHot(topkIndices, n, -1, 1.0, 0.0, x.dataType());

        // Expand grad for broadcast: [..., k] -> [..., k, 1]
        SDVariable gradExpanded = sameDiff.expandDims(gradValues, -1);

        // Element-wise multiply: [..., k, n] * [..., k, 1] = [..., k, n]
        SDVariable weighted = oneHotMatrix.mul(gradExpanded);

        // Sum over the k axis (second-to-last = -2): [..., k, n] -> [..., n]
        SDVariable gradX = sameDiff.sum(weighted, -2);

        return Collections.singletonList(gradX);
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("sorted")) {
            this.sorted = getBooleanFromProperty("sorted",properties);
        }

        if(properties.containsKey("k")) {
            this.k = getIntValueFromProperty("k",properties);
        }

    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // 2 outputs: values (same dtype as input) and indices.
        // The native top_k C++ op writes int64 (LONG) indices regardless of the
        // Java-declared type. Pre-allocating INT32 (4 bytes/element) causes a
        // write-overflow for any 2D+ input because the second row's int64 indices
        // land beyond the allocated buffer, corrupting heap memory and producing
        // garbage VALUES for rows after the first. Use LONG to match native output.
        return Arrays.asList(dataTypes.get(0), DataType.LONG);
    }
}
