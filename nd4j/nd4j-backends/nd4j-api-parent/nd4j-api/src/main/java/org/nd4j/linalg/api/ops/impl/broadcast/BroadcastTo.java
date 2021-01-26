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

package org.nd4j.linalg.api.ops.impl.broadcast;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * BroadcastTo op: given 2 input arrays, content X and shape Y, broadcast X to the shape specified by the content of Y.
 * Y should be a 1d vector
 *
 * @author Alex Black
 */
@NoArgsConstructor
public class BroadcastTo extends DynamicCustomOp {


    public BroadcastTo(SameDiff sameDiff, SDVariable input, SDVariable shape) {
        super(null, sameDiff, new SDVariable[] {input,shape}, false);
    }

    public BroadcastTo(@NonNull INDArray input, @NonNull long[] shape, @NonNull INDArray output){
        this(input, Nd4j.createFromArray(shape), output);
    }

    public BroadcastTo(@NonNull INDArray input, @NonNull INDArray shape, @NonNull INDArray output){
        super(null, new INDArray[]{input, shape}, new INDArray[]{output});
    }

    @Override
    public String opName() {
        return "broadcast_to";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);

    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "BroadcastTo";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient){
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected 2 input datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }
}
