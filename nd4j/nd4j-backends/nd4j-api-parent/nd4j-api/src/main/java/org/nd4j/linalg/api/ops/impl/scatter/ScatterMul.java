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

package org.nd4j.linalg.api.ops.impl.scatter;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;


public class ScatterMul extends DynamicCustomOp {

    public ScatterMul(SameDiff sameDiff, SDVariable ref, SDVariable indices, SDVariable updates) {
        super(null, sameDiff, new SDVariable[]{ref, indices, updates}, false);
    }

    public ScatterMul() {}

    public ScatterMul(@NonNull INDArray ref, @NonNull INDArray indices, @NonNull INDArray update){
        super(new INDArray[]{ref, indices, update}, null);
    }

    @Override
    public String opName() {
        return "scatter_mul";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "ScatterMul";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradOut){
        //3 args: ref, indices, updates
        //out = ref times every update at its index, so dL/dref = dL/dOut times those updates (a scatterMul of the
        //gradient), and dL/du = dL/dOut * ref * the OTHER updates at u's index: ref alone only when no index repeats.
        //That product is (ref * the non-zero updates there) / u when u != 0 and no other update there is 0, the same
        //product itself when u is the one zero update there, and 0 otherwise, so no zero is ever divided by.

        SDVariable ref = arg(0);
        SDVariable indices = arg(1);
        SDVariable updates = arg(2);
        SDVariable grad = gradOut.get(0);

        List<SDVariable> ret = new ArrayList<>(3);
        SDVariable gradRef = sameDiff.scatterMul(grad, indices, updates);
        ret.add(gradRef);            //Reference array
        ret.add(sameDiff.zerosLike(arg(1)));  //Indices

        SDVariable isZero = updates.eq(0.0).castTo(updates.dataType());
        SDVariable nonZero = updates.add(isZero);   //each zero update replaced by 1
        SDVariable nonZeroProduct = sameDiff.scatterMul(ref, indices, nonZero);
        SDVariable zeroCount = sameDiff.scatterAdd(sameDiff.zerosLike(ref), indices, isZero);
        //1 where every OTHER update at the index is non-zero: the zero count there is u's own
        SDVariable othersNonZero = sameDiff.gather(zeroCount, indices, 0).eq(isZero).castTo(updates.dataType());
        SDVariable updateGrad = sameDiff.gather(grad, indices, 0)
                .mul(sameDiff.gather(nonZeroProduct, indices, 0))
                .div(nonZero)
                .mul(othersNonZero);
        ret.add(updateGrad);

        return ret;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected exactly 3 input datatypes for %s, got %s", getClass(), inputDataTypes);
        Preconditions.checkState(inputDataTypes.get(0) == inputDataTypes.get(2), "Reference (input 0) and updates (input 2) must have exactly same data types, got %s and %s",
                inputDataTypes.get(0), inputDataTypes.get(2));
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
