/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.linalg.api.ops.impl.loss;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.loss.bp.SparseSoftmaxCrossEntropyLossWithLogitsBp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


/**
 * Sparse softmax cross entropy loss with logits.
 * Applies softmax to the input, then calculates cross entropy loss. Labels should be in integer-index format,
 * not one-hot format
 *
 * @author Alex Black
 */
@NoArgsConstructor
public class SparseSoftmaxCrossEntropyLossWithLogits extends DynamicCustomOp {

    public SparseSoftmaxCrossEntropyLossWithLogits(@NonNull SameDiff sameDiff, @NonNull SDVariable logits, @NonNull SDVariable labels) {
        super(null, sameDiff, new SDVariable[]{labels, logits}, false);
    }

    public SparseSoftmaxCrossEntropyLossWithLogits(@NonNull INDArray logits, @NonNull INDArray labels){
        super(new INDArray[]{labels, logits}, null);
    }

    public void addArgs() {
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);

        //Switch order: TF uses [logits, labels]; libnd4j expects [labels, logits]
        SameDiffOp op = initWith.getOps().get(this.getOwnName());
        List<String> list = op.getInputsToOp();
        List<String> newList = Arrays.asList(list.get(1), list.get(0));
        op.setInputsToOp(newList);
    }

    @Override
    public String opName() {
        return "sparse_softmax_cross_entropy_loss_with_logits";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "SparseSoftmaxCrossEntropyWithLogits";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 2, "Expected 2 input datatypes for %s, got %s", getClass(), inputDataTypes);
        if(dArguments != null && !dArguments.isEmpty())
            return Arrays.asList(dArguments.get(0));
        return Collections.singletonList(inputDataTypes.get(1));    //Same as predictions (logits)
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //args: label, logits
        SDVariable labelsGrad = sameDiff.zerosLike(arg(0));
        SDVariable logitsGrad = new SparseSoftmaxCrossEntropyLossWithLogitsBp(sameDiff, arg(1), arg(0)).outputVariable();
        return Arrays.asList(labelsGrad, logitsGrad);
    }
}
