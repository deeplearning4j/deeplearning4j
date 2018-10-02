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

package org.nd4j.linalg.api.ops.impl.reduce;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@NoArgsConstructor
public class Moments extends DynamicCustomOp {

    private int[] axes;

    public Moments(SameDiff sameDiff, SDVariable input) {
        this(sameDiff, input, null);
    }

    public Moments(SameDiff sameDiff, SDVariable input, int[] axes) {
        super(null, sameDiff, new SDVariable[] {input}, false);
        this.axes = axes;
        addArgs();
    }

    public Moments(INDArray in, INDArray outMean, INDArray outStd, int... axes){
        super(null, new INDArray[]{in}, new INDArray[]{outMean, outStd}, null, axes);
    }

    private void addArgs() {
        if(axes != null) {
            for (int axis : axes) {
                addIArgument(axis);
            }
        }
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    @Override
    public String opName() {
        return "moments";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "moments";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        SDVariable dLdMean = grad.get(0);
        SDVariable dLdVar = grad.get(1);        //Note: non-bias-corrected variance
        SDVariable meanBp = f().meanBp(arg(), dLdMean, false, axes);
        SDVariable varBp = f().varianceBp(arg(), dLdVar, false, false, axes);
        return Collections.singletonList(meanBp.add(varBp));
    }

}
