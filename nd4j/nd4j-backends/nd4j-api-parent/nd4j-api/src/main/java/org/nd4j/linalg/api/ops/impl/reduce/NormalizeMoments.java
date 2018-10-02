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

import java.util.ArrayList;
import java.util.Map;

@NoArgsConstructor
public class NormalizeMoments extends DynamicCustomOp {

    private double shift = 0.0;  // reporting for duty

    public NormalizeMoments(SameDiff sameDiff, SDVariable counts, SDVariable means, SDVariable variances) {
        this(sameDiff, counts, means, variances, 0.0);
    }

    public NormalizeMoments(SameDiff sameDiff, SDVariable counts, SDVariable means, SDVariable variances, double shift) {
        super(null, sameDiff, new SDVariable[] {counts, means, variances}, false);
        this.shift = shift;
        addArgs();
    }

    public NormalizeMoments(INDArray counts, INDArray ssSum, INDArray ssSqSum, INDArray outMean, INDArray outVar) {
        super(null, new INDArray[]{counts, ssSum, ssSqSum}, new INDArray[]{outMean, outVar},
                new ArrayList<Double>(), new ArrayList<Integer>());

        addArgs();
    }

    private void addArgs() {
        addTArgument(shift);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    @Override
    public String opName() {
        return "normalize_moments";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "normalize_moments";
    }

}
