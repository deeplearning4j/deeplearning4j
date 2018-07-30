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

package org.nd4j.linalg.api.ops.random.custom;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Uniform distribution wrapper
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DistributionUniform extends DynamicCustomOp {
    private double min = 0.0;
    private double max = 1.0;

    public DistributionUniform() {
        //
    }

    public DistributionUniform(SameDiff sd, SDVariable shape, double min, double max){
        super(null, sd, new SDVariable[]{shape});
        Preconditions.checkState(min <= max, "Minimum (%s) must be <= max (%s)", min, max);
        addTArgument(min, max);
    }

    public DistributionUniform(INDArray shape, INDArray out, double min, double max){
        super(null, new INDArray[]{shape}, new INDArray[]{out}, Arrays.asList(min, max), (List<Integer>)null);
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
        addArgs();
    }

    protected void addArgs() {
        addTArgument(min, max);
    }

    @Override
    public String opName() {
        return "randomuniform";
    }

    @Override
    public String tensorflowName() {
        return "RandomUniform";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }
}
