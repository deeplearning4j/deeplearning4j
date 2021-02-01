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

package org.nd4j.linalg.api.ops.impl.reduce;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

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

    public NormalizeMoments(INDArray counts, INDArray means, INDArray variances, double shift) {
        super(null, new INDArray[]{counts, means, variances}, null);
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
        TFGraphMapper.initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    @Override
    public String opName() {
        return "normalize_moments";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected 3 input datatypes for %s, got %s", getClass(), inputDataTypes);
        //Count, mean_ss, variance_ss
        if(inputDataTypes.get(1).isFPType())
            return Arrays.asList(inputDataTypes.get(0), inputDataTypes.get(0));
        return Arrays.asList(Nd4j.defaultFloatingPointType(), Nd4j.defaultFloatingPointType());
    }

}
