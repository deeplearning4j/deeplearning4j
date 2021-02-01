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
package org.nd4j.linalg.api.ops.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@NoArgsConstructor
public class LinearSolve extends DynamicCustomOp {

    public LinearSolve(INDArray a, INDArray b, boolean adjoint) {
        addInputArgument(a, b);
        addBArgument(adjoint);
    }

    public LinearSolve(INDArray a, INDArray b) {
        this(a,b,false);
    }

    public LinearSolve(SameDiff sameDiff, SDVariable a, SDVariable b, SDVariable adjoint) {
        super(sameDiff, new SDVariable[] {a, b, adjoint});
    }

    public LinearSolve(SameDiff sameDiff, SDVariable a, SDVariable b, boolean adjoint) {
        super(sameDiff, new SDVariable[] {a, b});
        addBArgument(adjoint);
    }

    @Override
    public String opName() {
        return "solve";
    }

    @Override
    public String tensorflowName() {
        return "MatrixSolve";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        boolean adjoint = attributesForNode.containsKey("adjoint") ? attributesForNode.get("adjoint").getB() : false;
        addBArgument(adjoint);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        int n = args().length;
        Preconditions.checkState(dataTypes != null && dataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }
}
