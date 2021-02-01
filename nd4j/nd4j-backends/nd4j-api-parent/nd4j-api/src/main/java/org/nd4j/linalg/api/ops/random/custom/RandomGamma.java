
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
package org.nd4j.linalg.api.ops.random.custom;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.descriptors.properties.adapters.DataTypeAdapter;
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
public class RandomGamma extends DynamicCustomOp {

    public RandomGamma(@NonNull INDArray shape, @NonNull INDArray alpha, INDArray beta,
                       int... seeds) {
        if (beta != null) {
            addInputArgument(shape,alpha,beta);
        }
        addInputArgument(shape,alpha);
        addIArgument(seeds);
    }

    public RandomGamma(@NonNull INDArray shape, @NonNull INDArray alpha, INDArray beta) {

        this(shape,alpha,beta,0,0);
    }

    public RandomGamma(@NonNull SameDiff sameDiff, @NonNull SDVariable shape,
                       @NonNull SDVariable alpha, SDVariable beta, int... seeds) {
        super(null, sameDiff, beta != null ? new SDVariable[]{shape, alpha, beta} :
                                                      new SDVariable[]{shape, alpha});
        addIArgument(seeds);
    }

    @Override
    public String opName() {
        return "random_gamma";
    }

    @Override
    public String tensorflowName() {
        return "RandomGamma";
    }

    private DataType outputDataType = DataType.FLOAT;

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
            outputDataType = DataTypeAdapter.dtypeConv(attributesForNode.get("T").getType());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null, "Expected exactly input datatypes for %s, got null", getClass());
        return Collections.singletonList(outputDataType);
    }
}
