
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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@NoArgsConstructor
public class RandomPoisson extends DynamicCustomOp {

    private DataType outputDataType = DataType.FLOAT;

    public RandomPoisson(@NonNull INDArray shape, @NonNull INDArray rate, int... seeds) {
        addInputArgument(shape, rate);
        addIArgument(seeds);
    }

    public RandomPoisson(@NonNull  INDArray shape, @NonNull INDArray rate) {
        this(shape, rate, 0,0);
    }

    public RandomPoisson(@NonNull SameDiff sameDiff, @NonNull SDVariable shape, @NonNull SDVariable rate, int... seeds) {
        super(null, sameDiff, new SDVariable[]{shape, rate});
        addIArgument(seeds);
    }

    @Override
    public String opName() {
        return "random_poisson";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"RandomPoisson","RandomPoissonV2"};
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
       //TODO: change op descriptor to have proper data type matching java
        if(attributesForNode.containsKey("dtype")) {
            outputDataType = DataTypeAdapter.dtypeConv(attributesForNode.get("dtype").getType());
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes.size() == 2, "Expected exactly 2 input datatypes for %s, got %s",
                getClass(), inputDataTypes.size());

        if(!dArguments.isEmpty())
            return Arrays.asList(dArguments.get(0));
        return Collections.singletonList(outputDataType);
    }
}
