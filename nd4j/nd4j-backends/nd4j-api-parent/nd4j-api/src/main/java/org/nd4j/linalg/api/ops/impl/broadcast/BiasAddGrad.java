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

package org.nd4j.linalg.api.ops.impl.broadcast;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

@NoArgsConstructor
public class BiasAddGrad extends DynamicCustomOp {
    protected boolean nchw = true;

    public BiasAddGrad(SameDiff sameDiff, SDVariable input, SDVariable bias, SDVariable gradient, boolean nchw) {
        super(null, sameDiff, new SDVariable[]{input, bias, gradient});
        this.nchw = nchw;
        addBArgument(nchw);
    }

    public BiasAddGrad(@NonNull INDArray input, @NonNull INDArray bias, @NonNull INDArray gradient, INDArray output){
        super(new INDArray[]{input, bias, gradient}, wrapOrNull(output));
    }

    public BiasAddGrad(@NonNull INDArray input, @NonNull INDArray bias, @NonNull INDArray gradient,
                       boolean nchw) {
        addInputArgument(input, bias, gradient);
        this.nchw = nchw;
        addBArgument(nchw);
    }

    public BiasAddGrad(@NonNull INDArray input, @NonNull INDArray bias, @NonNull INDArray gradient) {
        this(input, bias, gradient, false);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "biasadd_bp";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Differentiation not supported for op " + getClass().getSimpleName());
    }

    @Override
    public String onnxName() {
        return "BiasAddGrad";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected 3 input data types for %s, got %s", getClass(), inputDataTypes);
        return Arrays.asList(inputDataTypes.get(0), inputDataTypes.get(1));
    }
}
