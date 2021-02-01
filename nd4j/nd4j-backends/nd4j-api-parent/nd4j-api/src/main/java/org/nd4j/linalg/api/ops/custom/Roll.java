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

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class Roll extends DynamicCustomOp {

    public Roll() {}

    public Roll(@NonNull INDArray input, @NonNull INDArray axes, @NonNull INDArray shifts) {
        Preconditions.checkArgument(axes.rank() == shifts.rank(), "Roll: shifts and axes should be the same rank");
        Preconditions.checkArgument(axes.length() == shifts.length(), "Roll: shifts and axes should be the same length");
        addInputArgument(input, axes, shifts);
    }

    public Roll(@NonNull INDArray input, int shift) {
        addInputArgument(input);
        addIArgument(shift);
    }

    public Roll(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable shift) {
        super("", sameDiff, new SDVariable[]{input,shift});
    }

    public Roll(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable axes, @NonNull SDVariable shift) {
        super("", sameDiff, new SDVariable[]{input,axes,shift});
    }

    public Roll(@NonNull SameDiff sameDiff, @NonNull SDVariable input, int shift) {
        super("", sameDiff, new SDVariable[]{input});
        addIArgument(shift);
    }

    @Override
    public String opName() {
        return "roll";
    }

    @Override
    public String tensorflowName() {
        return "Roll";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
