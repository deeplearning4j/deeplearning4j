/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.audio;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

@NoArgsConstructor
public class ZeroCrossingRate extends DynamicCustomOp {
    private int frameLength = 2048;
    private int hopLength = 512;

    public ZeroCrossingRate(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                             int frameLength, int hopLength) {
        super(sameDiff, new SDVariable[]{input});
        this.frameLength = frameLength;
        this.hopLength = hopLength;
        addArgs();
    }

    public ZeroCrossingRate(@NonNull INDArray input, int frameLength, int hopLength) {
        super(new INDArray[]{input}, null);
        this.frameLength = frameLength;
        this.hopLength = hopLength;
        addArgs();
    }

    @Override
    public String opName() {
        return "zero_crossing_rate";
    }

    protected void addArgs() {
        addIArgument(frameLength);
        addIArgument(hopLength);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Gradient for zero_crossing_rate not implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 1,
                "Expected at least 1 input datatype for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
