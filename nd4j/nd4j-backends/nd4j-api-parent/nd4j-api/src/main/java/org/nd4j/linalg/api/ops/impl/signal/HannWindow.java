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

package org.nd4j.linalg.api.ops.impl.signal;

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

/**
 * Hann window function generator.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@NoArgsConstructor
public class HannWindow extends DynamicCustomOp {
    private boolean periodic = true;

    public HannWindow(@NonNull SameDiff sameDiff, @NonNull SDVariable size, boolean periodic) {
        super(sameDiff, new SDVariable[]{size});
        this.periodic = periodic;
        addArgs();
    }

    public HannWindow(@NonNull SameDiff sameDiff, @NonNull SDVariable size) {
        this(sameDiff, size, true);
    }

    public HannWindow(@NonNull INDArray size, boolean periodic, INDArray output) {
        super(new INDArray[]{size}, wrapOrNull(output));
        this.periodic = periodic;
        addArgs();
    }

    public HannWindow(@NonNull INDArray size, boolean periodic) {
        this(size, periodic, null);
    }

    @Override
    public String opName() {
        return "hann_window";
    }

    protected void addArgs() {
        addIArgument(periodic ? 1 : 0);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Gradient for HannWindow not implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(DataType.FLOAT);
    }
}
