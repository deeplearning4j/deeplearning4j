/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.nd4j.linalg.api.ops.custom;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

@NoArgsConstructor
public class Lstsq extends DynamicCustomOp {

    public Lstsq(@NonNull INDArray matrix, @NonNull INDArray rhs, double l2_regularizer, boolean fast) {
        addInputArgument(matrix, rhs);
        addTArgument(l2_regularizer);
        addBArgument(fast);
    }

    public Lstsq(@NonNull INDArray matrix, @NonNull INDArray rhs) {
        this(matrix, rhs, 0.0, true);
    }

    @Override
    public String opName() {
        return "lstsq";
    }
}
