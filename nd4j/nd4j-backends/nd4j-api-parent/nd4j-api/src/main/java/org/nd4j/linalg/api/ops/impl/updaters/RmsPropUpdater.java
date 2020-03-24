/*******************************************************************************
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

package org.nd4j.linalg.api.ops.impl.updaters;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

/**
 *
 * @author raver119@gmail.com
 */
public class RmsPropUpdater extends DynamicCustomOp {

    public RmsPropUpdater() {
        //
    }

    public RmsPropUpdater(@NonNull INDArray gradients, @NonNull INDArray state, double lr, double decay, double epsilon) {
        this(gradients, state, gradients, state, lr, decay, epsilon);
    }

    public RmsPropUpdater(@NonNull INDArray gradients, @NonNull INDArray state, @NonNull INDArray updates, @NonNull INDArray updatedState, double lr, double decay, double epsilon) {
        addInputArgument(gradients, state);
        addOutputArgument(updates, updatedState);
        addTArgument(lr, decay, epsilon);
    }

    @Override
    public String opName() {
        return "rms_prop_updater";
    }
}
