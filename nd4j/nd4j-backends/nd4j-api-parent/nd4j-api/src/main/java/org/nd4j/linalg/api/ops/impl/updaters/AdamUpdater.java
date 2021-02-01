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

package org.nd4j.linalg.api.ops.impl.updaters;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class AdamUpdater extends DynamicCustomOp {

    public AdamUpdater() {
        //
    }

    public AdamUpdater(@NonNull INDArray gradients, @NonNull INDArray stateU, @NonNull INDArray stateM, double lr, double beta1, double beta2, double epsilon, int iteration) {
        this(gradients, stateU, stateM, gradients, stateU, stateM, lr, beta1, beta2, epsilon, iteration);
    }

    public AdamUpdater(@NonNull INDArray gradients, @NonNull INDArray stateU, @NonNull INDArray stateM, @NonNull INDArray updates, @NonNull INDArray updatedStateU, @NonNull INDArray updatedStateM, double lr, double beta1, double beta2, double epsilon, int iteration) {
        addInputArgument(gradients, stateU, stateM);
        addOutputArgument(updates, updatedStateU, updatedStateM);
        addTArgument(lr, beta1, beta2, epsilon);
        addIArgument(iteration);
    }

    @Override
    public String opName() {
        return "adam_updater";
    }
}
