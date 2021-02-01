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

/**
 *
 * @author raver119@gmail.com
 */
public class SgdUpdater extends DynamicCustomOp {

    public SgdUpdater() {
        //
    }

    public SgdUpdater(@NonNull INDArray input, double lr) {
        this(input, input, lr);
    }

    public SgdUpdater(@NonNull INDArray input, @NonNull INDArray output, double lr) {
        addInputArgument(input);
        addOutputArgument(output);
        addTArgument(lr);
    }

    @Override
    public String opName() {
        return "sgd_updater";
    }
}
