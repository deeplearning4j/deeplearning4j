/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * MessageHandler implementation suited for ParallelWrapper running on single box
 *
 * PLEASE NOTE: This handler does NOT provide any network connectivity.
 *
 * @author raver119@gmail.com
 */
public class LocalHandler implements MessageHandler {
    protected transient GradientsAccumulator accumulator;

    public LocalHandler() {
        //
    }

    @Override
    public void initialize(@NonNull GradientsAccumulator accumulator) {
        this.accumulator = accumulator;
    }

    @Override
    public boolean broadcastUpdates(INDArray updates, int iterationNumber, int epochNumber) {
        // we just loop back data immediately
        accumulator.receiveUpdate(updates);

        updates.assign(0.0);

        Nd4j.getExecutioner().commit();

        return true;
    }
}
