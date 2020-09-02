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
package org.deeplearning4j.rl4j.agent.learning.update;

import lombok.Getter;
import lombok.NonNull;
import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.updater.INeuralNetUpdater;

import java.util.List;

/**
 * This implementation of {@link IUpdateRule} delegates the <i>features-labels</i> or <i>gradients</i> computations to
 * a {@link IUpdateAlgorithm}, and the networks update to a {@link INeuralNetUpdater}.
 *
 * @param <RESULT_TYPE> The type of result returned by the IUpdateAlgorithm
 * @param <EXPERIENCE_TYPE> The type of experience
 */
public class UpdateRule<RESULT_TYPE, EXPERIENCE_TYPE> implements IUpdateRule<EXPERIENCE_TYPE> {

    private final INeuralNetUpdater updater;

    private final IUpdateAlgorithm<RESULT_TYPE, EXPERIENCE_TYPE> updateAlgorithm;

    @Getter
    private int updateCount = 0;

    public UpdateRule(@NonNull IUpdateAlgorithm<RESULT_TYPE, EXPERIENCE_TYPE> updateAlgorithm,
                      @NonNull INeuralNetUpdater<RESULT_TYPE> updater) {
        this.updateAlgorithm = updateAlgorithm;
        this.updater = updater;
    }

    @Override
    public void update(List<EXPERIENCE_TYPE> trainingBatch) {
        RESULT_TYPE featuresLabels = updateAlgorithm.compute(trainingBatch);
        updater.update(featuresLabels);
        ++updateCount;
    }

    @Override
    public void notifyNewBatchStarted() {
        updater.synchronizeCurrent();
    }

}
