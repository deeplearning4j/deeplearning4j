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

package org.deeplearning4j.parallelism.factory;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.parallelism.trainer.DefaultTrainer;
import org.deeplearning4j.parallelism.trainer.Trainer;

/**
 * Creates {@link DefaultTrainer}
 * instances for use with {@link ParallelWrapper}
 * @author Adam Gibson
 */
public class DefaultTrainerContext implements TrainerContext {
    /**
     * Initialize the context
     *
     * @param model
     * @param args the arguments to initialize with (maybe null)
     */
    @Override
    public void init(Model model, Object... args) {

    }

    /**
     * Create a {@link Trainer}
     * based on the given parameters
     *
     * @param threadId   the thread id to use for this worker
     * @param model      the model to start the trainer with
     * @param rootDevice the root device id
     * @param useMDS     whether to use the {@link MagicQueue}
     *                   or not
     * @param wrapper    the wrapper instance to use with this trainer (this refernece is needed
     *                   for coordination with the {@link ParallelWrapper} 's {@link TrainingListener}
     * @return the created training instance
     */
    @Override
    public Trainer create(String uuid, int threadId, Model model, int rootDevice, boolean useMDS, ParallelWrapper wrapper,
                    WorkspaceMode mode, int averagingFrequency) {

        DefaultTrainer trainer = DefaultTrainer.builder().originalModel(model).replicatedModel(model).threadId(threadId)
                        .parallelWrapper(wrapper).workspaceMode(mode).useMDS(useMDS)
                        .uuid(uuid + "_thread_" + threadId)
                        .averagingFrequency(averagingFrequency).build();

        trainer.setName("DefaultTrainer thread " + threadId);
        trainer.setDaemon(true);

        return trainer;
    }

    @Override
    public void finalizeRound(Model originalModel, Model... models) {
        // apply averaging

        // TODO: move averaging here
    }

    @Override
    public void finalizeTraining(Model originalModel, Model... models) {
        finalizeRound(originalModel, models);
    }
}
