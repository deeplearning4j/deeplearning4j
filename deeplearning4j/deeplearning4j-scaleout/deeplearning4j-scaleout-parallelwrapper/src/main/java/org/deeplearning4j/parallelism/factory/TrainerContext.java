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
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.parallelism.trainer.Trainer;

/**
 * Creates {@link Trainer}
 * instances for use with {@link ParallelWrapper}
 *
 * @author Adam Gibson
 */
public interface TrainerContext {


    /**
     * Initialize the context
     * @param model
     * @param args the arguments to initialize with (maybe null)
     */
    void init(Model model, Object... args);

    /**
     * Create a {@link Trainer}
     * based on the given parameters
     * @param threadId the thread id to use for this worker
     * @param model the model to start the trainer with
     * @param rootDevice the root device id
     * @param useMDS     whether to use MultiDataSet or DataSet
     *               or not
     * @param wrapper the wrapper instance to use with this trainer (this refernece is needed
     *                for coordination with the {@link ParallelWrapper} 's {@link org.deeplearning4j.optimize.api.TrainingListener}
     * @return the created training instance
     */
    Trainer create(String uuid, int threadId, Model model, int rootDevice, boolean useMDS, ParallelWrapper wrapper,
                    WorkspaceMode workspaceMode, int averagingFrequency);


    /**
     * This method is called at averagingFrequency
     *
     * @param originalModel
     * @param models
     */
    void finalizeRound(Model originalModel, Model... models);

    /**
     * This method is called
     *
     * @param originalModel
     * @param models
     */
    void finalizeTraining(Model originalModel, Model... models);
}
