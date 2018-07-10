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

package org.deeplearning4j.parallelism.trainer;

import lombok.NonNull;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * A Trainer is an individual worker used in {@link org.deeplearning4j.parallelism.ParallelWrapper}
 * for handling training in multi core situations.
 *
 * @author Adam Gibson
 */
public interface Trainer extends Runnable {
    /**
     * Train on a {@link MultiDataSet}
     * @param dataSet the data set to train on
     */
    void feedMultiDataSet(@NonNull MultiDataSet dataSet, long etlTime);


    /**
     * Train on a {@link DataSet}
     * @param dataSet the data set to train on
     */
    void feedDataSet(@NonNull DataSet dataSet, long etlTime);

    /**
     * THe current model for the trainer
     * @return the current  {@link Model}
     * for the worker
     */
    Model getModel();

    /**
     * Update the current {@link Model}
     * for the worker
     * @param model the new model for this worker
     */
    void updateModel(@NonNull Model model);

    boolean isRunning();

    String getUuid();

    /**
     * Shutdown this worker
     */
    void shutdown();

    /**
     * Block the main thread
     * till the trainer is up and running.
     */
    void waitTillRunning();

    /**
     * Set the {@link java.lang.Thread.UncaughtExceptionHandler}
     * for this {@link Trainer}
     * @param handler the handler for uncaught errors
     */
    void setUncaughtExceptionHandler(Thread.UncaughtExceptionHandler handler);

    /**
     * Start this trainer
     */
    void start();

    /**
     * This method returns TRUE if this Trainer implementation assumes periodic aver
     * @return
     */
    boolean averagingRequired();
}
