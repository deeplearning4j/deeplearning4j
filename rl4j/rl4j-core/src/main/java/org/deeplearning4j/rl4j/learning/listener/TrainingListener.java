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
package org.deeplearning4j.rl4j.learning.listener;

/**
 * The base definition of all training event listeners
 *
 * @author Alexandre Boulanger
 */
public interface TrainingListener {
    enum ListenerResponse {
        /**
         * Tell the learning process to continue calling the listeners and the training.
         */
        CONTINUE,

        /**
         * Tell the learning process to stop calling the listeners and terminate the training.
         */
        STOP,
    }

    /**
     * Called once when the training starts.
     * @return A ListenerResponse telling the source of the event if it should go on or cancel the training.
     */
    ListenerResponse onTrainingStart();

    /**
     * Called once when the training has finished. This method is called even when the training has been aborted.
     */
    void onTrainingEnd();

    /**
     * Called before the start of every epoch.
     * @return A ListenerResponse telling the source of the event if it should continue or stop the training.
     */
    ListenerResponse onNewEpoch(IEpochTrainingEvent event);

    /**
     * Called when an epoch has been completed
     * @param event A EpochTrainingResultEvent containing the results of the epoch training
     * @return A ListenerResponse telling the source of the event if it should continue or stop the training.
     */
    ListenerResponse onEpochTrainingResult(IEpochTrainingResultEvent event);

    /**
     * Called regularly to monitor the training progress.
     * @param event A {@link ITrainingProgressEvent}
     * @return A ListenerResponse telling the source of the event if it should continue or stop the training.
     */
    ListenerResponse onTrainingProgress(ITrainingProgressEvent event);
}
