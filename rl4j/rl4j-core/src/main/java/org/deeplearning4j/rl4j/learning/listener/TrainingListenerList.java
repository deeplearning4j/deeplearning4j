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

import java.util.ArrayList;
import java.util.List;

/**
 * The base logic to notify training listeners with the different training events.
 *
 * @author Alexandre Boulanger
 */
public class TrainingListenerList {
    protected final List<TrainingListener> listeners = new ArrayList<>();

    /**
     * Add a listener at the end of the list
     * @param listener The listener to be added
     */
    public void add(TrainingListener listener) {
        listeners.add(listener);
    }

    /**
     * Notify the listeners that the training has started. Will stop early if a listener returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse#STOP}
     * @return whether or not the source training should be stopped
     */
    public boolean notifyTrainingStarted(ITrainingEvent event) {
        for (TrainingListener listener : listeners) {
            if (listener.onTrainingStart(event) == TrainingListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    /**
     * Notify the listeners that the training has finished.
     */
    public void notifyTrainingFinished(ITrainingEvent event) {
        for (TrainingListener listener : listeners) {
            listener.onTrainingEnd(event);
        }
    }

    /**
     * Notify the listeners that a new epoch has started. Will stop early if a listener returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse#STOP}
     * @return whether or not the source training should be stopped
     */
    public boolean notifyNewEpoch(IEpochTrainingEvent event) {
        for (TrainingListener listener : listeners) {
            if (listener.onNewEpoch(event) == TrainingListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    /**
     * Notify the listeners that an epoch has been completed and the training results are available. Will stop early if a listener returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse#STOP}
     * @return whether or not the source training should be stopped
     */
    public boolean notifyEpochTrainingResult(IEpochTrainingResultEvent event) {
        for (TrainingListener listener : listeners) {
            if (listener.onEpochTrainingResult(event) == TrainingListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

}
