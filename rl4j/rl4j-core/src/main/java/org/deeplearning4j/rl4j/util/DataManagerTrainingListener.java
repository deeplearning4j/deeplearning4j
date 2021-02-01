/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.util;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;

/**
 * DataManagerSyncTrainingListener can be added to the listeners of SyncLearning so that the
 * training process can be fed to the DataManager
 */
@Slf4j
public class DataManagerTrainingListener implements TrainingListener {
    private final IDataManager dataManager;

    private int lastSave = -Constants.MODEL_SAVE_FREQ;

    public DataManagerTrainingListener(IDataManager dataManager) {
        this.dataManager = dataManager;
    }

    @Override
    public ListenerResponse onTrainingStart() {
        return ListenerResponse.CONTINUE;
    }

    @Override
    public void onTrainingEnd() {

    }

    @Override
    public ListenerResponse onNewEpoch(IEpochTrainer trainer) {
        IHistoryProcessor hp = trainer.getHistoryProcessor();
        if(hp != null) {
            int[] shape = trainer.getMdp().getObservationSpace().getShape();
            String filename = dataManager.getVideoDir() + "/video-";
            if (trainer instanceof AsyncThread) {
                filename += ((AsyncThread) trainer).getThreadNumber() + "-";
            }
            filename += trainer.getEpochCount() + "-" + trainer.getStepCount() + ".mp4";
            hp.startMonitor(filename, shape);
        }

        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onEpochTrainingResult(IEpochTrainer trainer, IDataManager.StatEntry statEntry) {
        IHistoryProcessor hp = trainer.getHistoryProcessor();
        if(hp != null) {
            hp.stopMonitor();
        }
        try {
            dataManager.appendStat(statEntry);
        } catch (Exception e) {
            log.error("Training failed.", e);
            return ListenerResponse.STOP;
        }

        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onTrainingProgress(ILearning learning) {
        try {
            int stepCounter = learning.getStepCount();
            if (stepCounter - lastSave >= Constants.MODEL_SAVE_FREQ) {
                dataManager.save(learning);
                lastSave = stepCounter;
            }

            dataManager.writeInfo(learning);
        } catch (Exception e) {
            log.error("Training failed.", e);
            return ListenerResponse.STOP;
        }

        return ListenerResponse.CONTINUE;
    }
}
