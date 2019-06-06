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

package org.deeplearning4j.ui.stats;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.ui.stats.api.StatsInitializationConfiguration;
import org.deeplearning4j.ui.stats.api.StatsInitializationReport;
import org.deeplearning4j.ui.stats.api.StatsReport;
import org.deeplearning4j.ui.stats.api.StatsUpdateConfiguration;
import org.deeplearning4j.ui.stats.impl.DefaultStatsUpdateConfiguration;
import org.deeplearning4j.ui.stats.impl.java.JavaStatsInitializationReport;
import org.deeplearning4j.ui.stats.impl.java.JavaStatsReport;
import org.deeplearning4j.ui.storage.impl.JavaStorageMetaData;

/**
 * J7StatsListener: a version of the {@link StatsListener} but with Java 7 compatibility
 * <p>
 * Stats are collected and passed on to a {@link StatsStorageRouter} - for example, for storage and/or displaying in the UI,
 * use {@link org.deeplearning4j.ui.storage.InMemoryStatsStorage} or {@link org.deeplearning4j.ui.storage.FileStatsStorage}.
 *
 * @author Alex Black
 */
@Slf4j
public class J7StatsListener extends BaseStatsListener {

    /**
     * Create a StatsListener with network information collected at every iteration. Equivalent to {@link #J7StatsListener(StatsStorageRouter, int)}
     * with {@code listenerFrequency == 1}
     *
     * @param router Where/how to store the calculated stats. For example, {@link org.deeplearning4j.ui.storage.InMemoryStatsStorage} or
     *               {@link org.deeplearning4j.ui.storage.FileStatsStorage}
     */
    public J7StatsListener(StatsStorageRouter router) {
        this(router, null, null, null, null);
    }

    /**
     * Create a StatsListener with network information collected every n >= 1 time steps
     *
     * @param router            Where/how to store the calculated stats. For example, {@link org.deeplearning4j.ui.storage.InMemoryStatsStorage} or
     *                          {@link org.deeplearning4j.ui.storage.FileStatsStorage}
     * @param listenerFrequency Frequency with which to collect stats information
     */
    public J7StatsListener(StatsStorageRouter router, int listenerFrequency) {
        this(router, null, new DefaultStatsUpdateConfiguration.Builder().reportingFrequency(listenerFrequency).build(),
                        null, null);
    }

    public J7StatsListener(StatsStorageRouter router, StatsInitializationConfiguration initConfig,
                    StatsUpdateConfiguration updateConfig, String sessionID, String workerID) {
        super(router, initConfig, updateConfig, sessionID, workerID);
    }

    @Override
    public StatsInitializationReport getNewInitializationReport() {
        return new JavaStatsInitializationReport();
    }

    @Override
    public StatsReport getNewStatsReport() {
        return new JavaStatsReport();
    }

    @Override
    public StorageMetaData getNewStorageMetaData(long initTime, String sessionID, String workerID) {
        return new JavaStorageMetaData(initTime, sessionID, BaseStatsListener.TYPE_ID, workerID,
                        JavaStatsInitializationReport.class, JavaStatsReport.class);
    }

    @Override
    public J7StatsListener clone() {
        return new J7StatsListener(this.getStorageRouter(), this.getInitConfig(), this.getUpdateConfig(), null, null);
    }
}
