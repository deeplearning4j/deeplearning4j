/*
 *  ******************************************************************************
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
package org.nd4j.linalg.factory.config;

/**
 * NDArray/DataBuffer lifecycle tracking configuration: master switch,
 * view tracking, deletion tracking, stack depth, report intervals,
 * snapshot files, and per-tracker enable flags.
 */
public interface LifecycleEnvironmentConfig {

    boolean isLifecycleTracking();
    void setLifecycleTracking(boolean enabled);

    boolean isTrackViews();
    void setTrackViews(boolean track);
    boolean isTrackDeletions();
    void setTrackDeletions(boolean track);

    int getStackDepth();
    void setStackDepth(int depth);
    int getReportInterval();
    void setReportInterval(int seconds);
    long getMaxDeletionHistory();
    void setMaxDeletionHistory(long max);

    boolean isSnapshotFiles();
    void setSnapshotFiles(boolean enabled);
    boolean isTrackOperations();
    void setTrackOperations(boolean enabled);

    boolean isNDArrayTracking();
    void setNDArrayTracking(boolean enabled);
    boolean isDataBufferTracking();
    void setDataBufferTracking(boolean enabled);
    boolean isTADCacheTracking();
    void setTADCacheTracking(boolean enabled);
    boolean isShapeCacheTracking();
    void setShapeCacheTracking(boolean enabled);
    boolean isOpContextTracking();
    void setOpContextTracking(boolean enabled);
}
