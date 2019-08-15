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

package org.deeplearning4j.rl4j.util;

import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;

import java.io.IOException;

public interface IDataManager {

    boolean isSaveData();
    String getVideoDir();
    void appendStat(StatEntry statEntry) throws IOException;
    void writeInfo(ILearning iLearning) throws IOException;
    void save(Learning learning) throws IOException;

    //In order for jackson to serialize StatEntry
    //please use Lombok @Value (see QLStatEntry)
    interface StatEntry {
        int getEpochCounter();

        int getStepCounter();

        double getReward();
    }
}
