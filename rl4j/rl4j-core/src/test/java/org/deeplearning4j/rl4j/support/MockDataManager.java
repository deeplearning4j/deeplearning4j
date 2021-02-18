/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.util.IDataManager;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MockDataManager implements IDataManager {

    private final boolean isSaveData;
    public List<StatEntry> statEntries = new ArrayList<>();
    public int isSaveDataCallCount = 0;
    public int getVideoDirCallCount = 0;
    public int writeInfoCallCount = 0;
    public int saveCallCount = 0;

    public MockDataManager(boolean isSaveData) {
        this.isSaveData = isSaveData;
    }

    @Override
    public boolean isSaveData() {
        ++isSaveDataCallCount;
        return isSaveData;
    }

    @Override
    public String getVideoDir() {
        ++getVideoDirCallCount;
        return null;
    }

    @Override
    public void appendStat(StatEntry statEntry) throws IOException {
        statEntries.add(statEntry);
    }

    @Override
    public void writeInfo(ILearning iLearning) throws IOException {
        ++writeInfoCallCount;
    }

    @Override
    public void save(ILearning learning) throws IOException {
        ++saveCallCount;
    }
}
