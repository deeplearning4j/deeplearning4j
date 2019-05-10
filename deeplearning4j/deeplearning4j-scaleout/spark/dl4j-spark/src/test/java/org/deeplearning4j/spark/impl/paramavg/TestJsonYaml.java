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

package org.deeplearning4j.spark.impl.paramavg;

import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 20/09/2016.
 */
public class TestJsonYaml {

    @Test
    public void testJsonYaml() {
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(2).batchSizePerWorker(32)
                        .exportDirectory("hdfs://SomeDirectory/").saveUpdater(false).averagingFrequency(3)
                        .storageLevel(StorageLevel.MEMORY_ONLY_SER_2()).storageLevelStreams(StorageLevel.DISK_ONLY())
                        .build();

        String json = tm.toJson();
        String yaml = tm.toYaml();

        System.out.println(json);

        TrainingMaster fromJson = ParameterAveragingTrainingMaster.fromJson(json);
        TrainingMaster fromYaml = ParameterAveragingTrainingMaster.fromYaml(yaml);


        assertEquals(tm, fromJson);
        assertEquals(tm, fromYaml);

    }

}
