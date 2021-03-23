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

package org.deeplearning4j.spark.impl.paramavg;

import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import static org.junit.jupiter.api.Assertions.assertEquals;
@Tag(TagNames.FILE_IO)
@Tag(TagNames.SPARK)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
@Tag(TagNames.JACKSON_SERDE)
public class TestJsonYaml {

    @Test
    public void testJsonYaml() {
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(2).batchSizePerWorker(32)
                        .exportDirectory("hdfs://SomeDirectory/").saveUpdater(false).averagingFrequency(3)
                        .storageLevel(StorageLevel.MEMORY_ONLY_SER_2()).storageLevelStreams(StorageLevel.DISK_ONLY())
                        .build();

        String json = tm.toJson();
        String yaml = tm.toYaml();

//        System.out.println(json);

        TrainingMaster fromJson = ParameterAveragingTrainingMaster.fromJson(json);
        TrainingMaster fromYaml = ParameterAveragingTrainingMaster.fromYaml(yaml);


        assertEquals(tm, fromJson);
        assertEquals(tm, fromYaml);

    }

}
