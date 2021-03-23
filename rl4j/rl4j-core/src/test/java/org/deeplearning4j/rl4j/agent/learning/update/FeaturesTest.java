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

package org.deeplearning4j.rl4j.agent.learning.update;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
@Tag(TagNames.FILE_IO)
@NativeTag
public class FeaturesTest {

    @Test
    public void when_creatingFeatureWithBatchSize10_expectGetBatchSizeReturn10() {
        // Arrange
        INDArray[] featuresData = new INDArray[] {Nd4j.rand(10, 1)};

        // Act
        Features sut = new Features(featuresData);

        // Assert
        assertEquals(10, sut.getBatchSize());
    }

    @Test
    public void when_callingGetWithAChannelIndex_expectGetReturnsThatChannelData() {
        // Arrange
        INDArray channel0Data = Nd4j.rand(10, 1);
        INDArray channel1Data = Nd4j.rand(10, 1);
        INDArray[] featuresData = new INDArray[] { channel0Data, channel1Data };

        // Act
        Features sut = new Features(featuresData);

        // Assert
        assertSame(channel1Data, sut.get(1));
    }

}
