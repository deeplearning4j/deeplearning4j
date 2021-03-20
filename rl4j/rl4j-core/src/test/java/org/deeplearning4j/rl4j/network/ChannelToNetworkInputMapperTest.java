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

package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

@RunWith(MockitoJUnitRunner.class)
@Disabled("Mockito")
@Tag(TagNames.FILE_IO)
@NativeTag
public class ChannelToNetworkInputMapperTest {

    @Test
    public void when_mapIsEmpty_expect_exception() {
        try {
            new ChannelToNetworkInputMapper(new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[0], new String[] { "TEST" }, new String [] { "TEST" });
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "networkInputsToChannelNameMap is empty.";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_networkInputNamesIsEmpty_expect_exception() {
        try {
            ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] map = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
                    ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("TEST", "TEST")
            };
            new ChannelToNetworkInputMapper(map, new String[0], new String [] { "TEST" });
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "networkInputNames is empty.";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_channelNamesIsEmpty_expect_exception() {
        try {
            ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] map = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
                    ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("TEST", "TEST")
            };
            new ChannelToNetworkInputMapper(map, new String [] { "TEST" }, new String[0]);
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "channelNames is empty.";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_notAllInputsAreMapped_expect_exception() {
        try {
            ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] map = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
                    ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("TEST", "TEST")
            };
            new ChannelToNetworkInputMapper(map, new String [] { "TEST", "NOT-MAPPED" }, new String [] { "TEST" });
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "All network inputs must be mapped exactly once. Input 'NOT-MAPPED' is mapped 0 times.";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_anInputIsMappedMultipleTimes_expect_exception() {
        try {
            ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] map = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
                    ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("TEST", "TEST"),
                    ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("TEST1", "TEST"),
                    ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("TEST1", "TEST")
            };
            new ChannelToNetworkInputMapper(map, new String [] { "TEST", "TEST1" }, new String [] { "TEST" });
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "All network inputs must be mapped exactly once. Input 'TEST1' is mapped 2 times.";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_aMapInputDoesNotExist_expect_exception() {
        try {
            ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] map = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
                    ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("TEST", "TEST"),
                    ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("TEST1", "TEST"),
            };
            new ChannelToNetworkInputMapper(map, new String [] { "TEST" }, new String [] { "TEST" });
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "'TEST1' not found in networkInputNames";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_aMapFeatureDoesNotExist_expect_exception() {
        try {
            ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] map = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
                    ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("TEST", "TEST"),
                    ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("TEST1", "TEST1"),
            };
            new ChannelToNetworkInputMapper(map, new String [] { "TEST", "TEST1" }, new String [] { "TEST" });
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "'TEST1' not found in channelNames";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_callingObservationGetNetworkInputs_expect_aCorrectlyOrderedINDArrayArray() {
        // ARRANGE
        ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] map = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
                ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("IN-1", "FEATURE-1"),
                ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("IN-2", "FEATURE-2"),
                ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("IN-3", "FEATURE-3"),
        };
        String[] networkInputs = new String[] { "IN-1", "IN-2", "IN-3" };
        String[] channelNames = new String[] { "FEATURE-1", "FEATURE-2", "FEATURE-UNUSED", "FEATURE-3" };
        ChannelToNetworkInputMapper sut = new ChannelToNetworkInputMapper(map, networkInputs, channelNames);
        INDArray feature1 = Nd4j.rand(1, 2);
        INDArray feature2 = Nd4j.rand(1, 2);
        INDArray featureUnused = Nd4j.rand(1, 2);
        INDArray feature3 = Nd4j.rand(1, 2);
        Observation observation = new Observation(new INDArray[] { feature1, feature2, featureUnused, feature3 });

        // ACT
        INDArray[] results = sut.getNetworkInputs(observation);

        // ASSERT
        assertSame(feature1, results[0]);
        assertSame(feature2, results[1]);
        assertSame(feature3, results[2]);
    }

    @Test
    public void when_callingFeaturesGetNetworkInputs_expect_aCorrectlyOrderedINDArrayArray() {
        // ARRANGE
        ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] map = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
                ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("IN-1", "FEATURE-1"),
                ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("IN-2", "FEATURE-2"),
                ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("IN-3", "FEATURE-3"),
        };
        String[] networkInputs = new String[] { "IN-1", "IN-2", "IN-3" };
        String[] channelNames = new String[] { "FEATURE-1", "FEATURE-2", "FEATURE-UNUSED", "FEATURE-3" };
        ChannelToNetworkInputMapper sut = new ChannelToNetworkInputMapper(map, networkInputs, channelNames);
        INDArray feature1 = Nd4j.rand(1, 2);
        INDArray feature2 = Nd4j.rand(1, 2);
        INDArray featureUnused = Nd4j.rand(1, 2);
        INDArray feature3 = Nd4j.rand(1, 2);
        Features features = new Features(new INDArray[] { feature1, feature2, featureUnused, feature3 });

        // ACT
        INDArray[] results = sut.getNetworkInputs(features);

        // ASSERT
        assertSame(feature1, results[0]);
        assertSame(feature2, results[1]);
        assertSame(feature3, results[2]);
    }

}
