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

import org.deeplearning4j.nn.gradient.Gradient;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;

@RunWith(MockitoJUnitRunner.class)
@Disabled("mockito")
@Tag(TagNames.FILE_IO)
@NativeTag
public class GradientsTest {

    @Test
    public void when_getBatchSizeIsCalled_expect_batchSizeIsReturned() {
        // Arrange
        Gradients sut = new Gradients(5);

        // Act
        long batchSize = sut.getBatchSize();

        // Assert
        assertEquals(5, batchSize);
    }

    @Test
    public void when_puttingLabels_expect_getLabelReturnsLabels() {
        // Arrange
        Gradient gradient = mock(Gradient.class);
        Gradients sut = new Gradients(5);
        sut.putGradient("test", gradient);

        // Act
        Gradient result = sut.getGradient("test");

        // Assert
        assertSame(gradient, result);
    }
}
