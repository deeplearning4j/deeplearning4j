/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.rl4j.observation.transform.filter;

import org.deeplearning4j.rl4j.observation.transform.FilterOperation;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class UniformSkippingFilterTest {

    @Test(expected = IllegalArgumentException.class)
    public void when_negativeSkipFrame_expect_exception() {
        // Act
        new UniformSkippingFilter(-1);
    }

    @Test
    public void when_skippingIs4_expect_firstNotSkippedOther3Skipped() {
        // Assemble
        FilterOperation sut = new UniformSkippingFilter(4);
        boolean[] isSkipped = new boolean[8];

        // Act
        for(int i = 0; i < 8; ++i) {
            isSkipped[i] = sut.isSkipped(null, i, false);
        }

        // Assert
        assertFalse(isSkipped[0]);
        assertTrue(isSkipped[1]);
        assertTrue(isSkipped[2]);
        assertTrue(isSkipped[3]);

        assertFalse(isSkipped[4]);
        assertTrue(isSkipped[5]);
        assertTrue(isSkipped[6]);
        assertTrue(isSkipped[7]);
    }

    @Test
    public void when_isLastObservation_expect_observationNotSkipped() {
        // Assemble
        FilterOperation sut = new UniformSkippingFilter(4);

        // Act
        boolean isSkippedNotLastObservation = sut.isSkipped(null, 1, false);
        boolean isSkippedLastObservation = sut.isSkipped(null, 1, true);

        // Assert
        assertTrue(isSkippedNotLastObservation);
        assertFalse(isSkippedLastObservation);
    }

}
