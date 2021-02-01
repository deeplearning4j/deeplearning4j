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

package org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class RecurrentActorCriticHelperTest {

    private final RecurrentActorCriticHelper sut = new RecurrentActorCriticHelper(3);

    @Test
    public void when_callingCreateValueLabels_expect_INDArrayWithCorrectShape() {
        // Arrange

        // Act
        INDArray result = sut.createValueLabels(4);

        // Assert
        assertArrayEquals(new long[] { 1, 1, 4 }, result.shape());
    }

    @Test
    public void when_callingCreatePolicyLabels_expect_ZeroINDArrayWithCorrectShape() {
        // Arrange

        // Act
        INDArray result = sut.createPolicyLabels(4);

        // Assert
        assertArrayEquals(new long[] { 1, 3, 4 }, result.shape());
        for(int j = 0; j < 3; ++j) {
            for(int i = 0; i < 4; ++i) {
                assertEquals(0.0, result.getDouble(0, j, i), 0.00001);
            }
        }
    }

    @Test
    public void when_callingSetPolicy_expect_advantageSetAtCorrectLocation() {
        // Arrange
        INDArray policyArray = Nd4j.zeros(1, 3, 3);

        // Act
        sut.setPolicy(policyArray, 1, 2, 123.0);

        // Assert
        for(int j = 0; j < 3; ++j) {
            for(int i = 0; i < 3; ++i) {
                if(j == 2 && i == 1) {
                    assertEquals(123.0, policyArray.getDouble(0, j, i), 0.00001);
                } else {
                    assertEquals(0.0, policyArray.getDouble(0, j, i), 0.00001);
                }
            }
        }
    }

}
