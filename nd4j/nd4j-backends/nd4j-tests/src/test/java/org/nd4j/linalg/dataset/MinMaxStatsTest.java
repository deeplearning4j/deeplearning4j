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

package org.nd4j.linalg.dataset;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author Ede Meijer
 */

public class MinMaxStatsTest extends BaseNd4jTestWithBackends {

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testEnforcingNonZeroRange(Nd4jBackend backend) {
        INDArray lower = Nd4j.create(new double[] {2, 3, 4, 5});

        MinMaxStats stats = new MinMaxStats(lower.dup(),
                Nd4j.create(new double[] {8, 3, 3.9, 5 + Nd4j.EPS_THRESHOLD * 0.5}));

        INDArray expectedUpper = Nd4j.create(
                new double[] {8, 3 + Nd4j.EPS_THRESHOLD, 4 + Nd4j.EPS_THRESHOLD, 5 + Nd4j.EPS_THRESHOLD});

        assertEquals(lower, stats.getLower());
        assertEquals(expectedUpper, stats.getUpper());
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
