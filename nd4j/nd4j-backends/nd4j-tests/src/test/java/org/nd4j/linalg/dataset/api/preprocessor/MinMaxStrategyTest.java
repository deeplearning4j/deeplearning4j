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

package org.nd4j.linalg.dataset.api.preprocessor;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author Ede Meijer
 */
@RunWith(Parameterized.class)
public class MinMaxStrategyTest extends BaseNd4jTest {
    public MinMaxStrategyTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testRowVector() {
        MinMaxStrategy SUT = new MinMaxStrategy(0, 1);

        MinMaxStats stats = new MinMaxStats(Nd4j.create(new float[] {2, 3}), Nd4j.create(new float[] {4, 6}));

        INDArray input = Nd4j.create(new float[] {3, 3});
        INDArray inputCopy = input.dup();

        SUT.preProcess(input, null, stats);
        assertEquals(Nd4j.create(new float[] {0.5f, 0f}), input);

        SUT.revert(input, null, stats);
        assertEquals(inputCopy, input);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
