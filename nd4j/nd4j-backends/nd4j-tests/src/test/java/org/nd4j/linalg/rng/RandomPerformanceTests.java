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

package org.nd4j.linalg.rng;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.DropOutInverted;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class RandomPerformanceTests extends BaseNd4jTest {

    public RandomPerformanceTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testDropoutPerformance() throws Exception {

        for (int i = 0; i < 100; i++) {
            DropOutInverted opWarmup = new DropOutInverted(Nd4j.createUninitialized(1000000), 0.8);
            Nd4j.getExecutioner().exec(opWarmup, Nd4j.getRandom());
        }

        Nd4j.getExecutioner().commit();


        for (int i = 100; i < 100000001; i *= 10) {
            INDArray x1 = Nd4j.createUninitialized(i);
            INDArray x2 = Nd4j.createUninitialized(i);

            LegacyDropOutInverted op1 = new LegacyDropOutInverted(x1, 0.8);

            long time1 = System.nanoTime();

            Nd4j.getExecutioner().exec(op1);

            Nd4j.getExecutioner().commit();

            long time2 = System.nanoTime();

            long timeLegacy = time2 - time1;

            DropOutInverted op2 = new DropOutInverted(x2, 0.8);

            time1 = System.nanoTime();
            Nd4j.getExecutioner().exec(op2, Nd4j.getRandom());

            Nd4j.getExecutioner().commit();

            time2 = System.nanoTime();
            long timeRecent = time2 - time1;

            log.info("Length: {}; Legacy time: {} us, Current time: {} us; Legacy NPE: {} ns; Current NPE: {}", i,
                            timeLegacy / 1000, timeRecent / 1000, timeLegacy / x1.length(), timeRecent / x1.length());
        }
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
