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

package org.deeplearning4j.util;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 12/29/14.
 */
public class TimeSeriesUtilsTest extends BaseDL4JTest {

    @Test
    public void testMovingAverage() {
        INDArray a = Nd4j.arange(0, 20);
        INDArray result = Nd4j.create(new double[] {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f,
                        12.5f, 13.5f, 14.5f, 15.5f, 16.5f, 17.5f});

        INDArray movingAvg = TimeSeriesUtils.movingAverage(a, 4);
        assertEquals(result, movingAvg);
    }

}
