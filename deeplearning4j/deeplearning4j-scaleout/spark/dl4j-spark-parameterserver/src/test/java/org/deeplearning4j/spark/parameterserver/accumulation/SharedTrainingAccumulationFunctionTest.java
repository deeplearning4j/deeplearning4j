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

package org.deeplearning4j.spark.parameterserver.accumulation;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingAccumulationFunctionTest {
    @Before
    public void setUp() throws Exception {}

    @Test
    public void testAccumulation1() throws Exception {
        INDArray updates1 = Nd4j.create(1000).assign(1.0);
        INDArray updates2 = Nd4j.create(1000).assign(2.0);
        INDArray expUpdates = Nd4j.create(1000).assign(3.0);

        SharedTrainingAccumulationTuple tuple1 = SharedTrainingAccumulationTuple.builder().updaterStateArray(updates1)
                        .scoreSum(1.0).aggregationsCount(1).build();

        SharedTrainingAccumulationTuple tuple2 = SharedTrainingAccumulationTuple.builder().updaterStateArray(updates2)
                        .scoreSum(2.0).aggregationsCount(1).build();

        SharedTrainingAccumulationFunction accumulationFunction = new SharedTrainingAccumulationFunction();

        SharedTrainingAccumulationTuple tupleE = accumulationFunction.call(null, tuple1);

        // testing null + tuple accumulation
        assertEquals(1, tupleE.getAggregationsCount());
        assertEquals(1.0, tupleE.getScoreSum(), 0.01);
        assertEquals(updates1, tupleE.getUpdaterStateArray());


        // testing tuple + tuple accumulation
        SharedTrainingAccumulationTuple tupleResult = accumulationFunction.call(tuple1, tuple2);
        assertEquals(2, tupleResult.getAggregationsCount());
        assertEquals(3.0, tupleResult.getScoreSum(), 0.01);
        assertEquals(expUpdates, tupleResult.getUpdaterStateArray());

    }
}
