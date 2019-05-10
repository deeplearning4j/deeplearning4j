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

package org.nd4j.parameterserver.distributed.messages.aggregations;

import lombok.extern.slf4j.Slf4j;
import org.junit.*;
import org.junit.rules.Timeout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@Ignore
@Deprecated
public class VoidAggregationTest {
    private static final short NODES = 100;
    private static final int ELEMENTS_PER_NODE = 3;

    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Rule
    public Timeout globalTimeout = Timeout.seconds(30);

    /**
     * In this test we check for aggregation of sample vector.
     *
     * @throws Exception
     */
    @Test
    public void getAccumulatedResult1() throws Exception {
        INDArray exp = Nd4j.linspace(0, (NODES * ELEMENTS_PER_NODE) - 1, NODES * ELEMENTS_PER_NODE);

        List<VectorAggregation> aggregations = new ArrayList<>();
        for (int i = 0, j = 0; i < NODES; i++) {

            INDArray array = Nd4j.create(ELEMENTS_PER_NODE);

            for (int e = 0; e < ELEMENTS_PER_NODE; j++, e++) {
                array.putScalar(e, (double) j);
            }
            VectorAggregation aggregation = new VectorAggregation(1L, NODES, (short) i, array);

            aggregations.add(aggregation);
        }


        VectorAggregation aggregation = aggregations.get(0);
        for (VectorAggregation vectorAggregation : aggregations) {
            aggregation.accumulateAggregation(vectorAggregation);
        }

        INDArray payload = aggregation.getAccumulatedResult();
        log.info("Payload shape: {}", payload.shape());
        assertEquals(exp.length(), payload.length());
        assertEquals(exp, payload);
    }


    /**
     * This test checks for aggregation of single-array dot
     *
     * @throws Exception
     */
    @Test
    public void getScalarDotAggregation1() throws Exception {
        INDArray x = Nd4j.linspace(0, (NODES * ELEMENTS_PER_NODE) - 1, NODES * ELEMENTS_PER_NODE);
        INDArray y = x.dup();
        double exp = Nd4j.getBlasWrapper().dot(x, y);

        List<DotAggregation> aggregations = new ArrayList<>();
        for (int i = 0, j = 0; i < NODES; i++) {
            INDArray arrayX = Nd4j.create(ELEMENTS_PER_NODE);
            INDArray arrayY = Nd4j.create(ELEMENTS_PER_NODE);

            for (int e = 0; e < ELEMENTS_PER_NODE; j++, e++) {
                arrayX.putScalar(e, (double) j);
                arrayY.putScalar(e, (double) j);
            }

            double dot = Nd4j.getBlasWrapper().dot(arrayX, arrayY);

            DotAggregation aggregation = new DotAggregation(1L, NODES, (short) i, Nd4j.scalar(dot));

            aggregations.add(aggregation);
        }

        DotAggregation aggregation = aggregations.get(0);


        for (DotAggregation vectorAggregation : aggregations) {
            aggregation.accumulateAggregation(vectorAggregation);
        }

        INDArray result = aggregation.getAccumulatedResult();
        assertEquals(true, result.isScalar());
        assertEquals(exp, result.getDouble(0), 1e-5);
    }


    @Test
    public void getBatchedDotAggregation1() throws Exception {
        INDArray x = Nd4j.create(5, 300).assign(2.0);
        INDArray y = x.dup();

        x.muli(y);
        INDArray exp = x.sum(1);

        List<DotAggregation> aggregations = new ArrayList<>();
        for (int i = 0, j = 0; i < NODES; i++) {
            INDArray arrayX = Nd4j.create(5, ELEMENTS_PER_NODE);
            INDArray arrayY = Nd4j.create(5, ELEMENTS_PER_NODE);

            arrayX.assign(2.0);
            arrayY.assign(2.0);

            DotAggregation aggregation = new DotAggregation(1L, NODES, (short) i, arrayX.mul(arrayY));

            aggregations.add(aggregation);
        }

        DotAggregation aggregation = aggregations.get(0);

        int cnt = 1;
        for (DotAggregation vectorAggregation : aggregations) {
            aggregation.accumulateAggregation(vectorAggregation);
            cnt++;

            // we're checking for actual number of missing chunks
            //assertEquals( NODES - cnt,aggregation.getMissingChunks());
        }

        INDArray result = aggregation.getAccumulatedResult();
        assertArrayEquals(exp.shapeInfoDataBuffer().asInt(), result.shapeInfoDataBuffer().asInt());
        assertEquals(exp, result);
    }

}
