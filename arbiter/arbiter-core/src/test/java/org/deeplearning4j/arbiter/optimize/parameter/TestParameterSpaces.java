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

package org.deeplearning4j.arbiter.optimize.parameter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestParameterSpaces {


    @Test
    public void testContinuousParameterSpace() {

        ContinuousParameterSpace cps = new ContinuousParameterSpace(0, 1);
        cps.setIndices(0);

        for (int i = 0; i < 10; i++) {
            double d = i / 10.0;
            assertEquals(d, cps.getValue(new double[]{d}), 0.0);
        }

        cps = new ContinuousParameterSpace(10, 20);
        cps.setIndices(0);

        for (int i = 0; i < 10; i++) {
            double d = i / 10.0;
            double exp = d * 10 + 10;
            assertEquals(exp, cps.getValue(new double[]{d}), 0.0);
        }


        cps = new ContinuousParameterSpace(new NormalDistribution(0, 1));
        NormalDistribution nd = new NormalDistribution(0, 1);
        cps.setIndices(0);
        for (int i = 0; i < 11; i++) {
            double d = i / 10.0;
            assertEquals(nd.inverseCumulativeProbability(d), cps.getValue(new double[]{d}), 1e-4);
        }
    }

    @Test
    public void testDiscreteParameterSpace() {
        ParameterSpace<Integer> dps = new DiscreteParameterSpace<>(0, 1, 2, 3, 4);
        dps.setIndices(0);

        for (int i = 0; i < 5; i++) {
            double d = i / 5.0 + 0.1; //Center
            double dEdgeLower = i / 5.0 + 1e-8; //Edge case: just above split threshold
            double dEdgeUpper = (i + 1) / 5.0 - 1e-8; //Edge case: just below split threshold
            assertEquals(i, (int) dps.getValue(new double[]{d}));
            assertEquals(i, (int) dps.getValue(new double[]{dEdgeLower}));
            assertEquals(i, (int) dps.getValue(new double[]{dEdgeUpper}));
        }
    }

    @Test
    public void testIntegerParameterSpace() {
        ParameterSpace<Integer> ips = new IntegerParameterSpace(0, 4);
        ips.setIndices(0);

        for (int i = 0; i < 5; i++) {
            double d = i / 5.0 + 0.1; //Center
            double dEdgeLower = i / 5.0 + 1e-8; //Edge case: just above split threshold
            double dEdgeUpper = (i + 1) / 5.0 - 1e-8; //Edge case: just below split threshold
            assertEquals(i, (int) ips.getValue(new double[]{d}));
            assertEquals(i, (int) ips.getValue(new double[]{dEdgeLower}));
            assertEquals(i, (int) ips.getValue(new double[]{dEdgeUpper}));
        }
    }

    @Test
    public void testBooleanSpace() {
        ParameterSpace<Boolean> bSpace = new BooleanSpace();
        bSpace.setIndices(1); //randomly setting to non zero

        assertEquals(true, (boolean) bSpace.getValue(new double[]{0.0, 0.0}));
        assertEquals(true, (boolean) bSpace.getValue(new double[]{0.0, 0.5}));
        assertEquals(false, (boolean) bSpace.getValue(new double[]{0.0, 0.7}));
        assertEquals(false, (boolean) bSpace.getValue(new double[]{0.0, 1.0}));
    }

}
