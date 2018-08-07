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

package org.deeplearning4j.arbiter.optimize;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.arbiter.optimize.api.*;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.generator.GridSearchCandidateGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.CandidateStatus;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;
import org.junit.Test;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.Callable;

import static org.junit.Assert.*;

public class TestGridSearch {

    @Test
    public void testIndexing() {
        int[] nValues = {2, 3};
        int prod = 2 * 3;
        double[][] expVals = new double[][] {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.5}, {1.0, 0.5}, {0.0, 1.0}, {1.0, 1.0}};
        for (int i = 0; i < prod; i++) {
            double[] out = GridSearchCandidateGenerator.indexToValues(nValues, i, prod);
            double[] exp = expVals[i];
            assertArrayEquals(exp, out, 1e-4);
        }
    }

    @Test
    public void testGeneration() throws Exception {
        Map<String, Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY, new HashMap<>());

        //Define configuration:
        CandidateGenerator candidateGenerator = new GridSearchCandidateGenerator(new BraninSpace(), 4,
                        GridSearchCandidateGenerator.Mode.Sequential, commands);

        //Check sequential:
        double[] expValuesFirst = {-5, 0, 5, 10}; //Range: -5 to +10, with 4 values
        double[] expValuesSecond = {0, 5, 10, 15}; //Range: 0 to +15, with 4 values
        for (int i = 0; i < 4 * 4; i++) {
            BraninConfig conf = (BraninConfig) candidateGenerator.getCandidate().getValue();
            double expF = expValuesFirst[i % 4]; //Changes most rapidly
            double expS = expValuesSecond[i / 4];

            double actF = conf.getX1();
            double actS = conf.getX2();

            assertEquals(expF, actF, 1e-4);
            assertEquals(expS, actS, 1e-4);
        }

        //Check random order. specifically: check that all values are generated, in some order
        double[][] orderedOutput = new double[16][2];
        for (int i = 0; i < expValuesFirst.length; i++) {
            for (int j = 0; j < expValuesSecond.length; j++) {
                orderedOutput[4 * j + i][0] = expValuesFirst[i];
                orderedOutput[4 * j + i][1] = expValuesSecond[j];
            }
        }


        candidateGenerator = new GridSearchCandidateGenerator(new BraninSpace(), 4,
                        GridSearchCandidateGenerator.Mode.RandomOrder, commands);
        boolean[] seen = new boolean[16];
        int seenCount = 0;
        for (int i = 0; i < 4 * 4; i++) {
            assertTrue(candidateGenerator.hasMoreCandidates());
            BraninConfig config = (BraninConfig) candidateGenerator.getCandidate().getValue();
            double x1 = config.getX1();
            double x2 = config.getX2();
            //Work out which of the values this is...
            boolean matched = false;
            for (int j = 0; j < 16; j++) {
                if (Math.abs(orderedOutput[j][0] - x1) < 1e-5 && Math.abs(orderedOutput[j][1] - x2) < 1e-5) {
                    matched = true;
                    if (seen[j])
                        fail("Same candidate generated multiple times");
                    seen[j] = true;
                    seenCount++;
                    break;
                }
            }
            assertTrue("Candidate " + x1 + ", " + x2 + " not found; invalid?", matched);
        }
        assertFalse(candidateGenerator.hasMoreCandidates());
        assertEquals(16, seenCount);
    }


    public static class BraninSpace extends AbstractParameterSpace<BraninConfig> {
        private int[] indices;
        private ParameterSpace<Double> first = new ContinuousParameterSpace(-5, 10);
        private ParameterSpace<Double> second = new ContinuousParameterSpace(0, 15);

        @Override
        public BraninConfig getValue(double[] parameterValues) {
            double f = first.getValue(parameterValues);
            double s = second.getValue(parameterValues);
            return new BraninConfig(f, s); //-5 to +10 and 0 to 15
        }

        @Override
        public int numParameters() {
            return 2;
        }

        @Override
        public List<ParameterSpace> collectLeaves() {
            List<ParameterSpace> list = new ArrayList<>();
            list.addAll(first.collectLeaves());
            list.addAll(second.collectLeaves());
            return list;
        }

        @Override
        public boolean isLeaf() {
            return false;
        }

        @Override
        public void setIndices(int... indices) {
            throw new UnsupportedOperationException();
        }
    }

    @AllArgsConstructor
    @Data
    public static class BraninConfig implements Serializable {
        private double x1;
        private double x2;
    }

    public static class BraninScoreFunction implements ScoreFunction {
        private static final double a = 1.0;
        private static final double b = 5.1 / (4.0 * Math.PI * Math.PI);
        private static final double c = 5.0 / Math.PI;
        private static final double r = 6.0;
        private static final double s = 10.0;
        private static final double t = 1.0 / (8.0 * Math.PI);

        @Override
        public double score(Object m, DataProvider data, Map<String, Object> dataParameters) {
            BraninConfig model = (BraninConfig) m;
            double x1 = model.getX1();
            double x2 = model.getX2();

            return a * Math.pow(x2 - b * x1 * x1 + c * x1 - r, 2.0) + s * (1 - t) * Math.cos(x1) + s;
        }

        @Override
        public double score(Object model, Class<? extends DataSource> dataSource, Properties dataSourceProperties) {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean minimize() {
            return true;
        }

        @Override
        public List<Class<?>> getSupportedModelTypes() {
            return Collections.<Class<?>>singletonList(BraninConfig.class);
        }

        @Override
        public List<Class<?>> getSupportedDataTypes() {
            return Collections.<Class<?>>singletonList(Object.class);
        }
    }

    public static class BraninTaskCreator implements TaskCreator {
        @Override
        public Callable<OptimizationResult> create(final Candidate c, DataProvider dataProvider,
                                                   final ScoreFunction scoreFunction, final List<StatusListener> statusListeners,
                                                   IOptimizationRunner runner) {

            return new Callable<OptimizationResult>() {
                @Override
                public OptimizationResult call() throws Exception {

                    BraninConfig candidate = (BraninConfig) c.getValue();

                    double score = scoreFunction.score(candidate, null, (Map)null);
                    System.out.println(candidate.getX1() + "\t" + candidate.getX2() + "\t" + score);

                    Thread.sleep(20);

                    if (statusListeners != null) {
                        for (StatusListener sl : statusListeners) {
                            sl.onCandidateIteration(null, null, 0);
                        }
                    }

                    CandidateInfo ci = new CandidateInfo(-1, CandidateStatus.Complete, score, System.currentTimeMillis(),
                            null, null, null, null);

                    return new OptimizationResult(c, score, c.getIndex(), null, ci, null);
                }
            };
        }

        @Override
        public Callable<OptimizationResult> create(Candidate candidate, Class<? extends DataSource> dataSource, Properties dataSourceProperties, ScoreFunction scoreFunction, List<StatusListener> statusListeners, IOptimizationRunner runner) {
            throw new UnsupportedOperationException();
        }
    }
}
