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

package org.deeplearning4j.optimize.solver;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.solvers.ConjugateGradient;
import org.deeplearning4j.optimize.solvers.LBFGS;
import org.deeplearning4j.optimize.solvers.LineGradientDescent;
import org.deeplearning4j.optimize.solvers.StochasticGradientDescent;
import org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Cos;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.assertTrue;

public class TestOptimizers extends BaseDL4JTest {

    //For debugging.
    private static final boolean PRINT_OPT_RESULTS = true;

    @Test
    public void testOptimizersBasicMLPBackprop() {
        //Basic tests of the 'does it throw an exception' variety.

        DataSetIterator iter = new IrisDataSetIterator(5, 50);

        OptimizationAlgorithm[] toTest =
                        {OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, OptimizationAlgorithm.LINE_GRADIENT_DESCENT,
                                        OptimizationAlgorithm.CONJUGATE_GRADIENT, OptimizationAlgorithm.LBFGS
                        //OptimizationAlgorithm.HESSIAN_FREE	//Known to not work
                        };

        for (OptimizationAlgorithm oa : toTest) {
            MultiLayerNetwork network = new MultiLayerNetwork(getMLPConfigIris(oa));
            network.init();

            iter.reset();
            network.fit(iter);
        }
    }

    @Test
    public void testOptimizersMLP() {
        //Check that the score actually decreases over time

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        OptimizationAlgorithm[] toTest =
                        {OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, OptimizationAlgorithm.LINE_GRADIENT_DESCENT,
                                        OptimizationAlgorithm.CONJUGATE_GRADIENT, OptimizationAlgorithm.LBFGS
                        //OptimizationAlgorithm.HESSIAN_FREE	//Known to not work
                        };

        DataSet ds = iter.next();
        ds.normalizeZeroMeanZeroUnitVariance();

        for (OptimizationAlgorithm oa : toTest) {
            int nIter = 10;
            MultiLayerNetwork network = new MultiLayerNetwork(getMLPConfigIris(oa));
            network.init();
            double score = network.score(ds);
            assertTrue(score != 0.0 && !Double.isNaN(score));

            if (PRINT_OPT_RESULTS)
                System.out.println("testOptimizersMLP() - " + oa);

            int nCallsToOptimizer = 30;
            double[] scores = new double[nCallsToOptimizer + 1];
            scores[0] = score;
            for (int i = 0; i < nCallsToOptimizer; i++) {
                for( int j=0; j<nIter; j++ ) {
                    network.fit(ds);
                }
                double scoreAfter = network.score(ds);
                scores[i + 1] = scoreAfter;
                assertTrue("Score is NaN after optimization", !Double.isNaN(scoreAfter));
                assertTrue("OA= " + oa + ", before= " + score + ", after= " + scoreAfter, scoreAfter <= score);
                score = scoreAfter;
            }

            if (PRINT_OPT_RESULTS)
                System.out.println(oa + " - " + Arrays.toString(scores));
        }
    }

    private static MultiLayerConfiguration getMLPConfigIris(OptimizationAlgorithm oa) {
        MultiLayerConfiguration c = new NeuralNetConfiguration.Builder().optimizationAlgo(oa)
                        .updater(new AdaGrad(1e-1)).seed(12345L)
                        .list().layer(0,
                                        new DenseLayer.Builder().nIn(4).nOut(3).weightInit(WeightInit.XAVIER)
                                                        .activation(Activation.RELU)
                                                        .build())
                        .layer(1, new OutputLayer.Builder(LossFunction.MCXENT).nIn(3).nOut(3)
                                        .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build())
                        .backprop(true).pretrain(false).build();

        return c;
    }

    //==================================================
    // Sphere Function Optimizer Tests

    @Test
    public void testSphereFnOptStochGradDescent() {
        testSphereFnOptHelper(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, 5, 2);
        testSphereFnOptHelper(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, 5, 10);
        testSphereFnOptHelper(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, 5, 100);
    }

    @Test
    public void testSphereFnOptLineGradDescent() {
        //Test a single line search with calculated search direction (with multiple line search iterations)
        int[] numLineSearchIter = {5, 10};
        for (int n : numLineSearchIter)
            testSphereFnOptHelper(OptimizationAlgorithm.LINE_GRADIENT_DESCENT, n, 2);

        for (int n : numLineSearchIter)
            testSphereFnOptHelper(OptimizationAlgorithm.LINE_GRADIENT_DESCENT, n, 10);

        for (int n : numLineSearchIter)
            testSphereFnOptHelper(OptimizationAlgorithm.LINE_GRADIENT_DESCENT, n, 100);
    }

    @Test
    public void testSphereFnOptCG() {
        //Test a single line search with calculated search direction (with multiple line search iterations)
        int[] numLineSearchIter = {5, 10};
        for (int n : numLineSearchIter)
            testSphereFnOptHelper(OptimizationAlgorithm.CONJUGATE_GRADIENT, n, 2);

        for (int n : numLineSearchIter)
            testSphereFnOptHelper(OptimizationAlgorithm.CONJUGATE_GRADIENT, n, 10);

        for (int n : numLineSearchIter)
            testSphereFnOptHelper(OptimizationAlgorithm.CONJUGATE_GRADIENT, n, 100);
    }

    @Test
    public void testSphereFnOptLBFGS() {
        //Test a single line search with calculated search direction (with multiple line search iterations)
        int[] numLineSearchIter = {5, 10};
        for (int n : numLineSearchIter)
            testSphereFnOptHelper(OptimizationAlgorithm.LBFGS, n, 2);

        for (int n : numLineSearchIter)
            testSphereFnOptHelper(OptimizationAlgorithm.LBFGS, n, 10);

        for (int n : numLineSearchIter)
            testSphereFnOptHelper(OptimizationAlgorithm.LBFGS, n, 100);
    }

    public void testSphereFnOptHelper(OptimizationAlgorithm oa, int numLineSearchIter, int nDimensions) {

        if (PRINT_OPT_RESULTS)
            System.out.println("---------\n Alg= " + oa + ", nIter= " + numLineSearchIter + ", nDimensions= "
                            + nDimensions);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().maxNumLineSearchIterations(numLineSearchIter)
                        .updater(new Sgd(1e-2))
                        .layer(new DenseLayer.Builder().nIn(1).nOut(1).build()).build();
        conf.addVariable("W"); //Normally done by ParamInitializers, but obviously that isn't done here

        Random rng = new DefaultRandom(12345L);
        org.nd4j.linalg.api.rng.distribution.Distribution dist =
                        new org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution(rng, -10, 10);
        Model m = new SphereFunctionModel(nDimensions, dist, conf);
        m.computeGradientAndScore(LayerWorkspaceMgr.noWorkspaces());
        double scoreBefore = m.score();
        assertTrue(!Double.isNaN(scoreBefore) && !Double.isInfinite(scoreBefore));
        if (PRINT_OPT_RESULTS) {
            System.out.println("Before:");
            System.out.println(scoreBefore);
            System.out.println(m.params());
        }

        ConvexOptimizer opt = getOptimizer(oa, conf, m);

        opt.setupSearchState(m.gradientAndScore());
        for( int i=0; i<100; i++ ) {
            opt.optimize(LayerWorkspaceMgr.noWorkspaces());
        }
        m.computeGradientAndScore(LayerWorkspaceMgr.noWorkspaces());
        double scoreAfter = m.score();

        assertTrue(!Double.isNaN(scoreAfter) && !Double.isInfinite(scoreAfter));
        if (PRINT_OPT_RESULTS) {
            System.out.println("After:");
            System.out.println(scoreAfter);
            System.out.println(m.params());
        }

        //Expected behaviour after optimization:
        //(a) score is better (lower) after optimization.
        //(b) Parameters are closer to minimum after optimization (TODO)
        assertTrue("Score did not improve after optimization (b= " + scoreBefore + " ,a= " + scoreAfter + ")",
                        scoreAfter < scoreBefore);
    }

    private static ConvexOptimizer getOptimizer(OptimizationAlgorithm oa, NeuralNetConfiguration conf, Model m) {
        switch (oa) {
            case STOCHASTIC_GRADIENT_DESCENT:
                return new StochasticGradientDescent(conf, new NegativeDefaultStepFunction(), null, m);
            case LINE_GRADIENT_DESCENT:
                return new LineGradientDescent(conf, new NegativeDefaultStepFunction(), null, m);
            case CONJUGATE_GRADIENT:
                return new ConjugateGradient(conf, new NegativeDefaultStepFunction(), null, m);
            case LBFGS:
                return new LBFGS(conf, new NegativeDefaultStepFunction(), null, m);
            default:
                throw new UnsupportedOperationException();
        }
    }


    @Test
    public void testSphereFnOptStochGradDescentMultipleSteps() {
        //Earlier tests: only do a single line search, though each line search will do multiple iterations
        // of line search algorithm.
        //Here, do multiple optimization runs + multiple line search iterations within each run
        //i.e., gradient is re-calculated at each step/run
        //Single step tests earlier won't test storing of state between iterations

        testSphereFnMultipleStepsHelper(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, 100, 5);
    }

    @Test
    public void testSphereFnOptLineGradDescentMultipleSteps() {
        testSphereFnMultipleStepsHelper(OptimizationAlgorithm.LINE_GRADIENT_DESCENT, 100, 5);
    }

    @Test
    public void testSphereFnOptCGMultipleSteps() {
        testSphereFnMultipleStepsHelper(OptimizationAlgorithm.CONJUGATE_GRADIENT, 100, 5);
    }

    @Test
    public void testSphereFnOptLBFGSMultipleSteps() {
        testSphereFnMultipleStepsHelper(OptimizationAlgorithm.LBFGS, 100, 5);
    }


    private static void testSphereFnMultipleStepsHelper(OptimizationAlgorithm oa, int nOptIter,
                    int maxNumLineSearchIter) {
        double[] scores = new double[nOptIter + 1];

        for (int i = 0; i <= nOptIter; i++) {
            Random rng = new DefaultRandom(12345L);
            org.nd4j.linalg.api.rng.distribution.Distribution dist =
                            new org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution(rng, -10, 10);
            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                            .maxNumLineSearchIterations(maxNumLineSearchIter).updater(new Sgd(0.1))
                            .layer(new DenseLayer.Builder().nIn(1).nOut(1).build()).build();
            conf.addVariable("W"); //Normally done by ParamInitializers, but obviously that isn't done here

            Model m = new SphereFunctionModel(100, dist, conf);
            if (i == 0) {
                m.computeGradientAndScore(LayerWorkspaceMgr.noWorkspaces());
                scores[0] = m.score(); //Before optimization
            } else {
                ConvexOptimizer opt = getOptimizer(oa, conf, m);
                for( int j=0; j<100; j++ ) {
                    opt.optimize(LayerWorkspaceMgr.noWorkspaces());
                }
                m.computeGradientAndScore(LayerWorkspaceMgr.noWorkspaces());
                scores[i] = m.score();
                assertTrue(!Double.isNaN(scores[i]) && !Double.isInfinite(scores[i]));
            }
        }

        if (PRINT_OPT_RESULTS) {
            System.out.println("Multiple optimization iterations (" + nOptIter
                            + " opt. iter.) score vs iteration, maxNumLineSearchIter=" + maxNumLineSearchIter + ": "
                            + oa);
            System.out.println(Arrays.toString(scores));
        }

        for (int i = 1; i < scores.length; i++) {
            assertTrue(scores[i] <= scores[i - 1]);
        }
        assertTrue(scores[scores.length - 1] < 1.0); //Very easy function, expect score ~= 0 with any reasonable number of steps/numLineSearchIter
    }


    /** A non-NN optimization problem. Optimization function (cost function) is
     * \sum_i x_i^2. Has minimum of 0.0 at x_i=0 for all x_i
     * See: https://en.wikipedia.org/wiki/Test_functions_for_optimization
     */
    private static class SphereFunctionModel extends SimpleOptimizableModel {
        private static final long serialVersionUID = -6963606137417355405L;

        private SphereFunctionModel(int nParams, org.nd4j.linalg.api.rng.distribution.Distribution distribution,
                        NeuralNetConfiguration conf) {
            super(distribution.sample(new int[] {1, nParams}), conf);
        }


        @Override
        public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
            // Gradients: d(x^2)/dx = 2x
            INDArray gradient = parameters.mul(2);
            Gradient g = new DefaultGradient();
            g.gradientForVariable().put("W", this.gradientView);
            this.gradient = g;
            this.score = Nd4j.getBlasWrapper().dot(parameters, parameters); //sum_i x_i^2
            this.gradientView.assign(gradient);
        }

        @Override
        public int numParams(boolean backwards) {
            return 0;
        }

        @Override
        public void setParamsViewArray(INDArray params) {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public void setBackpropGradientsViewArray(INDArray gradients) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void setCacheMode(CacheMode mode) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void setListeners(TrainingListener... listeners) {

        }

        @Override
        public int getIndex() {
            return 0;
        }

        @Override
        public void setInput(INDArray input, LayerWorkspaceMgr workspaceMgr) {

        }

        @Override
        public boolean isPretrainLayer() {
            return false;
        }

        @Override
        public void clearNoiseWeightParams() {

        }
    }


    //==================================================
    // Rastrigin Function Optimizer Tests


    @Test
    public void testRastriginFnOptStochGradDescentMultipleSteps() {
        testRastriginFnMultipleStepsHelper(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, 5, 20);
    }

    @Test
    public void testRastriginFnOptLineGradDescentMultipleSteps() {
        testRastriginFnMultipleStepsHelper(OptimizationAlgorithm.LINE_GRADIENT_DESCENT, 10, 20);
    }

    @Test
    public void testRastriginFnOptCGMultipleSteps() {
        testRastriginFnMultipleStepsHelper(OptimizationAlgorithm.CONJUGATE_GRADIENT, 10, 20);
    }

    @Test
    public void testRastriginFnOptLBFGSMultipleSteps() {
        testRastriginFnMultipleStepsHelper(OptimizationAlgorithm.LBFGS, 10, 20);
    }


    private static void testRastriginFnMultipleStepsHelper(OptimizationAlgorithm oa, int nOptIter,
                    int maxNumLineSearchIter) {
        double[] scores = new double[nOptIter + 1];

        for (int i = 0; i <= nOptIter; i++) {
            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                            .maxNumLineSearchIterations(maxNumLineSearchIter).miniBatch(false)
                            .updater(new AdaGrad(1e-2))
                            .layer(new DenseLayer.Builder().nIn(1).nOut(1).build()).build();
            conf.addVariable("W"); //Normally done by ParamInitializers, but obviously that isn't done here

            Model m = new RastriginFunctionModel(10, conf);
            int nParams = m.numParams();
            if (i == 0) {
                m.computeGradientAndScore(LayerWorkspaceMgr.noWorkspaces());
                scores[0] = m.score(); //Before optimization
            } else {
                ConvexOptimizer opt = getOptimizer(oa, conf, m);
                opt.getUpdater().setStateViewArray((Layer) m, Nd4j.create(new int[] {1, nParams}, 'c'), true);
                opt.optimize(LayerWorkspaceMgr.noWorkspaces());
                m.computeGradientAndScore(LayerWorkspaceMgr.noWorkspaces());
                scores[i] = m.score();
                assertTrue(!Double.isNaN(scores[i]) && !Double.isInfinite(scores[i]));
            }
        }

        if (PRINT_OPT_RESULTS) {
            System.out.println("Rastrigin: Multiple optimization iterations (" + nOptIter
                            + " opt. iter.) score vs iteration, maxNumLineSearchIter=" + maxNumLineSearchIter + ": "
                            + oa);
            System.out.println(Arrays.toString(scores));
        }
        for (int i = 1; i < scores.length; i++) {
            if (i == 1) {
                assertTrue(scores[i] <= scores[i - 1]); //Require at least one step of improvement
            } else {
                assertTrue(scores[i] <= scores[i - 1]);
            }
        }
    }

    /** Rastrigin function: A much more complex non-NN multi-dimensional optimization problem.
     * Global minimum of 0 at x_i = 0 for all x_i.
     * Very large number of local minima. Can't expect to achieve global minimum with gradient-based (line search)
     * optimizers, but can expect significant improvement in score/cost relative to initial parameters.
     * This implementation has cost function = infinity if any parameters x_i are
     * outside of range [-5.12,5.12]
     * https://en.wikipedia.org/wiki/Rastrigin_function
     */
    private static class RastriginFunctionModel extends SimpleOptimizableModel {
        private static final long serialVersionUID = -1772954508787487941L;

        private RastriginFunctionModel(int nDimensions, NeuralNetConfiguration conf) {
            super(initParams(nDimensions), conf);
        }

        private static INDArray initParams(int nDimensions) {
            Random rng = new DefaultRandom(12345L);
            org.nd4j.linalg.api.rng.distribution.Distribution dist =
                            new org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution(rng, -5.12, 5.12);
            return dist.sample(new int[] {1, nDimensions});
        }


        @Override
        public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
            //Gradient decomposes due to sum, so:
            //d(x^2 - 10*cos(2*Pi*x))/dx
            // = 2x + 20*pi*sin(2*Pi*x)
            INDArray gradient = parameters.mul(2 * Math.PI);
            Nd4j.getExecutioner().exec(new Sin(gradient));
            gradient.muli(20 * Math.PI);
            gradient.addi(parameters.mul(2));

            Gradient g = new DefaultGradient(this.gradientView);
            g.gradientForVariable().put("W", this.gradientView);
            this.gradient = g;
            //If any parameters are outside range [-5.12,5.12]: score = infinity
            INDArray paramExceeds512 = parameters.cond(new Condition() {
                @Override
                public int condtionNum() {
                    return 0;
                }

                @Override
                public double getValue() {
                    return 0;
                }

                @Override
                public double epsThreshold() {
                    return 0;
                }

                @Override
                public Boolean apply(Number input) {
                    return Math.abs(input.doubleValue()) > 5.12;
                }

                @Override
                public Boolean apply(IComplexNumber input) {
                    throw new UnsupportedOperationException();
                }
            });

            int nExceeds512 = paramExceeds512.sum(Integer.MAX_VALUE).getInt(0);
            if (nExceeds512 > 0)
                this.score = Double.POSITIVE_INFINITY;

            //Otherwise:
            double costFn = 10 * parameters.length();
            costFn += Nd4j.getBlasWrapper().dot(parameters, parameters); //xi*xi
            INDArray temp = parameters.mul(2.0 * Math.PI);
            Nd4j.getExecutioner().exec(new Cos(temp));
            temp.muli(-10.0); //After this: each element is -10*cos(2*Pi*xi)
            costFn += temp.sum(Integer.MAX_VALUE).getDouble(0);

            this.score = costFn;
            this.gradientView.assign(gradient);
        }

        @Override
        public int numParams(boolean backwards) {
            return 0;
        }

        @Override
        public void setParamsViewArray(INDArray params) {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public void setBackpropGradientsViewArray(INDArray gradients) {
            throw new UnsupportedOperationException();
        }


        @Override
        public void setCacheMode(CacheMode mode) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void setListeners(TrainingListener... listeners) {

        }

        @Override
        public int getIndex() {
            return 0;
        }

        @Override
        public void setInput(INDArray input, LayerWorkspaceMgr workspaceMgr) {

        }

        @Override
        public boolean isPretrainLayer() {
            return false;
        }

        @Override
        public void clearNoiseWeightParams() {

        }
    }


    //==================================================
    // Rosenbrock Function Optimizer Tests

    @Test
    public void testRosenbrockFnOptLineGradDescentMultipleSteps() {
        testRosenbrockFnMultipleStepsHelper(OptimizationAlgorithm.LINE_GRADIENT_DESCENT, 20, 20);
    }

    @Test
    public void testRosenbrockFnOptCGMultipleSteps() {
        testRosenbrockFnMultipleStepsHelper(OptimizationAlgorithm.CONJUGATE_GRADIENT, 20, 20);
    }

    @Test
    public void testRosenbrockFnOptLBFGSMultipleSteps() {
        testRosenbrockFnMultipleStepsHelper(OptimizationAlgorithm.LBFGS, 20, 20);
    }


    private static void testRosenbrockFnMultipleStepsHelper(OptimizationAlgorithm oa, int nOptIter,
                    int maxNumLineSearchIter) {
        double[] scores = new double[nOptIter + 1];

        for (int i = 0; i <= nOptIter; i++) {
            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                            .maxNumLineSearchIterations(maxNumLineSearchIter)
                            .updater(new Sgd(1e-1))
                            .stepFunction(new org.deeplearning4j.nn.conf.stepfunctions.NegativeDefaultStepFunction())
                            .layer(new DenseLayer.Builder().nIn(1).nOut(1).build())
                            .build();
            conf.addVariable("W"); //Normally done by ParamInitializers, but obviously that isn't done here

            Model m = new RosenbrockFunctionModel(100, conf);
            if (i == 0) {
                m.computeGradientAndScore(LayerWorkspaceMgr.noWorkspaces());
                scores[0] = m.score(); //Before optimization
            } else {
                ConvexOptimizer opt = getOptimizer(oa, conf, m);
                opt.optimize(LayerWorkspaceMgr.noWorkspaces());
                m.computeGradientAndScore(LayerWorkspaceMgr.noWorkspaces());
                scores[i] = m.score();
                assertTrue("NaN or infinite score: " + scores[i],
                                !Double.isNaN(scores[i]) && !Double.isInfinite(scores[i]));
            }
        }

        if (PRINT_OPT_RESULTS) {
            System.out.println("Rosenbrock: Multiple optimization iterations ( " + nOptIter
                            + " opt. iter.) score vs iteration, maxNumLineSearchIter= " + maxNumLineSearchIter + ": "
                            + oa);
            System.out.println(Arrays.toString(scores));
        }
        for (int i = 1; i < scores.length; i++) {
            if (i == 1) {
                assertTrue(scores[i] < scores[i - 1]); //Require at least one step of improvement
            } else {
                assertTrue(scores[i] <= scores[i - 1]);
            }
        }
    }



    /**Rosenbrock function: a multi-dimensional 'valley' type function.
     * Has a single local/global minimum of f(x)=0 at x_i=1 for all x_i.
     * Expect gradient-based optimization functions to find global minimum eventually,
     * but optimization may be slow due to nearly flat gradient along valley.
     * Restricted here to the range [-5,5]. This implementation gives infinite cost/score
     * if any parameter is outside of this range.
     * Parameters initialized in range [-4,4]
     * See: http://www.sfu.ca/~ssurjano/rosen.html
     */
    private static class RosenbrockFunctionModel extends SimpleOptimizableModel {
        private static final long serialVersionUID = -5129494342531033706L;

        private RosenbrockFunctionModel(int nDimensions, NeuralNetConfiguration conf) {
            super(initParams(nDimensions), conf);
        }

        private static INDArray initParams(int nDimensions) {
            Random rng = new DefaultRandom(12345L);
            org.nd4j.linalg.api.rng.distribution.Distribution dist =
                            new org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution(rng, -4.0, 4.0);
            return dist.sample(new int[] {1, nDimensions});
        }

        @Override
        public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
            val nDims = parameters.length();
            INDArray gradient = Nd4j.zeros(nDims);
            double x0 = parameters.getDouble(0);
            double x1 = parameters.getDouble(1);
            double g0 = -400 * x0 * (x1 - x0 * x0) + 2 * (x0 - 1);
            gradient.put(0, 0, g0);
            for (int i = 1; i < nDims - 1; i++) {
                double xim1 = parameters.getDouble(i - 1);
                double xi = parameters.getDouble(i);
                double xip1 = parameters.getDouble(i + 1);
                double g = 200 * (xi - xim1 * xim1) - 400 * xi * (xip1 - xi * xi) + 2 * (xi - 1);
                gradient.put(0, i, g);
            }

            double xl = parameters.getDouble(nDims - 1);
            double xlm1 = parameters.getDouble(nDims - 2);
            double gl = 200 * (xl - xlm1 * xlm1);

            // FIXME: int cast
            gradient.put(0, (int) nDims - 1, gl);
            Gradient g = new DefaultGradient();
            g.gradientForVariable().put("W", gradient);
            this.gradient = g;

            INDArray paramExceeds5 = parameters.cond(new Condition() {
                @Override
                public int condtionNum() {
                    return 0;
                }

                @Override
                public double getValue() {
                    return 0;
                }

                @Override
                public double epsThreshold() {
                    return 0;
                }

                @Override
                public Boolean apply(Number input) {
                    return Math.abs(input.doubleValue()) > 5.0;
                }

                @Override
                public Boolean apply(IComplexNumber input) {
                    throw new UnsupportedOperationException();
                }
            });

            int nExceeds5 = paramExceeds5.sum(Integer.MAX_VALUE).getInt(0);
            if (nExceeds5 > 0)
                this.score = Double.POSITIVE_INFINITY;
            else {
                double score = 0.0;
                for (int i = 0; i < nDims - 1; i++) {
                    double xi = parameters.getDouble(i);
                    double xi1 = parameters.getDouble(i + 1);
                    score += 100.0 * Math.pow((xi1 - xi * xi), 2.0) + (xi - 1) * (xi - 1);
                }


                this.score = score;
            }


        }

        @Override
        public int numParams(boolean backwards) {
            return 0;
        }

        @Override
        public void setParamsViewArray(INDArray params) {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public void setBackpropGradientsViewArray(INDArray gradients) {
            throw new UnsupportedOperationException();
        }


        @Override
        public void setCacheMode(CacheMode mode) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void setListeners(TrainingListener... listeners) {

        }

        @Override
        public int getIndex() {
            return 0;
        }

        @Override
        public void setInput(INDArray input, LayerWorkspaceMgr workspaceMgr) {

        }

        @Override
        public boolean isPretrainLayer() {
            return false;
        }

        @Override
        public void clearNoiseWeightParams() {

        }
    }


    /** Simple abstract class to deal with the fact that we don't care about the majority of the Model/Layer
     * methods here. Classes extending this model for optimizer tests need only implement the score() and
     * gradient() methods.
     */
    private static abstract class SimpleOptimizableModel implements Model, Layer {
        private static final long serialVersionUID = 4409380971404019303L;
        protected INDArray parameters;
        protected INDArray gradientView;
        protected final NeuralNetConfiguration conf;
        protected Gradient gradient;
        protected double score;

        /**@param parameterInit Initial parameters. Also determines dimensionality of problem. Should be row vector.
         */
        private SimpleOptimizableModel(INDArray parameterInit, NeuralNetConfiguration conf) {
            this.parameters = parameterInit.dup();
            this.gradientView = Nd4j.create(parameterInit.shape());
            this.conf = conf;
        }

        @Override
        public void addListeners(TrainingListener... listener) {
            // no-op
        }

        @Override
        public TrainingConfig getConfig() {
            return conf.getLayer();
        }

        /**
         * Init the model
         */
        @Override
        public void init() {

        }

        @Override
        public int getIndex() {
            return 0;
        }

        @Override
        public void setInput(INDArray input, LayerWorkspaceMgr workspaceMgr) {

        }

        @Override
        public void fit() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void update(INDArray gradient, String paramType) {
            if (!"W".equals(paramType))
                throw new UnsupportedOperationException();
            parameters.subi(gradient);
        }

        @Override
        public void setListeners(TrainingListener... listeners) {

        }

        @Override
        public void update(Gradient gradient) {

        }

        @Override
        public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
            return null;
        }

        @Override
        public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
            return null;
        }

        @Override
        public double score() {
            return score;
        }

        @Override
        public Gradient gradient() {
            return gradient;
        }

        @Override
        public double calcL2(boolean backpropParamsOnly) {
            return 0;
        }

        @Override
        public double calcL1(boolean backpropParamsOnly) {
            return 0;
        }

        @Override
        public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
            throw new UnsupportedOperationException("Ensure you implement this function.");
        }

        @Override
        public void accumulateScore(double accum) {
            throw new UnsupportedOperationException();
        }

        @Override
        public INDArray params() {
            return parameters;
        }

        @Override
        public int numParams() {
            // FIXME: int cast
            return (int) parameters.length();
        }

        @Override
        public void setParams(INDArray params) {
            this.parameters = params;
        }

        @Override
        public void fit(INDArray data, LayerWorkspaceMgr workspaceMgr) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Pair<Gradient, Double> gradientAndScore() {
            computeGradientAndScore(LayerWorkspaceMgr.noWorkspaces());
            return new Pair<>(gradient(), score());
        }

        @Override
        public int batchSize() {
            return 1;
        }

        @Override
        public NeuralNetConfiguration conf() {
            return conf;
        }

        @Override
        public void setConf(NeuralNetConfiguration conf) {
            throw new UnsupportedOperationException();
        }

        @Override
        public INDArray input() {
            //Work-around for BaseUpdater.postApply(): Uses Layer.input().size(0)
            //in order to get mini-batch size. i.e., divide by 1 here.
            return Nd4j.zeros(1);
        }

        @Override
        public ConvexOptimizer getOptimizer() {
            throw new UnsupportedOperationException();
        }

        @Override
        public INDArray getParam(String param) {
            return parameters;
        }

        @Override
        public Map<String, INDArray> paramTable() {
            return Collections.singletonMap("W", getParam("W"));
        }

        @Override
        public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
            return paramTable();
        }

        @Override
        public void setParamTable(Map<String, INDArray> paramTable) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void setParam(String key, INDArray val) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void clear() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Type type() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr mgr) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Layer transpose() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Layer clone() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Collection<TrainingListener> getListeners() {
            return null;
        }

        @Override
        public void setListeners(Collection<TrainingListener> listeners) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void setIndex(int index) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void setInputMiniBatchSize(int size) {}

        @Override
        public int getInputMiniBatchSize() {
            return 1;
        }

        @Override
        public void setMaskArray(INDArray maskArray) {}

        @Override
        public INDArray getMaskArray() {
            return null;
        }

        @Override
        public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                        int minibatchSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public INDArray getGradientsViewArray() {
            return gradientView;
        }

        @Override
        public void applyConstraints(int iteration, int epoch) {

        }

        @Override
        public int getIterationCount() {
            return 0;
        }

        @Override
        public int getEpochCount() {
            return 0;
        }

        @Override
        public void setIterationCount(int iterationCount) {

        }

        @Override
        public void setEpochCount(int epochCount) {

        }

        @Override
        public void allowInputModification(boolean allow) {

        }

        @Override
        public LayerHelper getHelper() {
            return null;
        }
    }
}
