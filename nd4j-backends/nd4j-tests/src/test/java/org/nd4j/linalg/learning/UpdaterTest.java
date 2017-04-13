/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.nd4j.linalg.learning;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

@RunWith(Parameterized.class)
public class UpdaterTest extends BaseNd4jTest {

    private static final Logger log = LoggerFactory.getLogger(UpdaterTest.class);



    public UpdaterTest(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testAdaGrad1() {
        int rows = 1;
        int cols = 1;


        AdaGrad grad = new AdaGrad(rows, cols, 1e-3);
        grad.setStateViewArray(Nd4j.zeros(1, rows * cols), new int[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.ones(rows, cols);
        assertEquals(1e-1, grad.getGradient(W, 0).getDouble(0), 1e-1);
    }


    @Test
    public void testEnsureInPlace() {
        //All updaters MUST execute in-place operations on the arrays. This is important for how DL4J works

        GradientUpdater[] updaters = new GradientUpdater[] {new AdaDelta(0.95), new AdaGrad(0.1), new Adam(0.9),
                        new Nesterovs(0.9), new RmsProp(0.1, 0.95), new Sgd(0.1),};
        int[] m = new int[] {2, 1, 2, 1, 1, 1};

        Nd4j.getRandom().setSeed(12345);
        for (int i = 0; i < updaters.length; i++) {
            GradientUpdater u = updaters[i];
            System.out.println(u);
            u.setStateViewArray(Nd4j.zeros(1, m[i] * 10 * 10), new int[] {10, 10}, 'c', true);

            String msg = u.getClass().toString();

            INDArray input = Nd4j.rand(10, 10);
            for (int j = 0; j < 3; j++) {
                INDArray out = u.getGradient(input, j);
                assertTrue(msg, input == out); //Needs to be exact same object, not merely equal
            }
        }
    }

    @Test
    public void testNesterovs() {
        int rows = 10;
        int cols = 2;


        Nesterovs grad = new Nesterovs(0.5);
        grad.setStateViewArray(Nd4j.zeros(1, rows * cols), new int[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdagrad\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }


    @Test
    public void testAdaGrad() {
        int rows = 10;
        int cols = 2;


        AdaGrad grad = new AdaGrad(rows, cols, 0.1);
        grad.setStateViewArray(Nd4j.zeros(1, rows * cols), new int[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdagrad\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }

    }

    @Test
    public void testAdaDelta() {
        int rows = 10;
        int cols = 2;


        AdaDelta grad = new AdaDelta();
        grad.setStateViewArray(Nd4j.zeros(1, 2 * rows * cols), new int[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1e-3, 1e-3);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdaelta\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }

    @Test
    public void testAdam() {
        int rows = 10;
        int cols = 2;


        Adam grad = new Adam();
        grad.setStateViewArray(Nd4j.zeros(1, 2 * rows * cols), new int[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1e-3, 1e-3);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdam\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
