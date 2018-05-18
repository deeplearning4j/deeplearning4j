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
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.legacy.*;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class UpdaterTest extends BaseNd4jTest {

    public UpdaterTest(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testAdaGradLegacy() {
        int rows = 1;
        int cols = 1;


        org.nd4j.linalg.learning.legacy.AdaGrad grad = new org.nd4j.linalg.learning.legacy.AdaGrad(rows, cols, 1e-3);
        grad.setStateViewArray(Nd4j.zeros(1, rows * cols), new int[] {rows, cols}, 'c', true);
        INDArray w = Nd4j.ones(rows, cols);
        grad.getGradient(w, 0);
        assertEquals(1e-1, w.getDouble(0), 1e-1);
    }

    @Test
    public void testNesterovs() {
        int rows = 10;
        int cols = 2;


        NesterovsUpdater grad = new NesterovsUpdater(new Nesterovs(0.5, 0.9));
        grad.setStateViewArray(Nd4j.zeros(1, rows * cols), new long[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            //            String learningRates = String.valueOf("\nAdagrad\n " + grad.applyUpdater(W, i)).replaceAll(";", "\n");
            //            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }


    @Test
    public void testAdaGrad() {
        int rows = 10;
        int cols = 2;


        AdaGradUpdater grad = new AdaGradUpdater(new AdaGrad(0.1, AdaGrad.DEFAULT_ADAGRAD_EPSILON));
        grad.setStateViewArray(Nd4j.zeros(1, rows * cols), new long[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            //            String learningRates = String.valueOf("\nAdagrad\n " + grad.applyUpdater(W, i)).replaceAll(";", "\n");
            //            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }

    }

    @Test
    public void testAdaDelta() {
        int rows = 10;
        int cols = 2;


        AdaDeltaUpdater grad = new AdaDeltaUpdater(new AdaDelta());
        grad.setStateViewArray(Nd4j.zeros(1, 2 * rows * cols), new long[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1e-3, 1e-3);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            //            String learningRates = String.valueOf("\nAdaelta\n " + grad.applyUpdater(W, i)).replaceAll(";", "\n");
            //            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }

    @Test
    public void testAdam() {
        int rows = 10;
        int cols = 2;


        AdamUpdater grad = new AdamUpdater(new Adam());
        grad.setStateViewArray(Nd4j.zeros(1, 2 * rows * cols), new long[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1e-3, 1e-3);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            //            String learningRates = String.valueOf("\nAdamUpdater\n " + grad.applyUpdater(W, i)).replaceAll(";", "\n");
            //            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }

    @Test
    public void testNadam() {
        int rows = 10;
        int cols = 2;

        NadamUpdater grad = new NadamUpdater(new Nadam());
        grad.setStateViewArray(Nd4j.zeros(1, 2 * rows * cols), new long[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1e-3, 1e-3);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            //            String learningRates = String.valueOf("\nAdamUpdater\n " + grad.applyUpdater(W, i)).replaceAll(";", "\n");
            //            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }

    @Test
    public void testAdaMax() {
        int rows = 10;
        int cols = 2;


        AdaMaxUpdater grad = new AdaMaxUpdater(new AdaMax());
        grad.setStateViewArray(Nd4j.zeros(1, 2 * rows * cols), new long[] {rows, cols}, 'c', true);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1e-3, 1e-3);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            //            String learningRates = String.valueOf("\nAdaMax\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            //            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
