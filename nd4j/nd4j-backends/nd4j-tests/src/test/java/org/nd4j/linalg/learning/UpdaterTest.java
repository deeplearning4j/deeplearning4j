/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.learning;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.AdaMax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class UpdaterTest extends BaseNd4jTestWithBackends {



    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testAdaGradLegacy(Nd4jBackend backend) {
        int rows = 1;
        int cols = 1;


        org.nd4j.linalg.learning.legacy.AdaGrad grad = new org.nd4j.linalg.learning.legacy.AdaGrad(rows, cols, 1e-3);
        grad.setStateViewArray(Nd4j.zeros(1, rows * cols), new int[] {rows, cols}, 'c', true);
        INDArray w = Nd4j.ones(rows, cols);
        grad.getGradient(w, 0);
        assertEquals(1e-1, w.getDouble(0), 1e-1);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNesterovs(Nd4jBackend backend) {
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
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testAdaGrad(Nd4jBackend backend) {
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
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testAdaDelta(Nd4jBackend backend) {
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
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testAdam(Nd4jBackend backend) {
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
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testNadam(Nd4jBackend backend) {
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
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testAdaMax(Nd4jBackend backend) {
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
