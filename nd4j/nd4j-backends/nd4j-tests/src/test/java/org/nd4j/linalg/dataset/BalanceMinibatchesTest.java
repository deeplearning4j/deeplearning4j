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

package org.nd4j.linalg.dataset;


import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class BalanceMinibatchesTest extends BaseNd4jTestWithBackends {

    @TempDir Path testDir;

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBalance(Nd4jBackend backend) throws Exception {
        DataSetIterator iterator = new IrisDataSetIterator(10, 150);

        File minibatches = new File(testDir.toFile(),"mini-batch-dir");
        File saveDir = new File(testDir.toFile(),"save-dir");

        BalanceMinibatches balanceMinibatches = BalanceMinibatches.builder().dataSetIterator(iterator).miniBatchSize(10)
                        .numLabels(3).rootDir(minibatches).rootSaveDir(saveDir).build();
        balanceMinibatches.balance();
        DataSetIterator balanced = new ExistingMiniBatchDataSetIterator(balanceMinibatches.getRootSaveDir());
        while (balanced.hasNext()) {
            assertTrue(balanced.next().labelCounts().size() > 0);
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMiniBatchBalanced(Nd4jBackend backend) throws Exception {

        int miniBatchSize = 100;
        DataSetIterator iterator = new IrisDataSetIterator(miniBatchSize, 150);

        File minibatches = new File(testDir.toFile(),"mini-batch-dir");
        File saveDir = new File(testDir.toFile(),"save-dir");

        BalanceMinibatches balanceMinibatches = BalanceMinibatches.builder().dataSetIterator(iterator)
                        .miniBatchSize(miniBatchSize).numLabels(iterator.totalOutcomes())
                        .rootDir(minibatches).rootSaveDir(saveDir).build();
        balanceMinibatches.balance();
        DataSetIterator balanced = new ExistingMiniBatchDataSetIterator(balanceMinibatches.getRootSaveDir());

        assertTrue(iterator.resetSupported()); // this is testing the Iris dataset more than anything
        iterator.reset();
        double[] totalCounts = new double[iterator.totalOutcomes()];

        while (iterator.hasNext()) {
            Map<Integer, Double> outcomes = iterator.next().labelCounts();
            for (int i = 0; i < iterator.totalOutcomes(); i++) {
                if (outcomes.containsKey(i))
                    totalCounts[i] += outcomes.get(i);
            }
        }


        List<Integer> fullBatches = new ArrayList(totalCounts.length);
        for (int i = 0; i < totalCounts.length; i++) {
            fullBatches.add(iterator.totalOutcomes() * (int) totalCounts[i] / miniBatchSize);
        }


        // this is the number of batches for which we can balance every class
        int fullyBalanceableBatches = Collections.min(fullBatches);
        // check the first few batches are actually balanced
        for (int b = 0; b < fullyBalanceableBatches; b++) {
            Map<Integer, Double> balancedCounts = balanced.next().labelCounts();
            for (int i = 0; i < iterator.totalOutcomes(); i++) {
                double bCounts = (balancedCounts.containsKey(i) ? balancedCounts.get(i) : 0);
                assertTrue(   balancedCounts.containsKey(i) && balancedCounts.get(i) >= (double) miniBatchSize
                        / iterator.totalOutcomes(),"key " + i + " totalOutcomes: " + iterator.totalOutcomes() + " balancedCounts : "
                                + balancedCounts.containsKey(i) + " val : " + bCounts);
            }
        }


    }



    /**
     * The ordering for this test
     * This test will only be invoked for
     * the given test  and ignored for others
     *
     * @return the ordering for this test
     */
    @Override
    public char ordering() {
        return 'c';
    }
}
