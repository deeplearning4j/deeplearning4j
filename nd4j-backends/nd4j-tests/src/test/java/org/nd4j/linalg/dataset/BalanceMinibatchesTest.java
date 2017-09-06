package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.lang.reflect.Array;
import java.util.*;

import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 6/24/16.
 */
public class BalanceMinibatchesTest extends BaseNd4jTest {
    public BalanceMinibatchesTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testBalance() {
        DataSetIterator iterator = new IrisDataSetIterator(10, 150);
        BalanceMinibatches balanceMinibatches = BalanceMinibatches.builder().dataSetIterator(iterator).miniBatchSize(10)
                        .numLabels(3).rootDir(new File("minibatches")).rootSaveDir(new File("minibatchessave")).build();
        balanceMinibatches.balance();
        DataSetIterator balanced = new ExistingMiniBatchDataSetIterator(balanceMinibatches.getRootSaveDir());
        while (balanced.hasNext()) {
            assertTrue(balanced.next().labelCounts().size() > 0);
        }

    }

    @Test
    public void testMiniBatchBalanced() {

        int miniBatchSize = 100;
        DataSetIterator iterator = new IrisDataSetIterator(miniBatchSize, 150);
        BalanceMinibatches balanceMinibatches = BalanceMinibatches.builder().dataSetIterator(iterator)
                        .miniBatchSize(miniBatchSize).numLabels(iterator.totalOutcomes())
                        .rootDir(new File("minibatches")).rootSaveDir(new File("minibatchessave")).build();
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


        ArrayList<Integer> fullBatches = new ArrayList(totalCounts.length);
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
                assertTrue("key " + i + " totalOutcomes: " + iterator.totalOutcomes() + " balancedCounts : "
                                + balancedCounts.containsKey(i) + " val : " + bCounts,
                                balancedCounts.containsKey(i) && balancedCounts.get(i) >= (double) miniBatchSize
                                                / iterator.totalOutcomes());
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
