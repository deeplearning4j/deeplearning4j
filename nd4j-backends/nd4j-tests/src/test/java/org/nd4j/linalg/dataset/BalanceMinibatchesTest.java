package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

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

        int miniBatchSize = 10;
        DataSetIterator iterator = new IrisDataSetIterator(miniBatchSize, 150);
        BalanceMinibatches balanceMinibatches = BalanceMinibatches.builder().dataSetIterator(iterator).miniBatchSize(miniBatchSize)
                .numLabels(iterator.totalOutcomes()).rootDir(new File("minibatches")).rootSaveDir(new File("minibatchessave")).build();
        balanceMinibatches.balance();
        DataSetIterator balanced = new ExistingMiniBatchDataSetIterator(balanceMinibatches.getRootSaveDir());

        assertTrue(iterator.resetSupported()); // this is testing the Iris dataset more than anything
        iterator.reset();
        List<Double> totalCounts = new ArrayList<Double>(iterator.totalOutcomes());
        while (iterator.hasNext()) {
            Map<Integer, Double> outcomes = iterator.next().labelCounts();
            for (int i = 0; i < iterator.totalOutcomes(); i++) {
                if (outcomes.containsKey(i)) {
                    totalCounts.set(i, totalCounts.get(i) + outcomes.get(i));
                }
            }
        }

        List<Integer> fullBatches = new ArrayList<Integer>(totalCounts.size());
        for (int i = 0; i < totalCounts.size(); i++) {
            fullBatches.set(i, totalCounts.get(i).intValue() * iterator.totalOutcomes() / miniBatchSize);
        }

        // this is the number of batches for which we can balance every class
        int fullyBalanceableBatches = Collections.min(fullBatches);
        // check the first few batches are actually balanced
        for (int b = 0; b < fullyBalanceableBatches; b++){
            Map<Integer, Double> balancedCounts = balanced.next().labelCounts();
            for (int i = 0; i < iterator.totalOutcomes(); i++) {
                assertTrue(balancedCounts.containsKey(i) && balancedCounts.get(i) >= miniBatchSize / iterator.totalOutcomes());
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
