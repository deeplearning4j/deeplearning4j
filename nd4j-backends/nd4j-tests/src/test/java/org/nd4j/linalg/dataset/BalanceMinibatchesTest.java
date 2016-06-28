package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;

import static org.junit.Assert.*;
/**
 * Created by agibsonccc on 6/24/16.
 */
public class BalanceMinibatchesTest extends BaseNd4jTest {
    public BalanceMinibatchesTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testBalance() {
        DataSetIterator iterator = new IrisDataSetIterator(10,150);
        BalanceMinibatches balanceMinibatches = BalanceMinibatches.builder()
                .dataSetIterator(iterator).miniBatchSize(10).numLabels(3)
                .rootDir(new File("minibatches")).rootSaveDir(new File("minibatchessave"))
                .build();
        balanceMinibatches.balance();
        DataSetIterator balanced = new ExistingMiniBatchDataSetIterator(balanceMinibatches.getRootSaveDir());
        while(balanced.hasNext()) {
           assertTrue(balanced.next().labelCounts().size() > 0);
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
