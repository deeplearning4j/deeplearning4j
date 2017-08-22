package org.deeplearning4j.spark.impl.common.repartition;

import org.junit.Test;

import static org.junit.Assert.assertEquals;


/**
 * Created by huitseeker on 4/4/17.
 */
public class BalancedPartitionerTest {


    @Test
    public void balancedPartitionerFirstElements() {
        BalancedPartitioner bp = new BalancedPartitioner(10, 10, 0);
        // the 10 first elements should go in the 1st partition
        for (int i = 0; i < 10; i++) {
            int p = bp.getPartition(i);
            assertEquals("Found wrong partition output " + p + ", not 0", p, 0);
        }
    }

    @Test
    public void balancedPartitionerFirstElementsWithRemainder() {
        BalancedPartitioner bp = new BalancedPartitioner(10, 10, 1);
        // the 10 first elements should go in the 1st partition
        for (int i = 0; i < 10; i++) {
            int p = bp.getPartition(i);
            assertEquals("Found wrong partition output " + p + ", not 0", p, 0);
        }
    }

    @Test
    public void balancedPartitionerDoesBalance() {
        BalancedPartitioner bp = new BalancedPartitioner(10, 10, 0);
        int[] countPerPartition = new int[10];
        for (int i = 0; i < 10 * 10; i++) {
            int p = bp.getPartition(i);
            countPerPartition[p] += 1;
        }
        for (int i = 0; i < 10; i++) {
            assertEquals(countPerPartition[i], 10);
        }
    }

    @Test
    public void balancedPartitionerDoesBalanceWithRemainder() {
        BalancedPartitioner bp = new BalancedPartitioner(10, 10, 7);
        int[] countPerPartition = new int[10];
        for (int i = 0; i < 10 * 10 + 7; i++) {
            int p = bp.getPartition(i);
            countPerPartition[p] += 1;
        }
        for (int i = 0; i < 10; i++) {
            if (i < 7)
                assertEquals(countPerPartition[i], 10 + 1);
            else
                assertEquals(countPerPartition[i], 10);
        }
    }

}
