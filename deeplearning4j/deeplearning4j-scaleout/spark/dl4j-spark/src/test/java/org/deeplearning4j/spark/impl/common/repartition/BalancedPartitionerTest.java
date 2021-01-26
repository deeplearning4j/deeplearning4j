/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

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
            assertEquals("Found wrong partition output " + p + ", not 0", 0, p);
        }
    }

    @Test
    public void balancedPartitionerFirstElementsWithRemainder() {
        BalancedPartitioner bp = new BalancedPartitioner(10, 10, 1);
        // the 10 first elements should go in the 1st partition
        for (int i = 0; i < 10; i++) {
            int p = bp.getPartition(i);
            assertEquals("Found wrong partition output " + p + ", not 0", 0, p);
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
            assertEquals(10, countPerPartition[i]);
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
                assertEquals(10 + 1, countPerPartition[i]);
            else
                assertEquals(10, countPerPartition[i]);
        }
    }

}
