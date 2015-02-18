/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.indexing;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import static org.junit.Assert.assertTrue;

/**
 * Test indexing
 *
 * @author Adam Gibson
 */
public class IndicesTests {

    private static Logger log = LoggerFactory.getLogger(IndicesTests.class);

    @Test
    public void testIndexing() {
        int[] indexing = Indices.shape(NDArrayIndex.interval(2, 4), NDArrayIndex.interval(3, 5));
        int[] shape = {2, 2};
        assertTrue(Arrays.equals(shape, indexing));
    }

    @Test
    public void testFillIn() {
        NDArrayIndex[] indexes = new NDArrayIndex[3];
        int[] shape = {4, 3, 2};
        for (int i = 0; i < indexes.length; i++) {
            //already filled in
            if (i == 0)
                indexes[i] = NDArrayIndex.interval(0, 1);
            else
                indexes[i] = NDArrayIndex.interval(0, shape[i]);
        }

        NDArrayIndex[] fillIn = {NDArrayIndex.interval(0, 1)};

        NDArrayIndex[] filledIn = Indices.fillIn(shape, fillIn);
        assertTrue(Arrays.equals(indexes, filledIn));
    }


}
