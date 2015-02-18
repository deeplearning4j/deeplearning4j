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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Some basic tests for the NDArrayIndex
 *
 * @author Adam Gibson
 */
public class NDArrayIndexTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayIndexTests.class);

    @Test
    public void testInterval() {
        int[] interval = NDArrayIndex.interval(0, 2).indices();
        assertTrue(Arrays.equals(interval, new int[]{0, 1}));
        int[] interval2 = NDArrayIndex.interval(1, 3).indices();
        assertEquals(2, interval2.length);

    }

}
