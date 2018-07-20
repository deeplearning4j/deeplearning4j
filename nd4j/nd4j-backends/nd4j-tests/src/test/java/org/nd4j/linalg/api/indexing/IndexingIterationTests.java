/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.indexing;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.*;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class IndexingIterationTests extends BaseNd4jTest {

    public IndexingIterationTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testAll() {
        INDArrayIndex all = NDArrayIndex.all();
        INDArray init = Nd4j.create(2, 2);
        all.init(init, 1);
        assertTrue(all.hasNext());
        assertEquals(0, all.current());
        assertEquals(0, all.next());
        assertEquals(2, all.length());
        assertEquals(1, all.next());
        assertFalse(all.hasNext());
    }

    @Test
    public void testInterval() {
        INDArrayIndex interval = NDArrayIndex.interval(0, 2);
        assertTrue(interval.hasNext());
        assertEquals(2, interval.length());
        assertEquals(0, interval.next());
        assertEquals(1, interval.next());
        assertFalse(interval.hasNext());

    }

    @Test
    public void testIntervalInclusive() {
        INDArrayIndex interval = NDArrayIndex.interval(0, 1, 2, true);
        assertTrue(interval.hasNext());
        assertEquals(3, interval.end());
        assertEquals(3, interval.length());
        assertEquals(0, interval.next());
        assertEquals(1, interval.next());
        assertTrue(interval.hasNext());
        assertEquals(2, interval.next());
        assertFalse(interval.hasNext());

    }

    @Test
    public void testIntervalWithStride() {
        INDArrayIndex interval = NDArrayIndex.interval(3, 2, 6);
        assertTrue(interval.hasNext());
        assertEquals(2, interval.length());
        assertEquals(3, interval.next());
        assertTrue(interval.hasNext());
        assertEquals(5, interval.next());
        assertFalse(interval.hasNext());

    }

    @Test
    public void testNewAxis() {
        INDArrayIndex newAxis = NDArrayIndex.newAxis();
        assertEquals(0, newAxis.length());
        assertFalse(newAxis.hasNext());

    }


    @Test
    public void testIntervalStrideGreaterThan1() {
        INDArrayIndex interval = NDArrayIndex.interval(0, 2, 2);
        assertTrue(interval.hasNext());
        assertEquals(1, interval.length());
        assertEquals(0, interval.next());
        assertFalse(interval.hasNext());

    }

    @Test
    public void testPoint() {
        INDArrayIndex point = new PointIndex(1);
        assertTrue(point.hasNext());
        assertEquals(1, point.length());
        assertEquals(1, point.next());
        assertFalse(point.hasNext());
    }

    @Test
    public void testEmpty() {
        INDArrayIndex empty = new NDArrayIndexEmpty();
        assertFalse(empty.hasNext());
        assertEquals(0, empty.length());
    }

    @Test
    public void testSpecifiedIndex() {
        INDArrayIndex indArrayIndex = new SpecifiedIndex(2);
        assertEquals(1, indArrayIndex.length());
        assertTrue(indArrayIndex.hasNext());
        assertEquals(2, indArrayIndex.next());
        assertEquals(2, indArrayIndex.current());
        assertEquals(2, indArrayIndex.end());
        assertEquals(indArrayIndex.offset(), indArrayIndex.end());
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
