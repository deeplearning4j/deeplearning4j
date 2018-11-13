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

package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.parallel.AsyncBlockIterator;
import org.deeplearning4j.datasets.iterator.parallel.VirtualDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class AsyncBlockIteratorTests {

    @Test
    public void testSimple1Thread(){

        DataSetIterator iris = new IrisDataSetIterator(30, 150);
        DataSetIterator iris2 = new IrisDataSetIterator(30, 150);

        DataSetIterator iris3 = new IrisDataSetIterator(30, 150);
        List<DataSet> irisAsList = new ArrayList<>();
        while(iris3.hasNext()){
            irisAsList.add(iris3.next());
        }

        AsyncBlockIterator iter = new AsyncBlockIterator(new int[]{0}, 2, Arrays.asList(iris, iris2));

        for( int i=0; i<10; i++ ) {
            assertTrue(iter.hasAnything());
            DataSet[] arr = iter.next(1);
            assertEquals(1, arr.length);
            assertEquals(irisAsList.get(i%5), arr[0]);
        }
        assertFalse(iter.hasAnything());
    }

    @Test
    public void testSimpleNThreads(){
        DataSetIterator iris3 = new IrisDataSetIterator(30, 150);
        List<DataSet> irisAsList = new ArrayList<>();
        while (iris3.hasNext()) {
            irisAsList.add(iris3.next());
        }

        for( int numThreads : new int[]{2, 4}) {
//        for( int numThreads : new int[]{4}) {
            System.out.println("---------------------------------------------");
            List<DataSetIterator> initalIters = new ArrayList<>();
            for( int i=0; i<2*numThreads; i++ ){
                initalIters.add(new IrisDataSetIterator(30, 150));
            }

            int[] deviceThreadAffinity = new int[numThreads];    //Simple test: all device 0
            AsyncBlockIterator iter = new AsyncBlockIterator(deviceThreadAffinity, 2, initalIters);
            DataSet[] first = iter.next(numThreads);
            assertEquals(numThreads, first.length);
            for( int i=0; i<numThreads; i++ ){
                //All N threads should get first DataSet from their respective copies of the iris iterator
                assertEquals(irisAsList.get(0), first[i]);
            }

            //Add new iters
            List<DataSetIterator> newIters = new ArrayList<>();
            for( int i=0; i<2*numThreads; i++ ){
                newIters.add(new IrisDataSetIterator(30, 150));
            }
            iter.attach(newIters);

            //N threads, 4 iterators each, x5 DataSets per epoch
            for(int i=1; i<20; i++ ){
                assertTrue(iter.hasAnything());
                DataSet[] arr = iter.next(numThreads);
                assertEquals(numThreads, arr.length);
                for( int j=0; j<numThreads; j++ ) {
                    assertEquals(irisAsList.get(i % 5), arr[j]);
                }
            }
            assertFalse(iter.hasAnything());

            //Check that expected number of iterators were allocated to each thread (expected 4)
            int pos = 0;
            for(VirtualDataSetIterator vi : iter.getVirtualIters()){
                String s = String.valueOf(pos);
                assertEquals(s, 4, vi.getIterators().size());
            }
        }

        System.out.println("---------------------------------------------");
    }

    @Test
    public void testAdditionAfterAllConsumed(){


    }
}
