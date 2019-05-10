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

package org.deeplearning4j.spark.parameterserver.iterators;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class VirtualDataSetIteratorTest {
    @Before
    public void setUp() throws Exception {}


    @Test
    public void testSimple1() throws Exception {
        List<Iterator<DataSet>> iterators = new ArrayList<>();

        List<DataSet> first = new ArrayList<>();
        List<DataSet> second = new ArrayList<>();

        for (int i = 0; i < 100; i++) {
            INDArray features = Nd4j.create(100).assign(i);
            INDArray labels = Nd4j.create(10).assign(i);
            DataSet ds = new DataSet(features, labels);

            if (i < 25)
                first.add(ds);
            else
                second.add(ds);
        }

        iterators.add(first.iterator());
        iterators.add(second.iterator());

        VirtualDataSetIterator vdsi = new VirtualDataSetIterator(iterators);
        int cnt = 0;
        while (vdsi.hasNext()) {
            DataSet ds = vdsi.next();

            assertEquals((double) cnt, ds.getFeatures().meanNumber().doubleValue(), 0.0001);
            assertEquals((double) cnt, ds.getLabels().meanNumber().doubleValue(), 0.0001);

            cnt++;
        }

        assertEquals(100, cnt);
    }
}
