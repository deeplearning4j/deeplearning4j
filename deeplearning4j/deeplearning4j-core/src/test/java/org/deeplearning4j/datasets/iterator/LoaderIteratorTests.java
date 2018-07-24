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

import org.deeplearning4j.datasets.iterator.loader.DataSetLoaderIterator;
import org.deeplearning4j.datasets.iterator.loader.MultiDataSetLoaderIterator;
import org.junit.Test;
import org.nd4j.api.loader.Loader;
import org.nd4j.api.loader.LocalFileSourceFactory;
import org.nd4j.api.loader.Source;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class LoaderIteratorTests {

    @Test
    public void testDSLoaderIter(){

        for(boolean r : new boolean[]{false, true}) {
            List<String> l = Arrays.asList("3", "0", "1");
            Random rng = r ? new Random(12345) : null;
            DataSetIterator iter = new DataSetLoaderIterator(l, rng, new Loader<DataSet>() {
                @Override
                public DataSet load(Source source) throws IOException {
                    INDArray i = Nd4j.scalar(Integer.valueOf(source.getPath()));
                    return new DataSet(i, i);
                }
            }, new LocalFileSourceFactory());

            int count = 0;
            int[] exp = {3, 0, 1};
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                if(!r) {
                    assertEquals(exp[count], ds.getFeatures().getInt(0));
                }
                count++;
            }
            assertEquals(3, count);

            iter.reset();
            assertTrue(iter.hasNext());
        }
    }

    @Test
    public void testMDSLoaderIter(){

        for(boolean r : new boolean[]{false, true}) {
            List<String> l = Arrays.asList("3", "0", "1");
            Random rng = r ? new Random(12345) : null;
            MultiDataSetIterator iter = new MultiDataSetLoaderIterator(l, null, new Loader<MultiDataSet>() {
                @Override
                public MultiDataSet load(Source source) throws IOException {
                    INDArray i = Nd4j.scalar(Integer.valueOf(source.getPath()));
                    return new org.nd4j.linalg.dataset.MultiDataSet(i, i);
                }
            }, new LocalFileSourceFactory());

            int count = 0;
            int[] exp = {3, 0, 1};
            while (iter.hasNext()) {
                MultiDataSet ds = iter.next();
                if(!r) {
                    assertEquals(exp[count], ds.getFeatures()[0].getInt(0));
                }
                count++;
            }
            assertEquals(3, count);

            iter.reset();
            assertTrue(iter.hasNext());
        }
    }

}
