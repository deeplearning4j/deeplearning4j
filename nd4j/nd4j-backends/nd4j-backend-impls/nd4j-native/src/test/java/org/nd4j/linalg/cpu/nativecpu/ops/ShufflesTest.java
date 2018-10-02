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

package org.nd4j.linalg.cpu.nativecpu.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.MatchCondition;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

import static org.junit.Assert.assertEquals;

/**
 * Created by raver119 on 05.08.16.
 */
@Ignore
public class ShufflesTest {



    @Test
    public void testSimpleShuffle1() {
        INDArray array = Nd4j.zeros(10, 10);
        for (int x = 0; x < 10; x++) {
            array.getRow(x).assign(x);
        }

        System.out.println(array);


        System.out.println();

        Nd4j.shuffle(array, 1);

        System.out.println(array);

    }

    @Test
    public void testSymmetricShuffle4() throws Exception {
        INDArray features = Nd4j.zeros(10, 3, 4, 2);
        INDArray labels = Nd4j.zeros(10, 5);

        for (int x = 0; x < 10; x++) {
            features.slice(x).assign(x);
            labels.slice(x).assign(x);
        }

//        System.out.println(features);

        System.out.println();

        DataSet ds = new DataSet(features, labels);
        ds.shuffle();

//        System.out.println(features);

        System.out.println("------------------");


        for (int x = 0; x < 10; x++) {
            double val = features.slice(x).getDouble(0);
            INDArray row = labels.slice(x);

            for (int y = 0; y < row.length(); y++ ) {
                assertEquals(val, row.getDouble(y), 0.001);
            }
        }
    }


    @Test
    public void testSymmetricShuffle4F() throws Exception {
        INDArray features = Nd4j.create(new int[] {10, 3, 4, 2} , 'f');
        INDArray labels = Nd4j.create(new int[] {10, 5}, 'f');

        for (int x = 0; x < 10; x++) {
            features.slice(x).assign(x);
            labels.slice(x).assign(x);
        }


        System.out.println("features.length: " + features.length());

        System.out.println(labels);

        System.out.println();

        DataSet ds = new DataSet(features, labels);
        ds.shuffle();

        System.out.println(labels);

        System.out.println("------------------");


        for (int x = 0; x < 10; x++) {
            double val = features.slice(x).getDouble(0);
            INDArray row = labels.slice(x);

            for (int y = 0; y < row.length(); y++ ) {
                assertEquals(val, row.getDouble(y), 0.001);
            }
        }
    }

    @Test
    public void testBinomial() {
        Distribution distribution = Nd4j.getDistributions().createBinomial(3, Nd4j.create(10).putScalar(1, 0.00001));

        for (int x = 0; x < 10000; x++) {
            INDArray z = distribution.sample(new int[]{1, 10});

            System.out.println();

            MatchCondition condition = new MatchCondition(z, Conditions.equals(0.0));
            int match = Nd4j.getExecutioner().exec(condition, Integer.MAX_VALUE).getInt(0);
            assertEquals(z.length(), match);
        }
    }
}
