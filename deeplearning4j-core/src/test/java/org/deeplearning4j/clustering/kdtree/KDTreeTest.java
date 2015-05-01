/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.clustering.kdtree;

import org.deeplearning4j.berkeley.Pair;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 1/1/15.
 */
public class KDTreeTest {
    @Test
    public void testTree() {
        KDTree tree = new KDTree(2);
        INDArray half = Nd4j.create(Nd4j.createBuffer(new double[]{0.5, 0.5}));
        INDArray one = Nd4j.create(Nd4j.createBuffer(new double[]{1, 1}));
        tree.insert(half);
        tree.insert(one);
        Pair<Double,INDArray> pair = tree.nn(Nd4j.create(Nd4j.createBuffer(new double[]{0.5,0.5})));
        assertEquals(half,pair.getSecond());
    }


}
