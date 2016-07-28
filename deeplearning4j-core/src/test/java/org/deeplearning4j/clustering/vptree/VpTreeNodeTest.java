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

package org.deeplearning4j.clustering.vptree;

import org.deeplearning4j.clustering.sptree.DataPoint;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author Anatoly Borisov
 */
public class VpTreeNodeTest {

    @Test
    public void vpTreeTest() {
        List<DataPoint> points = new ArrayList<>();
        points.add(new DataPoint(0,Nd4j.create(new double[]{55,55})));
        points.add(new DataPoint(1,Nd4j.create(new double[]{60,60})));
        points.add(new DataPoint(2,Nd4j.create(new double[]{65,65})));
        VPTree tree = new VPTree(points);
        List<DataPoint> add = new ArrayList<>();
        List<Double> distances = new ArrayList<>();
        tree.search(new DataPoint(0,Nd4j.create(new double[]{50,50})),1,add,distances);
        DataPoint assertion = add.get(0);
        assertEquals(new DataPoint(0,Nd4j.create(new double[]{55,55})),assertion);


    }

}
