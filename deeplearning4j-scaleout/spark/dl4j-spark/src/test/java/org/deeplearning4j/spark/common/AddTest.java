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

package org.deeplearning4j.spark.common;

import static org.junit.Assert.*;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.impl.common.Add;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 2/8/15.
 */
public class AddTest extends BaseSparkTest {

    @Test
    public void testAdd() {
        List<INDArray> list = new ArrayList<>();
        for(int i = 0; i < 5; i++)
            list.add(Nd4j.ones(5));
        JavaRDD<INDArray> rdd = sc.parallelize(list);
        INDArray sum = rdd.fold(Nd4j.zeros(5),new Add());
        assertEquals(25,sum.sum(Integer.MAX_VALUE).getDouble(0),1e-1);
    }

}
