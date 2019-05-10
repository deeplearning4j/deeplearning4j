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

package org.deeplearning4j.spark.common;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.impl.common.Add;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 2/8/15.
 */
public class AddTest extends BaseSparkTest {

    @Test
    public void testAdd() {
        List<INDArray> list = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            list.add(Nd4j.ones(5));
        JavaRDD<INDArray> rdd = sc.parallelize(list);
        INDArray sum = rdd.fold(Nd4j.zeros(5), new Add());
        assertEquals(25, sum.sum(Integer.MAX_VALUE).getDouble(0), 1e-1);
    }

}
