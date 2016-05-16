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

package org.deeplearning4j.util;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.Assert.*;
/**
 * Created by agibsonccc on 12/29/14.
 */
public class TimeSeriesUtilsTest {

    @Test
    public void testMovingAverage() {
        INDArray a = Nd4j.arange(0,20);
        INDArray result = Nd4j.create(new double[]{
                        1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f,10.5f,11.5f,12.5f,13.5f,14.5f,15.5f,16.5f,17.5f});

        INDArray movingAvg = TimeSeriesUtils.movingAverage(a,4);
        assertEquals(result,movingAvg);
    }

}
