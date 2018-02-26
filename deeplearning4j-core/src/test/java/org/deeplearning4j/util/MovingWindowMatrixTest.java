/*-
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

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 6/11/14.
 */
public class MovingWindowMatrixTest extends BaseDL4JTest {

    @Test
    public void testMovingWindow() {
        INDArray ones = Nd4j.ones(4, 4);
        MovingWindowMatrix m = new MovingWindowMatrix(ones, 2, 2);
        List<INDArray> windows = m.windows();
        assertEquals(4, windows.size());
        MovingWindowMatrix m2 = new MovingWindowMatrix(ones, 2, 2, true);
        List<INDArray> windowsRotate = m2.windows();
        assertEquals(16, windowsRotate.size());


    }



}
