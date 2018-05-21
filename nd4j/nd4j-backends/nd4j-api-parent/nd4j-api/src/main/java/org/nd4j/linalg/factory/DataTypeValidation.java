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
 *
 */

package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Data opType validation
 *
 * @author Adam Gibson
 */
public class DataTypeValidation {
    public static void assertDouble(INDArray... d) {
        for (INDArray d1 : d)
            assertDouble(d1);
    }

    public static void assertFloat(INDArray... d2) {
        for (INDArray d3 : d2)
            assertFloat(d3);
    }

    public static void assertDouble(INDArray d) {
        if (d.data().dataType() != DataBuffer.Type.DOUBLE)
            throw new IllegalStateException("Given ndarray does not have data opType double");
    }

    public static void assertFloat(INDArray d2) {
        if (d2.data().dataType() != DataBuffer.Type.FLOAT) {
            throw new IllegalStateException("Given ndarray does not have data opType float");
        }
    }

    public static void assertSameDataType(INDArray... indArrays) {
        if (indArrays == null || indArrays.length < 2)
            return;
        DataBuffer.Type type = indArrays[0].data().dataType();
        for (int i = 1; i < indArrays.length; i++) {
            assert indArrays[i].data().dataType() == (type);
        }
    }


}
