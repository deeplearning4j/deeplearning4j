/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.buffer;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertThrows;

@RunWith(Parameterized.class)
public class DataTypeValidationTests extends BaseNd4jTest {
    DataType initialType;

    public DataTypeValidationTests(Nd4jBackend backend) {
        super(backend);
    }

    @BeforeEach
    public void setUp() {
        initialType = Nd4j.dataType();
        Nd4j.setDataType(DataType.FLOAT);
    }

    @AfterEach
    public void shutUp() {
        Nd4j.setDataType(initialType);
    }

    /**
     * Testing basic assign
     */
    /*
    @Test(expected = ND4JIllegalStateException.class)
    public void testOpValidation1() {
        INDArray x = Nd4j.create(10);

        Nd4j.setDataType(DataType.DOUBLE);

        INDArray y = Nd4j.create(10);

        x.addi(y);

        Nd4j.getExecutioner().commit();
    }
*/
    /**
     * Testing level1 blas
     */
    @Test()
    public void testBlasValidation1() {
       assertThrows(ND4JIllegalStateException.class,() -> {
           INDArray x = Nd4j.create(10);

           Nd4j.setDataType(DataType.DOUBLE);

           INDArray y = Nd4j.create(10);

           Nd4j.getBlasWrapper().dot(x, y);
       });

    }

    /**
     * Testing level2 blas
     */
    @Test()
    public void testBlasValidation2() {
        assertThrows(RuntimeException.class,() -> {
            INDArray a = Nd4j.create(100, 10);
            INDArray x = Nd4j.create(100);

            Nd4j.setDataType(DataType.DOUBLE);

            INDArray y = Nd4j.create(100);

            Nd4j.getBlasWrapper().gemv(1.0, a, x, 1.0, y);
        });

    }

    /**
     * Testing level3 blas
     */
    @Test()
    public void testBlasValidation3() {
       assertThrows(IllegalStateException.class,() -> {
           INDArray x = Nd4j.create(100, 100);

           Nd4j.setDataType(DataType.DOUBLE);

           INDArray y = Nd4j.create(100, 100);

           x.mmul(y);
       });

    }


    @Override
    public char ordering() {
        return 'c';
    }
}
