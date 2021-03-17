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

package org.nd4j.imports;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

public class TestReverse extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverse(Nd4jBackend backend) {

        INDArray in = Nd4j.createFromArray(new double[]{1,2,3,4,5,6});
        INDArray out = Nd4j.create(DataType.DOUBLE, 6);

        DynamicCustomOp op = DynamicCustomOp.builder("reverse")
                .addInputs(in)
                .addOutputs(out)
                .addIntegerArguments(0)
                .build();

        Nd4j.getExecutioner().exec(op);

        System.out.println(out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverse2(Nd4jBackend backend){

        INDArray in = Nd4j.createFromArray(new double[]{1,2,3,4,5,6});
        INDArray axis = Nd4j.scalar(0);
        INDArray out = Nd4j.create(DataType.DOUBLE, 6);

        DynamicCustomOp op = DynamicCustomOp.builder("reverse")
                .addInputs(in, axis)
                .addOutputs(out)
                .build();

        Nd4j.getExecutioner().exec(op);

        System.out.println(out);
    }
}
