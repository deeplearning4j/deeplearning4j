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

package org.nd4j.imports.tfgraphs;

import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class CustomOpTests extends BaseNd4jTestWithBackends {


    @Override
    public char ordering(){
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPad(Nd4jBackend backend) {

        INDArray in = Nd4j.create(DataType.FLOAT, 1, 28, 28, 264);
        INDArray pad = Nd4j.createFromArray(new int[][]{{0,0},{0,1},{0,1},{0,0}});
        INDArray out = Nd4j.create(DataType.FLOAT, 1, 29, 29, 264);

        DynamicCustomOp op = DynamicCustomOp.builder("pad")
                .addInputs(in, pad)
                .addOutputs(out)
                .addIntegerArguments(0) //constant mode, with no constant specified
                .build();

        val outShape = Nd4j.getExecutioner().calculateOutputShape(op);
        assertEquals(1, outShape.size());
        assertArrayEquals(new long[]{1, 29, 29, 264}, outShape.get(0).getShape());

        Nd4j.getExecutioner().exec(op); //Crash here
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testResizeBilinearEdgeCase(Nd4jBackend backend){
        INDArray in = Nd4j.ones(DataType.FLOAT, 1, 1, 1, 3);
        INDArray size = Nd4j.createFromArray(8, 8);
        INDArray out = Nd4j.create(DataType.FLOAT, 1, 8, 8, 3);

        DynamicCustomOp op = DynamicCustomOp.builder("resize_bilinear")
                .addInputs(in, size)
                .addOutputs(out)
                .addIntegerArguments(1) //1 = center. Though TF works with align_corners == false or true
                .build();

        Nd4j.getExecutioner().exec(op);

        INDArray exp = Nd4j.ones(DataType.FLOAT, 1, 8, 8, 3);
        assertEquals(exp, out);
    }
}
