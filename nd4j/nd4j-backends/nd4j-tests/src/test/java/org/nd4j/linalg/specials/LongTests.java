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

package org.nd4j.linalg.specials;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Isolated;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

@Slf4j
@NativeTag
@Isolated
public class LongTests extends BaseNd4jTestWithBackends {

    DataType initialType = Nd4j.dataType();

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    public void testSomething1(Nd4jBackend backend) {
        // we create 2D array, total nr. of elements is 2.4B elements, > MAX_INT
        INDArray huge = Nd4j.create(8000000, 300);

        // we apply element-wise scalar ops, just to make sure stuff still works
        huge.subi(0.5).divi(2);


        // now we're checking different rows, they should NOT equal
        INDArray row0 = huge.getRow(100001).assign(1.0);
        INDArray row1 = huge.getRow(100002).assign(2.0);
        assertNotEquals(row0, row1);


        // same idea, but this code is broken: rowA and rowB will be pointing to the same offset
        INDArray rowA = huge.getRow(huge.rows() - 3);
        INDArray rowB = huge.getRow(huge.rows() - 10);

        // safety check, to see if we're really working on the same offset.
        rowA.addi(1.0);

        // and this fails, so rowA and rowB are pointing to the same offset, despite different getRow() arguments were used
        assertNotEquals(rowA, rowB);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    public void testSomething2(Nd4jBackend backend) {
        // we create 2D array, total nr. of elements is 2.4B elements, > MAX_INT
        INDArray huge = Nd4j.create(100, 10);

        // we apply element-wise scalar ops, just to make sure stuff still works
        huge.subi(0.5).divi(2);


        // now we're checking different rows, they should NOT equal
        INDArray row0 = huge.getRow(73).assign(1.0);
        INDArray row1 = huge.getRow(74).assign(2.0);
        assertNotEquals(row0, row1);


        // same idea, but this code is broken: rowA and rowB will be pointing to the same offset
        INDArray rowA = huge.getRow(huge.rows() - 3);
        INDArray rowB = huge.getRow(huge.rows() - 10);

        // safety check, to see if we're really working on the same offset.
        rowA.addi(1.0);

        // and this fails, so rowA and rowB are pointing to the same offset, despite different getRow() arguments were used
        assertNotEquals(rowA, rowB);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    public void testLongTadOffsets1(Nd4jBackend backend) {
        INDArray huge = Nd4j.create(230000000, 10);

        Pair<DataBuffer, DataBuffer> tad = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(huge, 1);

        assertEquals(230000000, tad.getSecond().length());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    public void testLongTadOp1(Nd4jBackend backend) {

        double exp = Transforms.manhattanDistance(Nd4j.create(1000).assign(1.0), Nd4j.create(1000).assign(2.0));

        INDArray hugeX = Nd4j.create(2200000, 1000).assign(1.0);
        INDArray hugeY = Nd4j.create(1, 1000).assign(2.0);

        for (int x = 0; x < hugeX.rows(); x++) {
            assertEquals(1000, hugeX.getRow(x).sumNumber().intValue(),"Failed at row " + x);
        }

        INDArray result = Nd4j.getExecutioner().exec(new ManhattanDistance(hugeX, hugeY, 1));
        for (int x = 0; x < hugeX.rows(); x++) {
            assertEquals(exp, result.getDouble(x), 1e-5);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    public void testLongTadOp2(Nd4jBackend backend) {

        INDArray hugeX = Nd4j.create(2300000, 1000).assign(1.0);
        hugeX.addiRowVector(Nd4j.create(1000).assign(2.0));

        for (int x = 0; x < hugeX.rows(); x++) {
            assertEquals( hugeX.getRow(x).sumNumber().intValue(),3000,"Failed at row " + x);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    public void testLongTadOp2_micro(Nd4jBackend backend) {

        INDArray hugeX = Nd4j.create(230, 1000).assign(1.0);
        hugeX.addiRowVector(Nd4j.create(1000).assign(2.0));

        for (int x = 0; x < hugeX.rows(); x++) {
            assertEquals( 3000, hugeX.getRow(x).sumNumber().intValue(),"Failed at row " + x);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    public void testLongTadOp3(Nd4jBackend backend) {

        INDArray hugeX = Nd4j.create(2300000, 1000).assign(1.0);
        INDArray mean = hugeX.mean(1);

        for (int x = 0; x < hugeX.rows(); x++) {
            assertEquals( 1.0, mean.getDouble(x), 1e-5,"Failed at row " + x);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    public void testLongTadOp4(Nd4jBackend backend) {

        INDArray hugeX = Nd4j.create(2300000, 1000).assign(1.0);
        INDArray mean = hugeX.argMax(1);

        for (int x = 0; x < hugeX.rows(); x++) {
            assertEquals(0.0, mean.getDouble(x), 1e-5,"Failed at row " + x);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    public void testLongTadOp5(Nd4jBackend backend) {

        List<INDArray> list = new ArrayList<>();
        for (int i = 0; i < 2300000; i++) {
            list.add(Nd4j.create(1000).assign(2.0));
        }

        INDArray hugeX = Nd4j.vstack(list);

        for (int x = 0; x < hugeX.rows(); x++) {
            assertEquals(2.0, hugeX.getRow(x).meanNumber().doubleValue(), 1e-5,"Failed at row " + x);
        }
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
