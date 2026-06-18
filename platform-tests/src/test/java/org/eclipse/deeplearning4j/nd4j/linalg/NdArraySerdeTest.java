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

package org.eclipse.deeplearning4j.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.FilenameUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests focused on NDArray serialization and deserialization: binary read/write,
 * text serialization, flattening operations, dup/copy operations, and legacy format support.
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
public class NdArraySerdeTest extends BaseNd4jTestWithBackends {

    @TempDir
    Path testDir;

    @BeforeEach
    public void before() throws Exception {
        Nd4j.getRandom().setSeed(123);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSerialization(Nd4jBackend backend) throws Exception {
        Nd4j.getRandom().setSeed(12345);
        INDArray arr = Nd4j.rand(1, 20).castTo(DataType.DOUBLE);

        File dir = testDir.resolve("new-dir-" + UUID.randomUUID().toString()).toFile();
        assertTrue(dir.mkdirs());

        String outPath = FilenameUtils.concat(dir.getAbsolutePath(), "dl4jtestserialization.bin");

        try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(outPath)))) {
            Nd4j.write(arr, dos);
        }

        INDArray in;
        try (DataInputStream dis = new DataInputStream(new FileInputStream(outPath))) {
            in = Nd4j.read(dis);
        }

        INDArray inDup = in.dup();

        assertEquals(arr, in); //Passes:   Original array "in" is OK, but array "inDup" is not!?
        assertEquals(in, inDup); //Fails
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWriteTxt() throws Exception {
        INDArray row = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Nd4j.write(row, new DataOutputStream(bos));
        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        INDArray ret = Nd4j.read(bis);
        assertEquals(row, ret);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test2dMatrixOrderingSwitch(Nd4jBackend backend) {
        char order = Nd4j.order();
        INDArray c = Nd4j.create(new double[][]{{1, 2}, {3, 4}}, 'c');
        assertEquals('c', c.ordering());
        assertEquals(order, Nd4j.order().charValue());
        INDArray f = Nd4j.create(new double[][]{{1, 2}, {3, 4}}, 'f');
        assertEquals('f', f.ordering());
        assertEquals(order, Nd4j.order().charValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFlattenedOrder(Nd4jBackend backend) {
        INDArray concatC = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape('c', 2, 2);
        INDArray concatF = Nd4j.create(new long[]{2, 2}, 'f');
        concatF.assign(concatC);
        INDArray assertionC = Nd4j.create(new double[]{1, 2, 3, 4, 1, 2, 3, 4});
        INDArray testC = Nd4j.toFlattened('c', concatC, concatF);
        assertEquals(assertionC, testC);
        INDArray test = Nd4j.toFlattened('f', concatC, concatF);
        INDArray assertion = Nd4j.create(new double[]{1, 3, 2, 4, 1, 3, 2, 4});
        assertEquals(assertion, test);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFlattened2(Nd4jBackend backend) {
        int rows = 3;
        int cols = 4;
        int dim2 = 5;
        int dim3 = 6;

        int length2d = rows * cols;
        int length3d = rows * cols * dim2;
        int length4d = rows * cols * dim2 * dim3;

        INDArray c2d = Nd4j.linspace(1, length2d, length2d, DataType.DOUBLE).reshape('c', rows, cols);
        INDArray f2d = Nd4j.create(new long[]{rows, cols}, 'f').assign(c2d).addi(0.1);

        INDArray c3d = Nd4j.linspace(1, length3d, length3d, DataType.DOUBLE).reshape('c', rows, cols, dim2);
        INDArray f3d = Nd4j.create(new long[]{rows, cols, dim2}).assign(c3d).addi(0.3);
        c3d.addi(0.2);

        INDArray c4d = Nd4j.linspace(1, length4d, length4d, DataType.DOUBLE).reshape('c', rows, cols, dim2, dim3);
        INDArray f4d = Nd4j.create(new long[]{rows, cols, dim2, dim3}).assign(c4d).addi(0.3);
        c4d.addi(0.4);

        assertEquals(toFlattenedViaIterator('c', c2d, f2d), Nd4j.toFlattened('c', c2d, f2d));
        assertEquals(toFlattenedViaIterator('f', c2d, f2d), Nd4j.toFlattened('f', c2d, f2d));
        assertEquals(toFlattenedViaIterator('c', f2d, c2d), Nd4j.toFlattened('c', f2d, c2d));
        assertEquals(toFlattenedViaIterator('f', f2d, c2d), Nd4j.toFlattened('f', f2d, c2d));

        assertEquals(toFlattenedViaIterator('c', c3d, f3d), Nd4j.toFlattened('c', c3d, f3d));
        assertEquals(toFlattenedViaIterator('f', c3d, f3d), Nd4j.toFlattened('f', c3d, f3d));
        assertEquals(toFlattenedViaIterator('c', c2d, f2d, c3d, f3d), Nd4j.toFlattened('c', c2d, f2d, c3d, f3d));
        assertEquals(toFlattenedViaIterator('f', c2d, f2d, c3d, f3d), Nd4j.toFlattened('f', c2d, f2d, c3d, f3d));

        assertEquals(toFlattenedViaIterator('c', c4d, f4d), Nd4j.toFlattened('c', c4d, f4d));
        assertEquals(toFlattenedViaIterator('f', c4d, f4d), Nd4j.toFlattened('f', c4d, f4d));
        assertEquals(toFlattenedViaIterator('c', c2d, f2d, c3d, f3d, c4d, f4d),
                Nd4j.toFlattened('c', c2d, f2d, c3d, f3d, c4d, f4d));
        assertEquals(toFlattenedViaIterator('f', c2d, f2d, c3d, f3d, c4d, f4d),
                Nd4j.toFlattened('f', c2d, f2d, c3d, f3d, c4d, f4d));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFlattenedOnViews(Nd4jBackend backend) {
        int rows = 8;
        int cols = 8;
        int dim2 = 4;
        int length = rows * cols;
        int length3d = rows * cols * dim2;

        INDArray first = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', rows, cols);
        INDArray second = Nd4j.create(new long[]{rows, cols}, 'f').assign(first);
        INDArray third = Nd4j.linspace(1, length3d, length3d, DataType.DOUBLE).reshape('c', rows, cols, dim2);
        first.addi(0.1);
        second.addi(0.2);
        third.addi(0.3);

        first = first.get(NDArrayIndex.interval(4, 8), NDArrayIndex.interval(0, 2, 8));
        second = second.get(NDArrayIndex.interval(3, 7), NDArrayIndex.all());
        third = third.permute(0, 2, 1);
        INDArray noViewC = Nd4j.toFlattened('c', first.dup('c'), second.dup('c'), third.dup('c'));
        INDArray noViewF = Nd4j.toFlattened('f', first.dup('f'), second.dup('f'), third.dup('f'));

        assertEquals(noViewC, Nd4j.toFlattened('c', first, second, third));

        //val result = Nd4j.exec(new Flatten('f', first, second, third))[0];
        //assertEquals(noViewF, result);
        assertEquals(noViewF, Nd4j.toFlattened('f', first, second, third));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFlattened3(Nd4jBackend backend) {
        INDArray inC1 = Nd4j.create(new long[]{10, 100}, 'c');
        INDArray inC2 = Nd4j.create(new long[]{1, 100}, 'c');

        INDArray inF1 = Nd4j.create(new long[]{10, 100}, 'f');
        //        INDArray inF1 = Nd4j.create(new long[]{784,1000},'f');
        INDArray inF2 = Nd4j.create(new long[]{1, 100}, 'f');

        Nd4j.toFlattened('f', inF1); //ok
        Nd4j.toFlattened('f', inF2); //ok

        Nd4j.toFlattened('f', inC1); //crash
        Nd4j.toFlattened('f', inC2); //crash

        Nd4j.toFlattened('c', inF1); //crash on shape [784,1000]. infinite loop on shape [10,100]
        Nd4j.toFlattened('c', inF2); //ok

        Nd4j.toFlattened('c', inC1); //ok
        Nd4j.toFlattened('c', inC2); //ok
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFlattened(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        List<INDArray> concat = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            concat.add(arr.dup());
        }

        INDArray assertion = Nd4j.create(new double[]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, new int[]{12});
        INDArray flattened = Nd4j.toFlattened(concat).castTo(assertion.dataType());
        assertEquals(assertion, flattened);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDup(Nd4jBackend backend) {
        for (int x = 0; x < 100; x++) {
            INDArray orig = Nd4j.linspace(1, 4, 4, DataType.DOUBLE);
            INDArray dup = orig.dup();
            assertEquals(orig, dup);

            INDArray matrix = Nd4j.create(new float[]{1, 2, 3, 4}, new long[]{2, 2});
            INDArray dup2 = matrix.dup();
            assertEquals(matrix, dup2);

            INDArray row1 = matrix.getRow(1);
            INDArray dupRow = row1.dup();
            assertEquals(row1, dupRow);

            INDArray columnSorted = Nd4j.create(new float[]{2, 1, 4, 3}, new long[]{2, 2});
            INDArray dup3 = columnSorted.dup();
            assertEquals(columnSorted, dup3);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDupAndDupWithOrder(Nd4jBackend backend) {
        List<Pair<INDArray, String>> testInputs =
                NDArrayCreationUtil.getAllTestMatricesWithShape(ordering(), 4, 5, 123, DataType.DOUBLE);
        for (Pair<INDArray, String> pair : testInputs) {

            String msg = pair.getSecond();
            INDArray in = pair.getFirst();
            INDArray dup = in.dup();
            INDArray dupc = in.dup('c');
            INDArray dupf = in.dup('f');

            assertEquals(dup.ordering(), ordering());
            assertEquals(dupc.ordering(), 'c');
            assertEquals(dupf.ordering(), 'f');
            assertEquals(in, dupc, msg);
            assertEquals(in, dupf, msg);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToOffsetZeroCopy(Nd4jBackend backend) {
        List<Pair<INDArray, String>> testInputs =
                NDArrayCreationUtil.getAllTestMatricesWithShape(ordering(), 4, 5, 123, DataType.DOUBLE);

        for (int i = 0; i < testInputs.size(); i++) {
            Pair<INDArray, String> pair = testInputs.get(i);
            String msg = pair.getSecond();
            msg += "Failed on " + i;
            INDArray in = pair.getFirst();
            INDArray dup = Shape.toOffsetZeroCopy(in, ordering());
            INDArray dupc = Shape.toOffsetZeroCopy(in, 'c');
            INDArray dupf = Shape.toOffsetZeroCopy(in, 'f');
            INDArray dupany = Shape.toOffsetZeroCopyAnyOrder(in);

            assertEquals(in, dup, msg);
            assertEquals(in, dupc, msg);
            assertEquals(in, dupf, msg);
            assertEquals(dupc.ordering(), 'c', msg);
            assertEquals(dupf.ordering(), 'f', msg);
            assertEquals(in, dupany, msg);

            assertEquals(dup.offset(), 0);
            assertEquals(dupc.offset(), 0);
            assertEquals(dupf.offset(), 0);
            assertEquals(dupany.offset(), 0);
            assertEquals(dup.length(), dup.data().length());
            assertEquals(dupc.length(), dupc.data().length());
            assertEquals(dupf.length(), dupf.data().length());
            assertEquals(dupany.length(), dupany.data().length());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLegacyDeserialization_1() throws Exception {
        val f = new ClassPathResource("legacy/NDArray_javacpp.bin").getFile();

        val array = Nd4j.read(new FileInputStream(f));
        val exp = Nd4j.linspace(1, 120, 120, DataType.DOUBLE).reshape(2, 3, 4, 5);

        assertEquals(120, array.length());
        assertArrayEquals(new long[]{2, 3, 4, 5}, array.shape());
        assertEquals(exp, array);

        val bos = new ByteArrayOutputStream();
        Nd4j.write(bos, array);

        val bis = new ByteArrayInputStream(bos.toByteArray());
        val array2 = Nd4j.read(bis);

        assertEquals(exp, array2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLegacyDeserialization_2() throws Exception {
        val f = new ClassPathResource("legacy/NDArray_longshape_float.bin").getFile();

        val array = Nd4j.read(new FileInputStream(f));
        val exp = Nd4j.linspace(1, 5, 5, DataType.FLOAT).reshape(1, -1);

        assertEquals(5, array.length());
        assertArrayEquals(new long[]{1, 5}, array.shape());
        assertEquals(exp.dataType(), array.dataType());
        assertEquals(exp, array);

        val bos = new ByteArrayOutputStream();
        Nd4j.write(bos, array);

        val bis = new ByteArrayInputStream(bos.toByteArray());
        val array2 = Nd4j.read(bis);

        assertEquals(exp, array2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLegacyDeserialization_3() throws Exception {
        val f = new ClassPathResource("legacy/NDArray_longshape_double.bin").getFile();

        val array = Nd4j.read(new FileInputStream(f));
        val exp = Nd4j.linspace(1, 5, 5, DataType.DOUBLE).reshape(1, -1);

        assertEquals(5, array.length());
        assertArrayEquals(new long[]{1, 5}, array.shape());
        assertEquals(exp, array);

        val bos = new ByteArrayOutputStream();
        Nd4j.write(bos, array);

        val bis = new ByteArrayInputStream(bos.toByteArray());
        val array2 = Nd4j.read(bis);

        assertEquals(exp, array2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testType1(Nd4jBackend backend) throws IOException {
        for (int i = 0; i < 10; ++i) {
            INDArray in1 = Nd4j.rand(DataType.DOUBLE, new int[]{100, 100}).castTo(DataType.DOUBLE);
            File dir = testDir.toFile();
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(dir, "test.bin")));
            oos.writeObject(in1);

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(dir, "test.bin")));
            INDArray in2 = null;
            try {
                in2 = (INDArray) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException("Deserialization failed", e);
            }

            assertEquals(in1, in2);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testType2(Nd4jBackend backend) throws IOException {
        for (int i = 0; i < 10; ++i) {
            INDArray in1 = Nd4j.ones(DataType.UINT16);
            File dir = testDir.toFile();
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(dir, "test1.bin")));
            oos.writeObject(in1);

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(dir, "test1.bin")));
            INDArray in2 = null;
            try {
                in2 = (INDArray) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException("Deserialization failed", e);
            }

            assertEquals(in1, in2);
        }

        for (int i = 0; i < 10; ++i) {
            INDArray in1 = Nd4j.ones(DataType.UINT32);
            File dir = testDir.toFile();
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(dir, "test2.bin")));
            oos.writeObject(in1);

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(dir, "test2.bin")));
            INDArray in2 = null;
            try {
                in2 = (INDArray) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException("Deserialization failed", e);
            }

            assertEquals(in1, in2);
        }

        for (int i = 0; i < 10; ++i) {
            INDArray in1 = Nd4j.ones(DataType.UINT64);
            File dir = testDir.toFile();
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(dir, "test3.bin")));
            oos.writeObject(in1);

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(dir, "test3.bin")));
            INDArray in2 = null;
            try {
                in2 = (INDArray) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException("Deserialization failed", e);
            }

            assertEquals(in1, in2);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFromByteArray() throws IOException {
        // simple test to get rid of toByteArray and fromByteArray compiler warnings.
        INDArray x = Nd4j.arange(10);
        byte[] xb = Nd4j.toByteArray(x);
        INDArray y = Nd4j.fromByteArray(xb);
        assertEquals(x, y);
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    private static INDArray toFlattenedViaIterator(char order, INDArray... toFlatten) {
        int length = 0;
        for (INDArray i : toFlatten)
            length += i.length();

        INDArray out = Nd4j.create(length);
        int i = 0;
        for (INDArray arr : toFlatten) {
            NdIndexIterator iter = new NdIndexIterator(order, arr.shape());
            while (iter.hasNext()) {
                double next = arr.getDouble(iter.next());
                out.putScalar(i++, next);
            }
        }

        return out;
    }
}
