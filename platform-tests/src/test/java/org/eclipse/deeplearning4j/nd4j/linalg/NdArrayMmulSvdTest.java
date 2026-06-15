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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.primitives.Pair;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.blas.params.GemmParams;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Svd;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for matrix multiplication (mmul/gemm) and SVD operations extracted from Nd4jTestsC.
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
public class NdArrayMmulSvdTest extends BaseNd4jTestWithBackends {

    @BeforeEach
    public void before() throws Exception {
        Nd4j.getRandom().setSeed(123);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmulWithTranspose(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1,4,4, DataType.DOUBLE).reshape(2,2);
        INDArray arr2 = Nd4j.linspace(1,4,4, DataType.DOUBLE).reshape(2,2).transpose();
        INDArray arrTransposeAssertion = arr.transpose().mmul(arr2);
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .transposeA(true)
                .build();

        INDArray testResult = arr.mmul(arr2,mMulTranspose);
        assertEquals(arrTransposeAssertion,testResult);


        INDArray bTransposeAssertion = arr.mmul(arr2.transpose());
        mMulTranspose = MMulTranspose.builder()
                .transposeB(true)
                .build();

        INDArray bTest = arr.mmul(arr2,mMulTranspose);
        assertEquals(bTransposeAssertion,bTest);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMMul(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});

        INDArray assertion = Nd4j.create(new double[][] {{14, 32}, {32, 77}});

        INDArray test = arr.mmul(arr.transpose());
        assertEquals(assertion, test,getFailureMessage(backend));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmulOp(Nd4jBackend backend) throws Exception {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        INDArray z = Nd4j.create(2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{14, 32}, {32, 77}});
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .transposeB(true)
                .build();

        DynamicCustomOp op = new Mmul(arr, arr, z, mMulTranspose);
        Nd4j.getExecutioner().execAndReturn(op);

        assertEquals(assertion, z,getFailureMessage(backend));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowVectorGemm(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(1, 4);
        INDArray other = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(4, 4);
        INDArray result = linspace.mmul(other);
        INDArray assertion = Nd4j.create(new double[] {90, 100, 110, 120}).reshape(1, 4);
        assertEquals(assertion, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemmStrided(){

        for( val x : new int[]{5, 1}) {

            List<Pair<INDArray, String>> la = NDArrayCreationUtil.getAllTestMatricesWithShape(5, x, 12345, DataType.DOUBLE);
            List<Pair<INDArray, String>> lb = NDArrayCreationUtil.getAllTestMatricesWithShape(x, 4, 12345, DataType.DOUBLE);

            for (int i = 0; i < la.size(); i++) {
                for (int j = 0; j < lb.size(); j++) {

                    String msg = "x=" + x + ", i=" + i + ", j=" + j;

                    INDArray a = la.get(i).getFirst();
                    INDArray b = lb.get(i).getFirst();

                    INDArray result1 = Nd4j.createUninitialized(DataType.DOUBLE, new long[]{5, 4}, 'f');
                    INDArray result2 = Nd4j.createUninitialized(DataType.DOUBLE, new long[]{5, 4}, 'f');
                    INDArray result3 = Nd4j.createUninitialized(DataType.DOUBLE, new long[]{5, 4}, 'f');

                    Nd4j.gemm(a.dup('c'), b.dup('c'), result1, false, false, 1.0, 0.0);
                    Nd4j.gemm(a.dup('f'), b.dup('f'), result2, false, false, 1.0, 0.0);
                    Nd4j.gemm(a, b, result3, false, false, 1.0, 0.0);

                    assertEquals(result1, result2,msg);
                    assertEquals(result1, result3,msg);     // Fails here
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTADMMul(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        val shape = new long[] {4, 5, 7};
        INDArray arr = Nd4j.rand(shape).castTo(DataType.DOUBLE);

        INDArray tad = arr.tensorAlongDimension(0, 1, 2);
        assertArrayEquals(tad.shape(), new long[] {5, 7});


        INDArray copy = Nd4j.zeros(5, 7).assign(0.0);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 7; j++) {
                copy.putScalar(new long[] {i, j}, tad.getDouble(i, j));
            }
        }


        assertTrue(tad.equals(copy));
        tad = tad.reshape(7, 5);
        copy = copy.reshape(7, 5);
        INDArray first = Nd4j.rand(new long[] {2, 7}).castTo(DataType.DOUBLE);
        INDArray mmul = first.mmul(tad);
        INDArray mmulCopy = first.mmul(copy);

        assertEquals(mmul, mmulCopy);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTADMMulLeadingOne(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        val shape = new long[] {1, 5, 7};
        INDArray arr = Nd4j.rand(shape).castTo(DataType.DOUBLE);

        INDArray tad = arr.tensorAlongDimension(0, 1, 2);
        boolean order = Shape.cOrFortranOrder(tad.shape(), tad.stride(), 1);
        assertArrayEquals(tad.shape(), new long[] {5, 7});


        INDArray copy = Nd4j.zeros(5, 7).castTo(DataType.DOUBLE);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 7; j++) {
                copy.putScalar(new long[] {i, j}, tad.getDouble(i, j));
            }
        }

        assertTrue(tad.equals(copy));

        tad = tad.reshape(7, 5);
        copy = copy.reshape(7, 5);
        INDArray first = Nd4j.rand(new long[] {2, 7}).castTo(DataType.DOUBLE);
        INDArray mmul = first.mmul(tad);
        INDArray mmulCopy = first.mmul(copy);

        assertTrue(mmul.equals(mmulCopy));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmul(Nd4jBackend backend) {
        DataBuffer data = Nd4j.linspace(1, 10, 10, DataType.DOUBLE).data();
        INDArray n = Nd4j.create(data, new long[] {1, 10}).castTo(DataType.DOUBLE);
        INDArray transposed = n.transpose();
        assertEquals(true, n.isRowVector());
        assertEquals(true, transposed.isColumnVector());

        INDArray d = Nd4j.create(n.rows(), n.columns()).castTo(DataType.DOUBLE);
        d.setData(n.data());


        INDArray d3 = Nd4j.create(new double[] {1, 2}).reshape(2, 1);
        INDArray d4 = Nd4j.create(new double[] {3, 4}).reshape(1, 2);
        INDArray resultNDArray = d3.mmul(d4);
        INDArray result = Nd4j.create(new double[][] {{3, 4}, {6, 8}}).castTo(DataType.DOUBLE);
        assertEquals(result, resultNDArray);


        INDArray innerProduct = n.mmul(transposed);

        INDArray scalar = Nd4j.scalar(385.0).reshape(1,1);
        assertEquals(scalar, innerProduct,getFailureMessage(backend));

        INDArray outerProduct = transposed.mmul(n);
        assertEquals(true, Shape.shapeEquals(new long[] {10, 10}, outerProduct.shape()),getFailureMessage(backend));



        INDArray three = Nd4j.create(new double[] {3, 4}).castTo(DataType.DOUBLE);
        INDArray test = Nd4j.create(Nd4j.linspace(1, 30, 30, DataType.DOUBLE).data(), new long[] {3, 5, 2});
        INDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(three, sliceRow,getFailureMessage(backend));

        INDArray twoSix = Nd4j.create(new double[] {2, 6}, new long[] {2, 1}).castTo(DataType.DOUBLE);
        INDArray threeTwoSix = three.mmul(twoSix);

        INDArray sliceRowTwoSix = sliceRow.mmul(twoSix);

        assertEquals(threeTwoSix, sliceRowTwoSix);


        INDArray vectorVector = Nd4j.create(new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4,
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
                30, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 0, 4, 8, 12, 16, 20, 24, 28, 32,
                36, 40, 44, 48, 52, 56, 60, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 0, 6,
                12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63,
                70, 77, 84, 91, 98, 105, 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 0, 9,
                18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 0, 10, 20, 30, 40, 50, 60, 70, 80,
                90, 100, 110, 120, 130, 140, 150, 0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143,
                154, 165, 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 0, 13, 26, 39,
                52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 0, 14, 28, 42, 56, 70, 84, 98, 112, 126,
                140, 154, 168, 182, 196, 210, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210,
                225}, new long[] {16, 16}).castTo(DataType.DOUBLE);


        INDArray n1 = Nd4j.create(Nd4j.linspace(0, 15, 16, DataType.DOUBLE).data(), new long[] {1, 16});
        INDArray k1 = n1.transpose();

        INDArray testVectorVector = k1.mmul(n1);
        assertEquals(vectorVector, testVectorVector,getFailureMessage(backend));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMMulMatrixTimesColVector(Nd4jBackend backend) {
        //[1 1 1 1 1; 10 10 10 10 10; 100 100 100 100 100] x [1; 1; 1; 1; 1] = [5; 50; 500]
        INDArray matrix = Nd4j.ones(3, 5).castTo(DataType.DOUBLE);
        matrix.getRow(1).muli(10);
        matrix.getRow(2).muli(100);

        INDArray colVector = Nd4j.ones(5, 1).castTo(DataType.DOUBLE);
        INDArray out = matrix.mmul(colVector);

        INDArray expected = Nd4j.create(new double[] {5, 50, 500}, new long[] {3, 1});
        assertEquals(expected, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMMulMixedOrder(Nd4jBackend backend) {
        INDArray first = Nd4j.ones(5, 2).castTo(DataType.DOUBLE);
        INDArray second = Nd4j.ones(2, 3).castTo(DataType.DOUBLE);
        INDArray out = first.mmul(second);
        assertArrayEquals(out.shape(), new long[] {5, 3});
        assertTrue(out.equals(Nd4j.ones(5, 3).muli(2)));
        //Above: OK

        INDArray firstC = Nd4j.create(new long[] {5, 2}, 'c');
        INDArray secondF = Nd4j.create(new long[] {2, 3}, 'f');
        for (int i = 0; i < firstC.length(); i++)
            firstC.putScalar(i, 1.0);
        for (int i = 0; i < secondF.length(); i++)
            secondF.putScalar(i, 1.0);
        assertTrue(first.equals(firstC));
        assertTrue(second.equals(secondF));

        INDArray outCF = firstC.mmul(secondF);
        assertArrayEquals(outCF.shape(), new long[] {5, 3});
        assertEquals(outCF, Nd4j.ones(5, 3).muli(2));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmulGet(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345L);
        INDArray elevenByTwo = Nd4j.rand(new long[] {11, 2}).castTo(DataType.DOUBLE);
        INDArray twoByEight = Nd4j.rand(new long[] {2, 8}).castTo(DataType.DOUBLE);

        INDArray view = twoByEight.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        INDArray viewCopy = view.dup();
        assertTrue(view.equals(viewCopy));

        INDArray mmul1 = elevenByTwo.mmul(view);
        INDArray mmul2 = elevenByTwo.mmul(viewCopy);

        assertTrue(mmul1.equals(mmul2));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMMulRowColVectorMixedOrder(Nd4jBackend backend) {
        INDArray colVec = Nd4j.ones(5, 1).castTo(DataType.DOUBLE);
        INDArray rowVec = Nd4j.ones(1, 3).castTo(DataType.DOUBLE);
        INDArray out = colVec.mmul(rowVec);
        assertArrayEquals(out.shape(), new long[] {5, 3});
        assertTrue(out.equals(Nd4j.ones(5, 3)));
        //Above: OK

        INDArray colVectorC = Nd4j.create(new long[] {5, 1}, 'c').castTo(DataType.DOUBLE);
        INDArray rowVectorF = Nd4j.create(new long[] {1, 3}, 'f').castTo(DataType.DOUBLE);
        for (int i = 0; i < colVectorC.length(); i++)
            colVectorC.putScalar(i, 1.0);
        for (int i = 0; i < rowVectorF.length(); i++)
            rowVectorF.putScalar(i, 1.0);
        assertTrue(colVec.equals(colVectorC));
        assertTrue(rowVec.equals(rowVectorF));

        INDArray outCF = colVectorC.mmul(rowVectorF);
        assertArrayEquals(outCF.shape(), new long[] {5, 3});
        assertEquals(outCF, Nd4j.ones(5, 3));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMMulRowColVectorMixedOrderLarge(Nd4jBackend backend) {
        INDArray colVectorC = Nd4j.linspace(1, 96, 96, DataType.FLOAT).reshape('c', 96, 1);
        INDArray rowVectorF = Nd4j.linspace(1, 384, 384, DataType.FLOAT).reshape('f', 1, 384);

        boolean originalBlas = Nd4j.getEnvironment().isEnableBlas();
        INDArray expected;
        INDArray actual;
        try {
            Nd4j.getEnvironment().setEnableBlas(false);
            expected = colVectorC.mmul(rowVectorF);

            Nd4j.getEnvironment().setEnableBlas(true);
            actual = colVectorC.mmul(rowVectorF);
        } finally {
            Nd4j.getEnvironment().setEnableBlas(originalBlas);
        }

        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSvdLargeRectangularFullUV(Nd4jBackend backend) {
        INDArray input = Nd4j.randn(DataType.FLOAT, 96, 384);
        INDArray s = Nd4j.linalg().svd(input, true, true);
        assertArrayEquals(new long[] {96}, s.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSvdLargeRectangularFullUVPreallocatedCOrderV(Nd4jBackend backend) {
        INDArray input = Nd4j.randn(DataType.FLOAT, 96, 384);
        INDArray s = Nd4j.create(DataType.FLOAT, 96);
        INDArray u = Nd4j.create(DataType.FLOAT, 96, 96);
        INDArray v = Nd4j.create(DataType.FLOAT, new long[] {384, 384}, 'c');
        Nd4j.exec(new Svd(input, true, s, u, v));
        assertArrayEquals(new long[] {384, 384}, v.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSvdLargeRectangularFullUVPreallocatedFOrderV(Nd4jBackend backend) {
        INDArray input = Nd4j.randn(DataType.FLOAT, 96, 384);
        INDArray s = Nd4j.create(DataType.FLOAT, 96);
        INDArray u = Nd4j.create(DataType.FLOAT, 96, 96);
        INDArray v = Nd4j.create(DataType.FLOAT, new long[] {384, 384}, 'f');
        Nd4j.exec(new Svd(input, true, s, u, v));
        assertArrayEquals(new long[] {384, 384}, v.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSvdLargeRectangularDescriptorOutputsPreallocated(Nd4jBackend backend) {
        INDArray input = Nd4j.randn(DataType.FLOAT, 96, 384);
        List<DataBuffer> shapes = Nd4j.getExecutioner().calculateOutputShape(new Svd(input, true, true, 16));

        INDArray s = Nd4j.createFromDescriptor(shapes.get(0));
        INDArray u = Nd4j.createFromDescriptor(shapes.get(1));
        INDArray v = Nd4j.createFromDescriptor(shapes.get(2));

        INDArray manualS = Nd4j.create(DataType.FLOAT, 96);
        INDArray manualU = Nd4j.create(DataType.FLOAT, 96, 96);
        INDArray manualV = Nd4j.create(DataType.FLOAT, 384, 384);

        assertArrayEquals(manualS.shape(), s.shape());
        assertArrayEquals(manualU.shape(), u.shape());
        assertArrayEquals(manualV.shape(), v.shape());
        assertArrayEquals(manualS.stride(), s.stride());
        assertArrayEquals(manualU.stride(), u.stride());
        assertArrayEquals(manualV.stride(), v.stride());
        assertEquals(manualS.data().length(), s.data().length());
        assertEquals(manualU.data().length(), u.data().length());
        assertEquals(manualV.data().length(), v.data().length());
        assertEquals(manualS.ordering(), s.ordering());
        assertEquals(manualU.ordering(), u.ordering());
        assertEquals(manualV.ordering(), v.ordering());

        Nd4j.exec(new Svd(input, true, s, u, v));
        assertArrayEquals(new long[] {384, 384}, v.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSvdLargeRectangularDescriptorOutputsPreallocatedWithShapeOverride(Nd4jBackend backend) throws Exception {
        INDArray input = Nd4j.randn(DataType.FLOAT, 96, 384);
        List<DataBuffer> shapes = Nd4j.getExecutioner().calculateOutputShape(new Svd(input, true, true, 16));

        INDArray s = Nd4j.createFromDescriptor(shapes.get(0));
        INDArray u = Nd4j.createFromDescriptor(shapes.get(1));
        INDArray v = Nd4j.createFromDescriptor(shapes.get(2));
        Svd op = new Svd(input, true, s, u, v);

        try (OpContext ctx = Nd4j.getExecutioner().buildContext()) {
            op.setupOpContextFromCustomOp(ctx);
            ctx.shapeFunctionOverride(true);
            Nd4j.getExecutioner().exec(op, ctx);
        }

        assertArrayEquals(new long[] {384, 384}, v.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSvdLargeRectangularManualOutputsPreallocatedWithShapeOverride(Nd4jBackend backend) throws Exception {
        INDArray input = Nd4j.randn(DataType.FLOAT, 96, 384);
        INDArray s = Nd4j.create(DataType.FLOAT, 96);
        INDArray u = Nd4j.create(DataType.FLOAT, 96, 96);
        INDArray v = Nd4j.create(DataType.FLOAT, 384, 384);
        Svd op = new Svd(input, true, s, u, v);

        try (OpContext ctx = Nd4j.getExecutioner().buildContext()) {
            op.setupOpContextFromCustomOp(ctx);
            ctx.shapeFunctionOverride(true);
            Nd4j.getExecutioner().exec(op, ctx);
        }

        assertArrayEquals(new long[] {384, 384}, v.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSvdLargeRectangularAutoOutputsReusedOpFreshContext(Nd4jBackend backend) throws Exception {
        INDArray input = Nd4j.randn(DataType.FLOAT, 96, 384);
        Svd op = new Svd(input, true, true, 16);
        boolean shapeOverride;

        try (OpContext shapeCtx = Nd4j.getExecutioner().buildContext()) {
            op.setupOpContextFromCustomOp(shapeCtx);
            shapeOverride = op.initializeOutputs(shapeCtx);
        }

        try (OpContext execCtx = Nd4j.getExecutioner().buildContext()) {
            DefaultOpExecutioner.initOpContext(op, shapeOverride, execCtx);
            Nd4j.getExecutioner().exec(op, execCtx);
        }

        assertArrayEquals(new long[] {384, 384}, op.getOutputArgument(2).shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSvdLargeRectangularLateOutputBindingFreshOp(Nd4jBackend backend) throws Exception {
        INDArray input = Nd4j.randn(DataType.FLOAT, 96, 384);
        INDArray s = Nd4j.create(DataType.FLOAT, 96);
        INDArray u = Nd4j.create(DataType.FLOAT, 96, 96);
        INDArray v = Nd4j.create(DataType.FLOAT, 384, 384);
        Svd op = new Svd(input, true, true, 16);
        op.addOutputArgument(s, u, v);

        try (OpContext ctx = Nd4j.getExecutioner().buildContext()) {
            DefaultOpExecutioner.initOpContext(op, true, ctx);
            Nd4j.getExecutioner().exec(op, ctx);
        }

        assertArrayEquals(new long[] {384, 384}, v.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSvdLargeRectangularFullUVAutoOutputsRawOp(Nd4jBackend backend) {
        INDArray input = Nd4j.randn(DataType.FLOAT, 96, 384);
        INDArray[] out = Nd4j.exec(new Svd(input, true, true, 16));
        try {
            assertArrayEquals(new long[] {96}, out[0].shape());
            assertArrayEquals(new long[] {96, 96}, out[1].shape());
            assertArrayEquals(new long[] {384, 384}, out[2].shape());
        } finally {
            out[1].close();
            out[2].close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMMulFTimesC(Nd4jBackend backend) {
        int nRows = 3;
        int nCols = 3;
        Random r = new Random(12345);

        INDArray arrC = Nd4j.create(new long[] {nRows, nCols}, 'c').castTo(DataType.DOUBLE);
        INDArray arrF = Nd4j.create(new long[] {nRows, nCols}, 'f').castTo(DataType.DOUBLE);
        INDArray arrC2 = Nd4j.create(new long[] {nRows, nCols}, 'c').castTo(DataType.DOUBLE);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                double rv = r.nextDouble();
                arrC.putScalar(new long[] {i, j}, rv);
                arrF.putScalar(new long[] {i, j}, rv);
                arrC2.putScalar(new long[] {i, j}, r.nextDouble());
            }
        }
        assertTrue(arrF.equals(arrC));

        INDArray fTimesC = arrF.mmul(arrC2);
        INDArray cTimesC = arrC.mmul(arrC2);

        assertEquals(fTimesC, cTimesC);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMMulColVectorRowVectorMixedOrder(Nd4jBackend backend) {
        INDArray colVec = Nd4j.ones(5, 1).castTo(DataType.DOUBLE);
        INDArray rowVec = Nd4j.ones(1, 5).castTo(DataType.DOUBLE);
        INDArray out = rowVec.mmul(colVec);
        assertArrayEquals(new long[] {1, 1}, out.shape());
        assertTrue(out.equals(Nd4j.ones(1, 1).muli(5)));

        INDArray colVectorC = Nd4j.create(new long[] {5, 1}, 'c');
        INDArray rowVectorF = Nd4j.create(new long[] {1, 5}, 'f');
        for (int i = 0; i < colVectorC.length(); i++)
            colVectorC.putScalar(i, 1.0);
        for (int i = 0; i < rowVectorF.length(); i++)
            rowVectorF.putScalar(i, 1.0);
        assertTrue(colVec.equals(colVectorC));
        assertTrue(rowVec.equals(rowVectorF));

        INDArray outCF = rowVectorF.mmul(colVectorC);
        assertArrayEquals(outCF.shape(), new long[] {1, 1});
        assertTrue(outCF.equals(Nd4j.ones(1, 1).muli(5)));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void mmulToScalar(Nd4jBackend backend) {
        final INDArray arr1 = Nd4j.create(new float[] {1,2,3}).reshape(1,3);
        final INDArray arr2 = arr1.reshape(3,1);
        assertEquals( DataType.FLOAT, arr1.mmul(arr2).dataType(),"Incorrect type!");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatMul1(Nd4jBackend backend) {
        val x = 2;
        val A1 = 3;
        val A2 = 4;
        val B1 = 4;
        val B2 = 3;

        val a = Nd4j.linspace(1, x * A1 * A2, x * A1 * A2, DataType.DOUBLE).reshape(x, A1, A2);
        val b = Nd4j.linspace(1, x * B1 * B2, x * B1 * B2, DataType.DOUBLE).reshape(x, B1, B2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemmStrides(Nd4jBackend backend) {
        // 4x5 matrix from arange(20)
        final INDArray X = Nd4j.arange(20).reshape(4,5);
        for (int i=0; i<5; i++){
            // Get i-th column vector
            final INDArray xi = X.get(NDArrayIndex.all(), NDArrayIndex.point(i));
            // Build outer product
            val trans = xi;
            final INDArray outerProduct = xi.mmul(trans);
            // Build outer product from duplicated column vectors
            final INDArray outerProductDuped = xi.dup().mmul(xi.dup());
            // Matrices should equal
            assertEquals(outerProductDuped, outerProduct);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmul_128by256(Nd4jBackend backend) {
        val mA = Nd4j.create(128, 156).assign(1.0f);
        val mB = Nd4j.create(156, 256).assign(1.0f);

        val mC = Nd4j.create(128, 256);
        val mE = Nd4j.create(128, 256).assign(156.0f);
        val mL = mA.mmul(mB);

        val op = DynamicCustomOp.builder("matmul")
                .addInputs(mA, mB)
                .addOutputs(mC)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(mE, mC);
    }

    /*
        Analog of this TF code:
         a = tf.constant([], shape=[0,1])
         b = tf.constant([], shape=[1, 0])
         c = tf.matmul(a, b)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmul_Empty(Nd4jBackend backend) {
        val mA = Nd4j.create(0,1);
        val mB = Nd4j.create(1,0);
        val mC = Nd4j.create(0,0);

        val op = DynamicCustomOp.builder("matmul")
                .addInputs(mA, mB)
                .addOutputs(mC)
                .build();

        Nd4j.getExecutioner().exec(op);
        assertEquals(Nd4j.create(0,0), mC);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmul_Empty1(Nd4jBackend backend) {
        val mA = Nd4j.create(1,0, 4);
        val mB = Nd4j.create(1,4, 0);
        val mC = Nd4j.create(1,0, 0);

        val op = DynamicCustomOp.builder("mmul")
                .addInputs(mA, mB)
                .addOutputs(mC)
                .addIntegerArguments(0,0)
                .build();

        Nd4j.getExecutioner().exec(op);
        assertEquals(Nd4j.create(1,0,0), mC);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmulViews_1(Nd4jBackend backend) {
        val arrayX = Nd4j.linspace(1, 27, 27, DataType.DOUBLE).reshape(3, 3, 3);

        val arrayA = Nd4j.linspace(1, 9, 9, DataType.DOUBLE).reshape(3, 3);

        val arrayB = arrayX.dup('f');

        val arraya = arrayX.slice(0);
        val arrayb = arrayB.slice(0);

        val exp = arrayA.mmul(arrayA);

        assertEquals(exp, arraya.mmul(arrayA));
        assertEquals(exp, arraya.mmul(arraya));

        assertEquals(exp, arrayb.mmul(arrayb));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSomething_1(Nd4jBackend backend) {
        val arrayX = Nd4j.create(128, 128, 'f');
        val arrayY = Nd4j.create(128, 128, 'f');
        val arrayZ = Nd4j.create(128, 128, 'f');

        int iterations = 100;
        // warmup
        for (int e = 0; e < 10; e++)
            arrayX.addi(arrayY);

        for (int e = 0; e < iterations; e++) {
            val c = new GemmParams(arrayX, arrayY, arrayZ);
        }

        val tS = System.nanoTime();
        for (int e = 0; e < iterations; e++) {
            arrayX.mmuli(arrayY, arrayZ);
        }

        val tE = System.nanoTime();

        log.info("Average time: {}", ((tE - tS) / iterations));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testINDArrayMmulWithTranspose(){
        Nd4j.getRandom().setSeed(12345);
        INDArray a = Nd4j.rand(2,5).castTo(DataType.DOUBLE);
        INDArray b = Nd4j.rand(5,3).castTo(DataType.DOUBLE);
        INDArray exp = a.mmul(b);
        Nd4j.getExecutioner().commit();

        exp = exp.transpose();

        INDArray act = a.mmul(b, MMulTranspose.builder().transposeResult(true).build());

        assertEquals(exp, act);

        a = Nd4j.rand(5,2).castTo(DataType.DOUBLE);
        b = Nd4j.rand(5,3).castTo(DataType.DOUBLE);
        exp = a.transpose().mmul(b);
        act = a.mmul(b, MMulTranspose.builder().transposeA(true).build());
        assertEquals(exp, act);

        a = Nd4j.rand(2,5).castTo(DataType.DOUBLE);
        b = Nd4j.rand(3,5).castTo(DataType.DOUBLE);
        exp = a.mmul(b.transpose());
        act = a.mmul(b, MMulTranspose.builder().transposeB(true).build());
        assertEquals(exp, act);

        a = Nd4j.rand(5,2).castTo(DataType.DOUBLE);
        b = Nd4j.rand(3,5).castTo(DataType.DOUBLE);
        exp = a.transpose().mmul(b.transpose());
        act = a.mmul(b, MMulTranspose.builder().transposeA(true).transposeB(true).build());
        assertEquals(exp, act);

        a = Nd4j.rand(5,2).castTo(DataType.DOUBLE);
        b = Nd4j.rand(3,5).castTo(DataType.DOUBLE);
        exp = a.transpose().mmul(b.transpose()).transpose();
        act = a.mmul(b, MMulTranspose.builder().transposeA(true).transposeB(true).transposeResult(true).build());
        assertEquals(exp, act);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv2DWeightsFormat1(Nd4jBackend backend) {
        int bS = 2, iH = 4, iW = 3, iC = 4, oC = 3, kH = 3, kW = 2, sH = 1, sW = 1, pH = 0, pW = 0, dH = 1, dW = 1;
        int       oH=2,oW=2;
        // Weights format tip :
        // 0 - kH, kW, iC, oC
        // 1 - oC, iC, kH, kW
        // 2 - oC, kH, kW, iC
        WeightsFormat format = WeightsFormat.OIYX;

        INDArray inArr = Nd4j.linspace(DataType.FLOAT, 25, -0.5, 96).reshape(new long[]{bS, iC, iH, iW});
        INDArray weights = Nd4j.createFromArray(new float[]{
                        -3.f, -1.8f, -0.6f, 0.6f, 1.8f, 3.f, -2.7f, -1.5f, -0.3f, 0.9f, 2.1f, 3.3f, -2.4f, -1.2f, 0.f, 1.2f, 2.4f, 3.6f, -2.1f, -0.9f, 0.3f, 1.5f,
                        2.7f, 3.9f, -2.9f, -1.7f, -0.5f, 0.7f, 1.9f, 3.1f, -2.6f, -1.4f, -0.2f, 1.f, 2.2f, 3.4f, -2.3f, -1.1f, 0.1f, 1.3f, 2.5f, 3.7f, -2.f, -0.8f, 0.4f, 1.6f,
                        2.8f, 4.f, -2.8f, -1.6f, -0.4f, 0.8f, 2.f, 3.2f, -2.5f, -1.3f, -0.1f, 1.1f, 2.3f, 3.5f, -2.2f, -1.f, 0.2f, 1.4f, 2.6f, 3.8f, -1.9f, -0.7f, 0.5f, 1.7f, 2.9f, 4.1f}).
                reshape(new long[]{oC, iC, kH, kW});

        INDArray bias = Nd4j.createFromArray(new float[]{-1, 2, 0.5f});

        Conv2DConfig c = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(pH).pW(pW)
                .sH(sH).sW(sW)
                .dH(dH).dW(dW)
                .paddingMode(PaddingMode.VALID)
                .weightsFormat(format)
                .build();

        INDArray[] ret = Nd4j.exec(new Conv2D(inArr, weights, bias, c));
        assertArrayEquals(new long[]{bS, oC, oH, oW}, ret[0].shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv2DWeightsFormat2(Nd4jBackend backend) {
        int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
        int oH=4,oW=3;
        WeightsFormat format = WeightsFormat.OYXI;

        INDArray inArr = Nd4j.linspace(DataType.FLOAT, 25, -0.5, 96).reshape(new long[]{bS, iH, iW, iC});
        INDArray weights = Nd4j.createFromArray(new float[]{
                        -3.f, -1.8f, -0.6f, 0.6f, 1.8f, 3.f, -2.7f, -1.5f, -0.3f, 0.9f, 2.1f, 3.3f, -2.4f, -1.2f, 0.f, 1.2f, 2.4f, 3.6f, -2.1f, -0.9f, 0.3f, 1.5f,
                        2.7f, 3.9f, -2.9f, -1.7f, -0.5f, 0.7f, 1.9f, 3.1f, -2.6f, -1.4f, -0.2f, 1.f, 2.2f, 3.4f, -2.3f, -1.1f, 0.1f, 1.3f, 2.5f, 3.7f, -2.f, -0.8f, 0.4f, 1.6f,
                        2.8f, 4.f, -2.8f, -1.6f, -0.4f, 0.8f, 2.f, 3.2f, -2.5f, -1.3f, -0.1f, 1.1f, 2.3f, 3.5f, -2.2f, -1.f, 0.2f, 1.4f, 2.6f, 3.8f, -1.9f, -0.7f, 0.5f, 1.7f, 2.9f, 4.1f}).
                reshape(new long[]{oC, kH, kW, iC});

        INDArray bias = Nd4j.createFromArray(new float[]{-1, 2, 0.5f});

        Conv2DConfig c = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(pH).pW(pW)
                .sH(sH).sW(sW)
                .dH(dH).dW(dW)
                .paddingMode(PaddingMode.SAME)
                .dataFormat("NHWC")
                .weightsFormat(format)
                .build();

        INDArray[] ret = Nd4j.exec(new Conv2D(inArr, weights, bias, c));
        System.out.println(java.util.Arrays.toString(ret[0].shape()));
        assertArrayEquals(new long[]{bS, oH, oW, oC}, ret[0].shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmulMethod_8(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.INT8, 3, 5).assign(1);
        val y = Nd4j.create(DataType.INT8, 5, 3).assign(1);
        val e = Nd4j.create(DataType.INT8, 3, 3).assign(5);

        val z = x.mmul(y);
        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmulMethod_7(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.INT16, 3, 5).assign(1);
        val y = Nd4j.create(DataType.INT16, 5, 3).assign(1);
        val e = Nd4j.create(DataType.INT16, 3, 3).assign(5);

        val z = x.mmul(y);
        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmulMethod_1(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.INT32, 3, 5).assign(1);
        val y = Nd4j.create(DataType.INT32, 5, 3).assign(1);
        val e = Nd4j.create(DataType.INT32, 3, 3).assign(5);

        val z = x.mmul(y);
        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmulMethod_2(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.INT64, 3, 5).assign(1);
        val y = Nd4j.create(DataType.INT64, 5, 3).assign(1);
        val e = Nd4j.create(DataType.INT64, 3, 3).assign(5);

        val z = x.mmul(y);
        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmulMethod_6(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.UINT8, 3, 5).assign(1);
        val y = Nd4j.create(DataType.UINT8, 5, 3).assign(1);
        val e = Nd4j.create(DataType.UINT8, 3, 3).assign(5);

        val z = x.mmul(y);
        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmulMethod_5(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.UINT16, 3, 5).assign(1);
        val y = Nd4j.create(DataType.UINT16, 5, 3).assign(1);
        val e = Nd4j.create(DataType.UINT16, 3, 3).assign(5);

        val z = x.mmul(y);
        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmulMethod_3(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.UINT32, 3, 5).assign(1);
        val y = Nd4j.create(DataType.UINT32, 5, 3).assign(1);
        val e = Nd4j.create(DataType.UINT32, 3, 3).assign(5);

        val z = x.mmul(y);
        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmulMethod_4(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.UINT64, 3, 5).assign(1);
        val y = Nd4j.create(DataType.UINT64, 5, 3).assign(1);
        val e = Nd4j.create(DataType.UINT64, 3, 3).assign(5);

        val z = x.mmul(y);
        assertEquals(e, z);
    }
}
