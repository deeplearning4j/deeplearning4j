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

package org.nd4j.linalg;


import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.blas.params.GemmParams;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.LogSumExp;
import org.nd4j.linalg.api.ops.impl.accum.Mmul;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Im2col;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.iter.INDArrayIterator;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.*;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMin;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Im2col;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.Set;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.MathUtils;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static org.junit.Assert.*;

/**
 * NDArrayTests
 *
 * @author Adam Gibson
 */
@Slf4j
@RunWith(Parameterized.class)
public class Nd4jTestsC extends BaseNd4jTest {

    DataBuffer.Type initialType;

    public Nd4jTestsC(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }


    @Before
    public void before() throws Exception {
        super.before();
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Nd4j.getRandom().setSeed(123);

    }

    @After
    public void after() throws Exception {
        super.after();
        Nd4j.setDataType(initialType);
    }



    @Test
    public void testArangeNegative() {
      INDArray arr = Nd4j.arange(-2,2);
      INDArray assertion = Nd4j.create(new double[]{-2, -1,  0,  1});
      assertEquals(assertion,arr);
    }

    @Test
    public void testTri() {
       INDArray assertion = Nd4j.create(new double[][]{
               {1,1,1,0,0},
               {1,1,1,1,0},
               {1,1,1,1,1}
       });

       INDArray tri = Nd4j.tri(3,5,2);
       assertEquals(assertion,tri);
    }


    @Test
    public void testTriu() {
        INDArray input = Nd4j.linspace(1,12,12).reshape(4,3);
        int k = -1;
        INDArray test = Nd4j.triu(input,k);
        INDArray create = Nd4j.create(new double[][]{
                {1,2,3},
                {4,5,6},
                {0,8,9},
                {0,0,12}
        });

        assertEquals(test,create);
    }

    @Test
    public void testDiag() {
      INDArray diag = Nd4j.diag(Nd4j.linspace(1,4,4).reshape(4,1));
      assertArrayEquals(new long[] {4,4},diag.shape());

    }

    @Test
    public void testSoftmaxDerivativeGradient() {
        INDArray input = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray inputDup = input.dup();
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftMaxDerivative(input,Nd4j.ones(2,2),input));
        Nd4j.getExecutioner().exec(new SoftMaxDerivative(inputDup));
        assertEquals(input,inputDup);
    }

    @Test
    public void testGetRowEdgeCase() {

        INDArray orig = Nd4j.linspace(1,300,300).reshape('c', 100, 3);
        INDArray col = orig.getColumn(0);

        for( int i = 0; i < 100; i++) {
            INDArray row = col.getRow(i);
            INDArray rowDup = row.dup();
            double d = orig.getDouble(i,0);
            double d2 = col.getDouble(i, 0);
            double dRowDup = rowDup.getDouble(0);
            double dRow = row.getDouble(0);

            String s = String.valueOf(i);
            assertEquals(s, d, d2, 0.0);
            assertEquals(s, d, dRowDup, 0.0);   //Fails
            assertEquals(s, d, dRow, 0.0);      //Fails
        }
    }

    @Test
    public void testNd4jEnvironment() {
        System.out.println(Nd4j.getExecutioner().getEnvironmentInformation());
        int manualNumCores = Integer.parseInt(Nd4j.getExecutioner().getEnvironmentInformation()
                .get(Nd4jEnvironment.CPU_CORES_KEY).toString());
        assertEquals(Runtime.getRuntime().availableProcessors(), manualNumCores);
        assertEquals(Runtime.getRuntime().availableProcessors(), Nd4jEnvironment.getEnvironment().getNumCores());
        System.out.println(Nd4jEnvironment.getEnvironment());
    }

    @Test
    public void testSerialization() throws Exception {
        Nd4j.getRandom().setSeed(12345);
        INDArray arr = Nd4j.rand(1, 20);

        String temp = System.getProperty("java.io.tmpdir");

        String outPath = FilenameUtils.concat(temp, "dl4jtestserialization.bin");

        try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(outPath)))) {
            Nd4j.write(arr, dos);
        }

        INDArray in;
        try (DataInputStream dis = new DataInputStream(new FileInputStream(outPath))) {
            in = Nd4j.read(dis);
        }

        INDArray inDup = in.dup();

        System.out.println(in);
        System.out.println(inDup);

        assertEquals(arr, in); //Passes:   Original array "in" is OK, but array "inDup" is not!?
        assertEquals(in, inDup); //Fails
    }

    @Test
    public void testTensorAlongDimension2() {
        INDArray array = Nd4j.create(new float[100], new long[] {50, 1, 2});
        assertArrayEquals(new long[] {1, 2}, array.slice(0, 0).shape());

    }

    @Ignore // with broadcastables mechanic it'll be ok
    @Test(expected = IllegalStateException.class)
    public void testShapeEqualsOnElementWise() {
        Nd4j.ones(10000, 1).sub(Nd4j.ones(1, 2));
    }

    @Test
    public void testIsMax() {
        INDArray arr = Nd4j.create(new double[] {1, 2, 4, 3}, new long[] {2, 2});
        INDArray assertion = Nd4j.create(new double[] {0, 0, 1, 0}, new long[] {2, 2});
        INDArray test = Nd4j.getExecutioner().exec(new IsMax(arr)).z();
        assertEquals(assertion, test);
    }

    @Test
    public void testArgMax() {
        INDArray toArgMax = Nd4j.linspace(1, 24, 24).reshape(4, 3, 2);
        INDArray argMaxZero = Nd4j.argMax(toArgMax, 0);
        INDArray argMax = Nd4j.argMax(toArgMax, 1);
        INDArray argMaxTwo = Nd4j.argMax(toArgMax, 2);
        INDArray valueArray = Nd4j.valueArrayOf(new long[] {4, 2}, 2.0);
        INDArray valueArrayTwo = Nd4j.valueArrayOf(new long[] {3, 2}, 3.0);
        INDArray valueArrayThree = Nd4j.valueArrayOf(new long[] {4, 3}, 1.0);
        assertEquals(valueArrayTwo, argMaxZero);
        assertEquals(valueArray, argMax);

        assertEquals(valueArrayThree, argMaxTwo);
    }


    @Test
    public void testAutoBroadcastShape() {
        val assertion = new long[]{2,2,2,5};
        val shapeTest = Shape.broadcastOutputShape(new long[]{2,1,2,1},new long[]{2,1,5});
        assertArrayEquals(assertion,shapeTest);
    }

    @Test
    @Ignore //temporary till libnd4j implements general broadcasting
    public void testAutoBroadcastAdd() {
        INDArray left = Nd4j.linspace(1,4,4).reshape(2,1,2,1);
        INDArray right = Nd4j.linspace(1,10,10).reshape(2,1,5);
        INDArray assertion = Nd4j.create(new double[]{2,3,4,5,6,3,4,5,6,7,7,8,9,10,11,8,9,10,11,12,4,5,6,7,8,5,6,7,8,9,9,10,11,12,13,10,11,12,13,14}).reshape(2,2,2,5);
        INDArray test = left.add(right);
        assertEquals(assertion,test);
    }


    @Test
    public void testAudoBroadcastAddMatrix() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray row = Nd4j.ones(2);
        INDArray assertion = arr.add(1.0);
        INDArray test = arr.add(row);
        assertEquals(assertion,test);
    }

    @Test
    public void testScalarOps() throws Exception {
        INDArray n = Nd4j.create(Nd4j.ones(27).data(), new long[] {3, 3, 3});
        assertEquals(27d, n.length(), 1e-1);
        n.checkDimensions(n.addi(Nd4j.scalar(1d)));
        n.checkDimensions(n.subi(Nd4j.scalar(1.0d)));
        n.checkDimensions(n.muli(Nd4j.scalar(1.0d)));
        n.checkDimensions(n.divi(Nd4j.scalar(1.0d)));

        n = Nd4j.create(Nd4j.ones(27).data(), new long[] {3, 3, 3});
        assertEquals(getFailureMessage(), 27, n.sumNumber().doubleValue(), 1e-1);
        INDArray a = n.slice(2);
        assertEquals(getFailureMessage(), true, Arrays.equals(new long[] {3, 3}, a.shape()));

    }


    @Test
    public void testTensorAlongDimension() {
        val shape = new long[] {4, 5, 7};
        int length = ArrayUtil.prod(shape);
        INDArray arr = Nd4j.linspace(1, length, length).reshape(shape);


        int[] dim0s = {0, 1, 2, 0, 1, 2};
        int[] dim1s = {1, 0, 0, 2, 2, 1};

        double[] sums = {1350., 1350., 1582, 1582, 630, 630};

        for (int i = 0; i < dim0s.length; i++) {
            int firstDim = dim0s[i];
            int secondDim = dim1s[i];
            INDArray tad = arr.tensorAlongDimension(0, firstDim, secondDim);
            tad.sumNumber();
            //            assertEquals("I " + i + " failed ",sums[i],tad.sumNumber().doubleValue(),1e-1);
        }

        INDArray testMem = Nd4j.create(10, 10);
    }


    @Test
    public void testMmulWithTranspose() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray arr2 = Nd4j.linspace(1,4,4).reshape(2,2).transpose();
        INDArray arrTransposeAssertion = arr.transpose().mmul(arr2);
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .transposeA(true)
                .a(arr)
                .b(arr2)
                .build();

        INDArray testResult = arr.mmul(arr2,mMulTranspose);
        assertEquals(arrTransposeAssertion,testResult);


        INDArray bTransposeAssertion = arr.mmul(arr2.transpose());
        mMulTranspose = MMulTranspose.builder()
                .transposeB(true)
                .a(arr)
                .b(arr2)
                .build();

        INDArray bTest = arr.mmul(arr2,mMulTranspose);
        assertEquals(bTransposeAssertion,bTest);
    }


    @Test
    public void testGetDouble() {
        INDArray n2 = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new long[] {3, 5, 2});
        INDArray swapped = n2.swapAxes(n2.shape().length - 1, 1);
        INDArray slice0 = swapped.slice(0).slice(1);
        INDArray assertion = Nd4j.create(new double[] {2, 4, 6, 8, 10});
        assertEquals(assertion, slice0);
    }

    @Test
    public void testWriteTxt() throws Exception {
        INDArray row = Nd4j.create(new double[][] {{1, 2}, {3, 4}});
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Nd4j.write(row, new DataOutputStream(bos));
        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        INDArray ret = Nd4j.read(bis);
        assertEquals(row, ret);

    }

    @Test
    public void test2dMatrixOrderingSwitch() throws Exception {
        char order = Nd4j.order();
        INDArray c = Nd4j.create(new double[][] {{1, 2}, {3, 4}}, 'c');
        assertEquals('c', c.ordering());
        assertEquals(order, Nd4j.order().charValue());
        INDArray f = Nd4j.create(new double[][] {{1, 2}, {3, 4}}, 'f');
        assertEquals('f', f.ordering());
        assertEquals(order, Nd4j.order().charValue());
    }


    @Test
    public void testMatrix() {
        INDArray arr = Nd4j.create(new float[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray brr = Nd4j.create(new float[] {5, 6}, new long[] {1, 2});
        INDArray row = arr.getRow(0);
        row.subi(brr);
        assertEquals(Nd4j.create(new double[] {-4, -4}), arr.getRow(0));

    }

    @Test
    @Ignore
    public void testParseComplexNumber() {
        IComplexNumber assertion = Nd4j.createComplexNumber(1, 1);
        String parse = "1 + 1i";
        IComplexNumber parsed = Nd4j.parseComplexNumber(parse);
        assertEquals(assertion, parsed);
    }



    @Test
    public void testMMul() {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});

        INDArray assertion = Nd4j.create(new double[][] {{14, 32}, {32, 77}});

        INDArray test = arr.mmul(arr.transpose());
        assertEquals(getFailureMessage(), assertion, test);
    }

    @Test
    public void testMmulOp() {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        INDArray z = Nd4j.create(2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{14, 32}, {32, 77}});
        MMulTranspose mMulTranspose = MMulTranspose.builder()
          .transposeB(true)
          .a(arr)
          .b(arr)
          .build();

        DynamicCustomOp op = new Mmul(arr, arr, z, mMulTranspose);
        Nd4j.getExecutioner().exec(op);
        
        assertEquals(getFailureMessage(), assertion, z);
    }


    @Test
    public void testSubiRowVector() throws Exception {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4).reshape('c', 2, 2);
        INDArray row1 = oneThroughFour.getRow(1);
        oneThroughFour.subiRowVector(row1);
        INDArray result = Nd4j.create(new float[] {-2, -2, 0, 0}, new long[] {2, 2});
        assertEquals(getFailureMessage(), result, oneThroughFour);

    }


    @Test
    public void testAddiRowVectorWithScalar() {
        INDArray colVector = Nd4j.create(5, 1).assign(0.0);
        INDArray scalar = Nd4j.create(1, 1).assign(0.0);
        scalar.putScalar(0, 1);

        assertEquals(scalar.getDouble(0), 1.0, 0.0);

        colVector.addiRowVector(scalar); //colVector is all zeros after this
        for (int i = 0; i < 5; i++)
            assertEquals(colVector.getDouble(i), 1.0, 0.0);
    }

    @Test
    public void testTADOnVector() {

        Nd4j.getRandom().setSeed(12345);
        INDArray rowVec = Nd4j.rand(1, 10);
        INDArray thirdElem = rowVec.tensorAlongDimension(2, 0);

        assertEquals(rowVec.getDouble(2), thirdElem.getDouble(0), 0.0);

        thirdElem.putScalar(0, 5);
        assertEquals(5, thirdElem.getDouble(0), 0.0);

        assertEquals(5, rowVec.getDouble(2), 0.0); //Both should be modified if thirdElem is a view

        //Same thing for column vector:
        INDArray colVec = Nd4j.rand(10, 1);
        thirdElem = colVec.tensorAlongDimension(2, 1);

        assertEquals(colVec.getDouble(2), thirdElem.getDouble(0), 0.0);

        thirdElem.putScalar(0, 5);
        assertEquals(5, thirdElem.getDouble(0), 0.0);
        assertEquals(5, colVec.getDouble(2), 0.0);
    }

    @Test
    public void testLength() {
        INDArray values = Nd4j.create(2, 2);
        INDArray values2 = Nd4j.create(2, 2);

        values.put(0, 0, 0);
        values2.put(0, 0, 2);
        values.put(1, 0, 0);
        values2.put(1, 0, 2);
        values.put(0, 1, 0);
        values2.put(0, 1, 0);
        values.put(1, 1, 2);
        values2.put(1, 1, 2);


        INDArray expected = Nd4j.repeat(Nd4j.scalar(2), 2).reshape(2, 1);

        Accumulation accum = Nd4j.getOpFactory().createAccum("euclidean", values, values2);
        INDArray results = Nd4j.getExecutioner().exec(accum, 1);
        assertEquals(expected, results);

    }

    @Test
    public void testBroadCasting() {
        INDArray first = Nd4j.arange(0, 3).reshape(3, 1);
        INDArray ret = first.broadcast(3, 4);
        INDArray testRet = Nd4j.create(new double[][] {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
        assertEquals(testRet, ret);
        INDArray r = Nd4j.arange(0, 4).reshape(1, 4);
        INDArray r2 = r.broadcast(4, 4);
        INDArray testR2 = Nd4j.create(new double[][] {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}});
        assertEquals(testR2, r2);

    }


    @Test
    public void testGetColumns() {
        INDArray matrix = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray matrixGet = matrix.getColumns(new int[] {1, 2});
        INDArray matrixAssertion = Nd4j.create(new double[][] {{2, 3}, {5, 6}});
        assertEquals(matrixAssertion, matrixGet);
    }

    @Test
    public void testSort() throws Exception {
        INDArray toSort = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray ascending = Nd4j.sort(toSort.dup(), 1, true);
        //rows already already sorted
        assertEquals(toSort, ascending);

        INDArray columnSorted = Nd4j.create(new float[] {2, 1, 4, 3}, new long[] {2, 2});
        INDArray sorted = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(columnSorted, sorted);
    }

    @Test
    public void testSortRows() {
        int nRows = 10;
        int nCols = 5;
        java.util.Random r = new java.util.Random(12345);

        for (int i = 0; i < nCols; i++) {
            INDArray in = Nd4j.linspace(1, nRows * nCols, nRows * nCols).reshape(nRows, nCols);

            List<Integer> order = new ArrayList<>(nRows);
            //in.row(order(i)) should end up as out.row(i) - ascending
            //in.row(order(i)) should end up as out.row(nRows-j-1) - descending
            for (int j = 0; j < nRows; j++)
                order.add(j);
            Collections.shuffle(order, r);
            for (int j = 0; j < nRows; j++)
                in.putScalar(new long[] {j, i}, order.get(j));

            INDArray outAsc = Nd4j.sortRows(in, i, true);
            INDArray outDesc = Nd4j.sortRows(in, i, false);

            System.out.println("outDesc: " + Arrays.toString(outAsc.data().asFloat()));
            for (int j = 0; j < nRows; j++) {
                assertEquals(outAsc.getDouble(j, i), j, 1e-1);
                int origRowIdxAsc = order.indexOf(j);
                assertTrue(outAsc.getRow(j).equals(in.getRow(origRowIdxAsc)));

                assertEquals((nRows - j - 1), outDesc.getDouble(j, i), 0.001f);
                int origRowIdxDesc = order.indexOf(nRows - j - 1);
                assertTrue(outDesc.getRow(j).equals(in.getRow(origRowIdxDesc)));
            }
        }
    }

    @Test
    public void testToFlattenedOrder() {
        INDArray concatC = Nd4j.linspace(1, 4, 4).reshape('c', 2, 2);
        INDArray concatF = Nd4j.create(new long[] {2, 2}, 'f');
        concatF.assign(concatC);
        INDArray assertionC = Nd4j.create(new double[] {1, 2, 3, 4, 1, 2, 3, 4});
        INDArray testC = Nd4j.toFlattened('c', concatC, concatF);
        assertEquals(assertionC, testC);
        INDArray test = Nd4j.toFlattened('f', concatC, concatF);
        INDArray assertion = Nd4j.create(new double[] {1, 3, 2, 4, 1, 3, 2, 4});
        assertEquals(assertion, test);


    }

    @Test
    public void testZero() {
        Nd4j.ones(11).sumNumber();
        Nd4j.ones(12).sumNumber();
        Nd4j.ones(2).sumNumber();
    }


    @Test
    public void testSumNumberRepeatability() {
        INDArray arr = Nd4j.ones(1, 450).reshape('c', 150, 3);

        double first = arr.sumNumber().doubleValue();
        double assertion = 450;
        assertEquals(assertion, first, 1e-1);
        for (int i = 0; i < 50; i++) {
            double second = arr.sumNumber().doubleValue();
            assertEquals(assertion, second, 1e-1);
            assertEquals(String.valueOf(i), first, second, 1e-2);
        }
    }

    @Test
    public void testToFlattened2() {
        int rows = 3;
        int cols = 4;
        int dim2 = 5;
        int dim3 = 6;

        int length2d = rows * cols;
        int length3d = rows * cols * dim2;
        int length4d = rows * cols * dim2 * dim3;

        INDArray c2d = Nd4j.linspace(1, length2d, length2d).reshape('c', rows, cols);
        INDArray f2d = Nd4j.create(new long[] {rows, cols}, 'f').assign(c2d).addi(0.1);

        INDArray c3d = Nd4j.linspace(1, length3d, length3d).reshape('c', rows, cols, dim2);
        INDArray f3d = Nd4j.create(new long[] {rows, cols, dim2}).assign(c3d).addi(0.3);
        c3d.addi(0.2);

        INDArray c4d = Nd4j.linspace(1, length4d, length4d).reshape('c', rows, cols, dim2, dim3);
        INDArray f4d = Nd4j.create(new long[] {rows, cols, dim2, dim3}).assign(c4d).addi(0.3);
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

    @Test
    public void testToFlattenedOnViews() {
        int rows = 8;
        int cols = 8;
        int dim2 = 4;
        int length = rows * cols;
        int length3d = rows * cols * dim2;

        INDArray first = Nd4j.linspace(1, length, length).reshape('c', rows, cols);
        INDArray second = Nd4j.create(new long[] {rows, cols}, 'f').assign(first);
        INDArray third = Nd4j.linspace(1, length3d, length3d).reshape('c', rows, cols, dim2);
        first.addi(0.1);
        second.addi(0.2);
        third.addi(0.3);

        first = first.get(NDArrayIndex.interval(4, 8), NDArrayIndex.interval(0, 2, 8));
        second = second.get(NDArrayIndex.interval(3, 7), NDArrayIndex.all());
        third = third.permute(0, 2, 1);
        INDArray cAssertion = Nd4j.create(new double[] {33.10, 35.10, 37.10, 39.10, 41.10, 43.10, 45.10, 47.10, 49.10,
                51.10, 53.10, 55.10, 57.10, 59.10, 61.10, 63.10, 25.20, 26.20, 27.20, 28.20, 29.20, 30.20,
                31.20, 32.20, 33.20, 34.20, 35.20, 36.20, 37.20, 38.20, 39.20, 40.20, 41.20, 42.20, 43.20,
                44.20, 45.20, 46.20, 47.20, 48.20, 49.20, 50.20, 51.20, 52.20, 53.20, 54.20, 55.20, 56.20, 1.30,
                5.30, 9.30, 13.30, 17.30, 21.30, 25.30, 29.30, 2.30, 6.30, 10.30, 14.30, 18.30, 22.30, 26.30,
                30.30, 3.30, 7.30, 11.30, 15.30, 19.30, 23.30, 27.30, 31.30, 4.30, 8.30, 12.30, 16.30, 20.30,
                24.30, 28.30, 32.30, 33.30, 37.30, 41.30, 45.30, 49.30, 53.30, 57.30, 61.30, 34.30, 38.30,
                42.30, 46.30, 50.30, 54.30, 58.30, 62.30, 35.30, 39.30, 43.30, 47.30, 51.30, 55.30, 59.30,
                63.30, 36.30, 40.30, 44.30, 48.30, 52.30, 56.30, 60.30, 64.30, 65.30, 69.30, 73.30, 77.30,
                81.30, 85.30, 89.30, 93.30, 66.30, 70.30, 74.30, 78.30, 82.30, 86.30, 90.30, 94.30, 67.30,
                71.30, 75.30, 79.30, 83.30, 87.30, 91.30, 95.30, 68.30, 72.30, 76.30, 80.30, 84.30, 88.30,
                92.30, 96.30, 97.30, 101.30, 105.30, 109.30, 113.30, 117.30, 121.30, 125.30, 98.30, 102.30,
                106.30, 110.30, 114.30, 118.30, 122.30, 126.30, 99.30, 103.30, 107.30, 111.30, 115.30, 119.30,
                123.30, 127.30, 100.30, 104.30, 108.30, 112.30, 116.30, 120.30, 124.30, 128.30, 129.30, 133.30,
                137.30, 141.30, 145.30, 149.30, 153.30, 157.30, 130.30, 134.30, 138.30, 142.30, 146.30, 150.30,
                154.30, 158.30, 131.30, 135.30, 139.30, 143.30, 147.30, 151.30, 155.30, 159.30, 132.30, 136.30,
                140.30, 144.30, 148.30, 152.30, 156.30, 160.30, 161.30, 165.30, 169.30, 173.30, 177.30, 181.30,
                185.30, 189.30, 162.30, 166.30, 170.30, 174.30, 178.30, 182.30, 186.30, 190.30, 163.30, 167.30,
                171.30, 175.30, 179.30, 183.30, 187.30, 191.30, 164.30, 168.30, 172.30, 176.30, 180.30, 184.30,
                188.30, 192.30, 193.30, 197.30, 201.30, 205.30, 209.30, 213.30, 217.30, 221.30, 194.30, 198.30,
                202.30, 206.30, 210.30, 214.30, 218.30, 222.30, 195.30, 199.30, 203.30, 207.30, 211.30, 215.30,
                219.30, 223.30, 196.30, 200.30, 204.30, 208.30, 212.30, 216.30, 220.30, 224.30, 225.30, 229.30,
                233.30, 237.30, 241.30, 245.30, 249.30, 253.30, 226.30, 230.30, 234.30, 238.30, 242.30, 246.30,
                250.30, 254.30, 227.30, 231.30, 235.30, 239.30, 243.30, 247.30, 251.30, 255.30, 228.30, 232.30,
                236.30, 240.30, 244.30, 248.30, 252.30, 256.30});
        INDArray fAssertion = Nd4j.create(new double[] {33.10, 41.10, 49.10, 57.10, 35.10, 43.10, 51.10, 59.10, 37.10,
                45.10, 53.10, 61.10, 39.10, 47.10, 55.10, 63.10, 25.20, 33.20, 41.20, 49.20, 26.20, 34.20,
                42.20, 50.20, 27.20, 35.20, 43.20, 51.20, 28.20, 36.20, 44.20, 52.20, 29.20, 37.20, 45.20,
                53.20, 30.20, 38.20, 46.20, 54.20, 31.20, 39.20, 47.20, 55.20, 32.20, 40.20, 48.20, 56.20, 1.30,
                33.30, 65.30, 97.30, 129.30, 161.30, 193.30, 225.30, 2.30, 34.30, 66.30, 98.30, 130.30, 162.30,
                194.30, 226.30, 3.30, 35.30, 67.30, 99.30, 131.30, 163.30, 195.30, 227.30, 4.30, 36.30, 68.30,
                100.30, 132.30, 164.30, 196.30, 228.30, 5.30, 37.30, 69.30, 101.30, 133.30, 165.30, 197.30,
                229.30, 6.30, 38.30, 70.30, 102.30, 134.30, 166.30, 198.30, 230.30, 7.30, 39.30, 71.30, 103.30,
                135.30, 167.30, 199.30, 231.30, 8.30, 40.30, 72.30, 104.30, 136.30, 168.30, 200.30, 232.30,
                9.30, 41.30, 73.30, 105.30, 137.30, 169.30, 201.30, 233.30, 10.30, 42.30, 74.30, 106.30, 138.30,
                170.30, 202.30, 234.30, 11.30, 43.30, 75.30, 107.30, 139.30, 171.30, 203.30, 235.30, 12.30,
                44.30, 76.30, 108.30, 140.30, 172.30, 204.30, 236.30, 13.30, 45.30, 77.30, 109.30, 141.30,
                173.30, 205.30, 237.30, 14.30, 46.30, 78.30, 110.30, 142.30, 174.30, 206.30, 238.30, 15.30,
                47.30, 79.30, 111.30, 143.30, 175.30, 207.30, 239.30, 16.30, 48.30, 80.30, 112.30, 144.30,
                176.30, 208.30, 240.30, 17.30, 49.30, 81.30, 113.30, 145.30, 177.30, 209.30, 241.30, 18.30,
                50.30, 82.30, 114.30, 146.30, 178.30, 210.30, 242.30, 19.30, 51.30, 83.30, 115.30, 147.30,
                179.30, 211.30, 243.30, 20.30, 52.30, 84.30, 116.30, 148.30, 180.30, 212.30, 244.30, 21.30,
                53.30, 85.30, 117.30, 149.30, 181.30, 213.30, 245.30, 22.30, 54.30, 86.30, 118.30, 150.30,
                182.30, 214.30, 246.30, 23.30, 55.30, 87.30, 119.30, 151.30, 183.30, 215.30, 247.30, 24.30,
                56.30, 88.30, 120.30, 152.30, 184.30, 216.30, 248.30, 25.30, 57.30, 89.30, 121.30, 153.30,
                185.30, 217.30, 249.30, 26.30, 58.30, 90.30, 122.30, 154.30, 186.30, 218.30, 250.30, 27.30,
                59.30, 91.30, 123.30, 155.30, 187.30, 219.30, 251.30, 28.30, 60.30, 92.30, 124.30, 156.30,
                188.30, 220.30, 252.30, 29.30, 61.30, 93.30, 125.30, 157.30, 189.30, 221.30, 253.30, 30.30,
                62.30, 94.30, 126.30, 158.30, 190.30, 222.30, 254.30, 31.30, 63.30, 95.30, 127.30, 159.30,
                191.30, 223.30, 255.30, 32.30, 64.30, 96.30, 128.30, 160.30, 192.30, 224.30, 256.30});
        assertEquals(cAssertion, Nd4j.toFlattened('c', first, second, third));
        assertEquals(fAssertion, Nd4j.toFlattened('f', first, second, third));
    }

    private static INDArray toFlattenedViaIterator(char order, INDArray... toFlatten) {
        int length = 0;
        for (INDArray i : toFlatten)
            length += i.length();

        INDArray out = Nd4j.create(1, length);
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



    @Test
    public void testIsMax2() {
        //Tests: full buffer...
        //1d
        INDArray arr1 = Nd4j.create(new double[] {1, 2, 3, 1});
        Nd4j.getExecutioner().execAndReturn(new IsMax(arr1));
        INDArray exp1 = Nd4j.create(new double[] {0, 0, 1, 0});

        assertEquals(exp1, arr1);

        arr1 = Nd4j.create(new double[] {1, 2, 3, 1});
        INDArray result = Nd4j.zeros(4);
        Nd4j.getExecutioner().execAndReturn(new IsMax(arr1, result));

        assertEquals(Nd4j.create(new double[] {1, 2, 3, 1}), arr1);
        assertEquals(exp1, result);

        //2d
        INDArray arr2d = Nd4j.create(new double[][] {{0, 1, 2}, {2, 9, 1}});
        INDArray exp2d = Nd4j.create(new double[][] {{0, 0, 0}, {0, 1, 0}});

        INDArray f = arr2d.dup('f');
        INDArray out2dc = Nd4j.getExecutioner().execAndReturn(new IsMax(arr2d.dup('c')));
        INDArray out2df = Nd4j.getExecutioner().execAndReturn(new IsMax(arr2d.dup('f')));
        assertEquals(exp2d, out2dc);
        assertEquals(exp2d, out2df);
    }

    @Test
    public void testToFlattened3() {
        INDArray inC1 = Nd4j.create(new long[] {10, 100}, 'c');
        INDArray inC2 = Nd4j.create(new long[] {1, 100}, 'c');

        INDArray inF1 = Nd4j.create(new long[] {10, 100}, 'f');
        //        INDArray inF1 = Nd4j.create(new long[]{784,1000},'f');
        INDArray inF2 = Nd4j.create(new long[] {1, 100}, 'f');

        Nd4j.toFlattened('f', inF1); //ok
        Nd4j.toFlattened('f', inF2); //ok

        Nd4j.toFlattened('f', inC1); //crash
        Nd4j.toFlattened('f', inC2); //crash

        Nd4j.toFlattened('c', inF1); //crash on shape [784,1000]. infinite loop on shape [10,100]
        Nd4j.toFlattened('c', inF2); //ok

        Nd4j.toFlattened('c', inC1); //ok
        Nd4j.toFlattened('c', inC2); //ok
    }

    @Test
    public void testIsMaxEqualValues() {
        //Assumption here: should only have a 1 for *first* maximum value, if multiple values are exactly equal

        //[1 1 1] -> [1 0 0]
        //Loop to double check against any threading weirdness...
        for (int i = 0; i < 10; i++) {
            assertEquals(Nd4j.create(new double[] {1, 0, 0}),
                    Nd4j.getExecutioner().execAndReturn(new IsMax(Nd4j.ones(3))));
        }

        //[0 0 0 2 2 0] -> [0 0 0 1 0 0]
        assertEquals(Nd4j.create(new double[] {0, 0, 0, 1, 0, 0}),
                Nd4j.getExecutioner().execAndReturn(new IsMax(Nd4j.create(new double[] {0, 0, 0, 2, 2, 0}))));

        //[0 2]    [0 1]
        //[2 1] -> [0 0]
        INDArray orig = Nd4j.create(new double[][] {{0, 2}, {2, 1}});
        INDArray exp = Nd4j.create(new double[][] {{0, 1}, {0, 0}});
        INDArray outc = Nd4j.getExecutioner().execAndReturn(new IsMax(orig.dup('c')));
        INDArray outf = Nd4j.getExecutioner().execAndReturn(new IsMax(orig.dup('f')));

        assertEquals(exp, outc);
        assertEquals(exp, outf);
    }

    @Test
    public void testIsMaxAlongDimension() {
        //1d: row vector
        INDArray orig = Nd4j.create(new double[] {1, 2, 3, 1});

        INDArray alongDim0 = Nd4j.getExecutioner().execAndReturn(new IsMax(orig.dup(), 0));
        INDArray alongDim1 = Nd4j.getExecutioner().execAndReturn(new IsMax(orig.dup(), 1));

        INDArray expAlong0 = Nd4j.ones(4);
        INDArray expAlong1 = Nd4j.create(new double[] {0, 0, 1, 0});

        assertEquals(expAlong0, alongDim0);
        assertEquals(expAlong1, alongDim1);

        //1d: col vector
        System.out.println("----------------------------------");
        INDArray col = Nd4j.create(new double[] {1, 2, 3, 1}, new long[] {4, 1});
        INDArray alongDim0col = Nd4j.getExecutioner().execAndReturn(new IsMax(col.dup(), 0));
        INDArray alongDim1col = Nd4j.getExecutioner().execAndReturn(new IsMax(col.dup(), 1));

        INDArray expAlong0col = Nd4j.create(new double[] {0, 0, 1, 0}, new long[] {4, 1});
        INDArray expAlong1col = Nd4j.ones(new long[] {4, 1});



        assertEquals(expAlong1col, alongDim1col);
        assertEquals(expAlong0col, alongDim0col);



        /*
        if (blockIdx.x == 0) {
            printf("original Z shape: \n");
            shape::printShapeInfoLinear(zShapeInfo);

            printf("Target dimension: [%i], dimensionLength: [%i]\n", dimension[0], dimensionLength);

            printf("TAD shape: \n");
            shape::printShapeInfoLinear(tad->tadOnlyShapeInfo);
        }
        */

        //2d:
        //[1 0 2]
        //[2 3 1]
        //Along dim 0:
        //[0 0 1]
        //[1 1 0]
        //Along dim 1:
        //[0 0 1]
        //[0 1 0]
        System.out.println("---------------------");
        INDArray orig2d = Nd4j.create(new double[][] {{1, 0, 2}, {2, 3, 1}});
        INDArray alongDim0c_2d = Nd4j.getExecutioner().execAndReturn(new IsMax(orig2d.dup('c'), 0));
        INDArray alongDim0f_2d = Nd4j.getExecutioner().execAndReturn(new IsMax(orig2d.dup('f'), 0));
        INDArray alongDim1c_2d = Nd4j.getExecutioner().execAndReturn(new IsMax(orig2d.dup('c'), 1));
        INDArray alongDim1f_2d = Nd4j.getExecutioner().execAndReturn(new IsMax(orig2d.dup('f'), 1));
        INDArray expAlong0_2d = Nd4j.create(new double[][] {{0, 0, 1}, {1, 1, 0}});
        INDArray expAlong1_2d = Nd4j.create(new double[][] {{0, 0, 1}, {0, 1, 0}});

        assertEquals(expAlong0_2d, alongDim0c_2d);
        assertEquals(expAlong0_2d, alongDim0f_2d);
        assertEquals(expAlong1_2d, alongDim1c_2d);
        assertEquals(expAlong1_2d, alongDim1f_2d);

    }

    @Test
    public void testIMaxSingleDim1() {
        INDArray orig2d = Nd4j.create(new double[][] {{1, 0, 2}, {2, 3, 1}});

        INDArray result = Nd4j.argMax(orig2d.dup('c'), 0);

        System.out.println("IMAx result: " + result);
    }

    @Test
    public void testIsMaxSingleDim1() {
        INDArray orig2d = Nd4j.create(new double[][] {{1, 0, 2}, {2, 3, 1}});
        INDArray alongDim0c_2d = Nd4j.getExecutioner().execAndReturn(new IsMax(orig2d.dup('c'), 0));
        INDArray expAlong0_2d = Nd4j.create(new double[][] {{0, 0, 1}, {1, 1, 0}});

        System.out.println("Original shapeInfo: " + orig2d.dup('c').shapeInfoDataBuffer());

        System.out.println("Expected: " + Arrays.toString(expAlong0_2d.data().asFloat()));
        System.out.println("Actual: " + Arrays.toString(alongDim0c_2d.data().asFloat()));
        assertEquals(expAlong0_2d, alongDim0c_2d);
    }

    @Test
    public void testBroadcastRepeated() {
        INDArray z = Nd4j.create(1, 4, 4, 3);
        INDArray bias = Nd4j.create(1, 3);
        BroadcastOp op = new BroadcastAddOp(z, bias, z, 3);
        Nd4j.getExecutioner().exec(op);
        System.out.println("First: OK");
        //OK at this point: executes successfully


        z = Nd4j.create(1, 4, 4, 3);
        bias = Nd4j.create(1, 3);
        op = new BroadcastAddOp(z, bias, z, 3);
        Nd4j.getExecutioner().exec(op); //Crashing here, when we are doing exactly the same thing as before...
        System.out.println("Second: OK");
    }

    @Test
    public void testTadShape() {
        INDArray arr = Nd4j.linspace(1, 12, 12).reshape(4, 3, 1, 1);
        INDArray javaTad = arr.javaTensorAlongDimension(0, 0, 2, 3);
        assertArrayEquals(new long[] {4, 1, 1}, javaTad.shape());
        INDArray tad = arr.tensorAlongDimension(0, 0, 2, 3);
        assertArrayEquals(javaTad.shapeInfoDataBuffer().asLong(), tad.shapeInfoDataBuffer().asLong());
        assertEquals(javaTad.shapeInfoDataBuffer(), tad.shapeInfoDataBuffer());
    }

    @Test
    public void testSoftmaxDerivative() {
        INDArray input = Nd4j.create(new double[] {-1.07, -0.01, 0.45, 0.95, 0.45, 0.16, 0.20, 0.80, 0.89, 0.25})
                .transpose();
        INDArray output = Nd4j.create(10, 1);
        Nd4j.getExecutioner().exec(new SoftMaxDerivative(input, output));
    }


    @Test
    public void testVStackDifferentOrders() {
        INDArray expected = Nd4j.linspace(1, 9, 9).reshape('c', 3, 3);

        for (char order : new char[] {'c', 'f'}) {
            System.out.println(order);
            Nd4j.factory().setOrder(order);

            INDArray arr1 = Nd4j.linspace(1, 6, 6).reshape('c', 2, 3);
            INDArray arr2 = Nd4j.linspace(7, 9, 3).reshape('c', 1, 3);

            INDArray merged = Nd4j.vstack(arr1, arr2);
            System.out.println(merged);
            System.out.println(expected);

            assertEquals(expected, merged);
        }
    }

    @Test
    public void testVStackEdgeCase() {
        INDArray arr = Nd4j.linspace(1, 4, 4);
        INDArray vstacked = Nd4j.vstack(arr);
        assertEquals(arr, vstacked);
    }


    @Test
    public void testEps3() {

        INDArray first = Nd4j.linspace(1, 10, 10);
        INDArray second = Nd4j.linspace(20, 30, 10);

        INDArray expAllZeros = Nd4j.getExecutioner().execAndReturn(new Eps(first, second, Nd4j.create(10), 10));
        INDArray expAllOnes = Nd4j.getExecutioner().execAndReturn(new Eps(first, first, Nd4j.create(10), 10));

        System.out.println(expAllZeros);
        System.out.println(expAllOnes);

        assertEquals(0, expAllZeros.sumNumber().doubleValue(), 0.0);
        assertEquals(10, expAllOnes.sumNumber().doubleValue(), 0.0);
    }

    @Test
    @Ignore
    public void testSumAlongDim1sEdgeCases() {
        val shapes = new long[][] {
                //Standard case:
                {2, 2, 3, 4},
                //Leading 1s:
                {1, 2, 3, 4}, {1, 1, 2, 3},
                //Trailing 1s:
                {4, 3, 2, 1}, {4, 3, 1, 1},
                //1s for non-leading/non-trailing dimensions
                {4, 1, 3, 2}, {4, 3, 1, 2}, {4, 1, 1, 2}};

        int[][] sumDims = {{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3},
                {0, 1, 2, 3}};
        /*        for( int[] shape : shapes) {
            for (int[] dims : sumDims) {
                System.out.println("Shape");
                System.out.println(Arrays.toString(shape));
                System.out.println("Dimensions");
                System.out.println(Arrays.toString(dims));
                int length = ArrayUtil.prod(shape);
                INDArray inC = Nd4j.linspace(1, length, length).reshape('c', shape);
                System.out.println("TAD shape");
                System.out.println(Arrays.toString((inC.tensorAlongDimension(0,dims).shape())));

                INDArray inF = inC.dup('f');
                System.out.println("C stride " + Arrays.toString(inC.tensorAlongDimension(0,dims).stride()) + " and f stride " + Arrays.toString(inF.tensorAlongDimension(0,dims).stride()));
                for(int i = 0; i < inC.tensorssAlongDimension(dims); i++) {
                    System.out.println(inC.tensorAlongDimension(i,dims).ravel());
                }
                for(int i = 0; i < inF.tensorssAlongDimension(dims); i++) {
                    System.out.println(inF.tensorAlongDimension(i,dims).ravel());
                }
            }
        }*/
        for (val shape : shapes) {
            for (int[] dims : sumDims) {
                System.out.println("Shape: " + Arrays.toString(shape) + ", sumDims=" + Arrays.toString(dims));
                int length = ArrayUtil.prod(shape);
                INDArray inC = Nd4j.linspace(1, length, length).reshape('c', shape);
                INDArray inF = inC.dup('f');
                assertEquals(inC, inF);

                INDArray sumC = inC.sum(dims);
                INDArray sumF = inF.sum(dims);
                assertEquals(sumC, sumF);

                //Multiple runs: check for consistency between runs (threading issues, etc)
                for (int i = 0; i < 100; i++) {
                    assertEquals(sumC, inC.sum(dims));
                    assertEquals(sumF, inF.sum(dims));
                }
            }
        }
    }

    @Test
    public void testIsMaxAlongDimensionSimple() {
        //Simple test: when doing IsMax along a dimension, we expect all values to be either 0 or 1
        //Do IsMax along dims 0&1 for rank 2, along 0,1&2 for rank 3, etc

        for (int rank = 2; rank <= 6; rank++) {

            int[] shape = new int[rank];
            for (int i = 0; i < rank; i++)
                shape[i] = 2;
            int length = ArrayUtil.prod(shape);


            for (int alongDimension = 0; alongDimension < rank; alongDimension++) {
                System.out.println("Testing rank " + rank + " along dimension " + alongDimension + ", (shape="
                        + Arrays.toString(shape) + ")");
                INDArray arrC = Nd4j.linspace(1, length, length).reshape('c', shape);
                INDArray arrF = arrC.dup('f');
                Nd4j.getExecutioner().execAndReturn(new IsMax(arrC, alongDimension));
                Nd4j.getExecutioner().execAndReturn(new IsMax(arrF, alongDimension));

                double[] cBuffer = arrC.data().asDouble();
                double[] fBuffer = arrF.data().asDouble();
                for (int i = 0; i < length; i++) {
                    assertTrue("c buffer value at [" + i + "]=" + cBuffer[i] + ", expected 0 or 1; dimension = "
                                    + alongDimension + ", rank = " + rank + ", shape=" + Arrays.toString(shape),
                            cBuffer[i] == 0.0 || cBuffer[i] == 1.0);
                }
                for (int i = 0; i < length; i++) {
                    assertTrue("f buffer value at [" + i + "]=" + fBuffer[i] + ", expected 0 or 1; dimension = "
                                    + alongDimension + ", rank = " + rank + ", shape=" + Arrays.toString(shape),
                            fBuffer[i] == 0.0 || fBuffer[i] == 1.0);
                }
            }
        }
    }

    @Test
    public void testSortColumns() {
        int nRows = 5;
        int nCols = 10;
        java.util.Random r = new java.util.Random(12345);

        for (int i = 0; i < nRows; i++) {
            INDArray in = Nd4j.rand(new long[] {nRows, nCols});

            List<Integer> order = new ArrayList<>(nRows);
            for (int j = 0; j < nCols; j++)
                order.add(j);
            Collections.shuffle(order, r);
            for (int j = 0; j < nCols; j++)
                in.putScalar(new long[] {i, j}, order.get(j));

            INDArray outAsc = Nd4j.sortColumns(in, i, true);
            INDArray outDesc = Nd4j.sortColumns(in, i, false);

            for (int j = 0; j < nCols; j++) {
                assertTrue(outAsc.getDouble(i, j) == j);
                int origColIdxAsc = order.indexOf(j);
                assertTrue(outAsc.getColumn(j).equals(in.getColumn(origColIdxAsc)));

                assertTrue(outDesc.getDouble(i, j) == (nCols - j - 1));
                int origColIdxDesc = order.indexOf(nCols - j - 1);
                assertTrue(outDesc.getColumn(j).equals(in.getColumn(origColIdxDesc)));
            }
        }
    }


    @Test
    public void testAddVectorWithOffset() throws Exception {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row1 = oneThroughFour.getRow(1);
        row1.addi(1);
        INDArray result = Nd4j.create(new float[] {1, 2, 4, 5}, new long[] {2, 2});
        assertEquals(getFailureMessage(), result, oneThroughFour);


    }



    @Test
    public void testLinearViewGetAndPut() throws Exception {
        INDArray test = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray linear = test.linearView();
        linear.putScalar(2, 6);
        linear.putScalar(3, 7);
        assertEquals(getFailureMessage(), 6, linear.getFloat(2), 1e-1);
        assertEquals(getFailureMessage(), 7, linear.getFloat(3), 1e-1);
    }



    @Test
    public void testRowVectorGemm() {
        INDArray linspace = Nd4j.linspace(1, 4, 4);
        INDArray other = Nd4j.linspace(1, 16, 16).reshape(4, 4);
        INDArray result = linspace.mmul(other);
        INDArray assertion = Nd4j.create(new double[] {90, 100, 110, 120});
        assertEquals(assertion, result);
    }

    @Test
    public void testGemmStrided(){

        for( val x : new int[]{5, 1}) {

            List<Pair<INDArray, String>> la = NDArrayCreationUtil.getAllTestMatricesWithShape(5, x, 12345);
            List<Pair<INDArray, String>> lb = NDArrayCreationUtil.getAllTestMatricesWithShape(x, 4, 12345);

            for (int i = 0; i < la.size(); i++) {
                for (int j = 0; j < lb.size(); j++) {

                    String msg = "x=" + x + ", i=" + i + ", j=" + j;

                    INDArray a = la.get(i).getFirst();
                    INDArray b = lb.get(i).getFirst();

                    INDArray result1 = Nd4j.createUninitialized(5, 4);
                    INDArray result2 = Nd4j.createUninitialized(5, 4);
                    INDArray result3 = Nd4j.createUninitialized(5, 4);

                    Nd4j.gemm(a.dup('c'), b.dup('c'), result1, false, false, 1.0, 0.0);
                    Nd4j.gemm(a.dup('f'), b.dup('f'), result2, false, false, 1.0, 0.0);
                    Nd4j.gemm(a, b, result3, false, false, 1.0, 0.0);

                    assertEquals(msg, result1, result2);
                    assertEquals(msg, result1, result3);     // Fails here
                }
            }
        }
    }


    @Test
    public void testMultiSum() {
        /**
         * ([[[ 0.,  1.],
         [ 2.,  3.]],

         [[ 4.,  5.],
         [ 6.,  7.]]])

         [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]


         Rank: 3,Offset: 0
         Order: c shape: [2,2,2], stride: [4,2,1]
         */
        /* */
        INDArray arr = Nd4j.linspace(0, 7, 8).reshape('c', 2, 2, 2);
        /* [0.0,4.0,2.0,6.0,1.0,5.0,3.0,7.0]
        *
        * Rank: 3,Offset: 0
            Order: f shape: [2,2,2], stride: [1,2,4]*/
        INDArray arrF = Nd4j.create(new long[] {2, 2, 2}, 'f').assign(arr);

        assertEquals(arr, arrF);
        //0,2,4,6 and 1,3,5,7
        assertEquals(Nd4j.create(new double[] {12, 16}), arr.sum(0, 1));
        //0,1,4,5 and 2,3,6,7
        assertEquals(Nd4j.create(new double[] {10, 18}), arr.sum(0, 2));
        //0,2,4,6 and 1,3,5,7
        assertEquals(Nd4j.create(new double[] {12, 16}), arrF.sum(0, 1));
        //0,1,4,5 and 2,3,6,7
        assertEquals(Nd4j.create(new double[] {10, 18}), arrF.sum(0, 2));

        //0,1,2,3 and 4,5,6,7
        assertEquals(Nd4j.create(new double[] {6, 22}), arr.sum(1, 2));
        //0,1,2,3 and 4,5,6,7
        assertEquals(Nd4j.create(new double[] {6, 22}), arrF.sum(1, 2));


        double[] data = new double[] {10, 26, 42};
        INDArray assertion = Nd4j.create(data);
        for (int i = 0; i < data.length; i++) {
            assertEquals(data[i], assertion.getDouble(i), 1e-1);
        }

        INDArray twoTwoByThree = Nd4j.linspace(1, 12, 12).reshape('f', 2, 2, 3);
        INDArray multiSum = twoTwoByThree.sum(0, 1);
        assertEquals(assertion, multiSum);
    }


    @Test
    public void testSum2dv2() {
        INDArray in = Nd4j.linspace(1, 8, 8).reshape('c', 2, 2, 2);

        val dims = new int[][] {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
        double[][] exp = new double[][] {{16, 20}, {16, 20}, {14, 22}, {14, 22}, {10, 26}, {10, 26}};

        System.out.println("dims\texpected\t\tactual");
        for (int i = 0; i < dims.length; i++) {
            val d = dims[i];
            double[] e = exp[i];

            INDArray out = in.sum(d);

            System.out.println(Arrays.toString(d) + "\t" + Arrays.toString(e) + "\t" + out);
            assertEquals(Nd4j.create(e, out.shape()), out);
        }
    }


    //Passes on 3.9:
    @Test
    public void testSum3Of4_2222() {
        int[] shape = {2, 2, 2, 2};
        int length = ArrayUtil.prod(shape);
        INDArray arrC = Nd4j.linspace(1, length, length).reshape('c', shape);
        INDArray arrF = Nd4j.create(arrC.shape()).assign(arrC);

        int[][] dimsToSum = new int[][] {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
        double[][] expD = new double[][] {{64, 72}, {60, 76}, {52, 84}, {36, 100}};

        for (int i = 0; i < dimsToSum.length; i++) {
            int[] d = dimsToSum[i];

            INDArray outC = arrC.sum(d);
            INDArray outF = arrF.sum(d);
            INDArray exp = Nd4j.create(expD[i], outC.shape());

            assertEquals(exp, outC);
            assertEquals(exp, outF);

            System.out.println(Arrays.toString(d) + "\t" + outC + "\t" + outF);
        }
    }

    @Test
    public void testBroadcast1d() {
        int[] shape = {4, 3, 2};
        int[] toBroadcastDims = new int[] {0, 1, 2};
        int[][] toBroadcastShapes = new int[][] {{1, 4}, {1, 3}, {1, 2}};

        //Expected result values in buffer: c order, need to reshape to {4,3,2}. Values taken from 0.4-rc3.8
        double[][] expFlat = new double[][] {
                {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0,
                        4.0, 4.0, 4.0, 4.0, 4.0},
                {1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0,
                        1.0, 2.0, 2.0, 3.0, 3.0},
                {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0,
                        2.0, 1.0, 2.0, 1.0, 2.0}};

        double[][] expLinspaced = new double[][] {
                {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                        21.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0},
                {2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 8.0, 9.0, 11.0, 12.0, 14.0, 15.0, 14.0, 15.0, 17.0, 18.0, 20.0,
                        21.0, 20.0, 21.0, 23.0, 24.0, 26.0, 27.0},
                {2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0, 10.0, 12.0, 12.0, 14.0, 14.0, 16.0, 16.0, 18.0, 18.0,
                        20.0, 20.0, 22.0, 22.0, 24.0, 24.0, 26.0}};

        for (int i = 0; i < toBroadcastDims.length; i++) {
            int dim = toBroadcastDims[i];
            int[] vectorShape = toBroadcastShapes[i];
            int length = ArrayUtil.prod(vectorShape);

            INDArray zC = Nd4j.create(shape, 'c');
            zC.setData(Nd4j.linspace(1, 24, 24).data());
            for (int tad = 0; tad < zC.tensorssAlongDimension(dim); tad++) {
                INDArray javaTad = zC.javaTensorAlongDimension(tad, dim);
                System.out.println("Tad " + tad + " is " + zC.tensorAlongDimension(tad, dim));
            }

            INDArray zF = Nd4j.create(shape, 'f');
            zF.assign(zC);
            INDArray toBroadcast = Nd4j.linspace(1, length, length);

            Op opc = new BroadcastAddOp(zC, toBroadcast, zC, dim);
            Op opf = new BroadcastAddOp(zF, toBroadcast, zF, dim);
            INDArray exp = Nd4j.create(expLinspaced[i], shape, 'c');
            INDArray expF = Nd4j.create(shape, 'f');
            expF.assign(exp);
            for (int tad = 0; tad < zC.tensorssAlongDimension(dim); tad++) {
                System.out.println(zC.tensorAlongDimension(tad, dim).offset() + " and f offset is "
                        + zF.tensorAlongDimension(tad, dim).offset());
            }

            Nd4j.getExecutioner().exec(opc);
            Nd4j.getExecutioner().exec(opf);

            assertEquals(exp, zC);
            assertEquals(exp, zF);
        }
    }

    @Test
    public void testSum3Of4_3322() {
        int[] shape = {3, 3, 2, 2};
        int length = ArrayUtil.prod(shape);
        INDArray arrC = Nd4j.linspace(1, length, length).reshape('c', shape);
        INDArray arrF = Nd4j.create(arrC.shape()).assign(arrC);

        int[][] dimsToSum = new int[][] {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
        double[][] expD = new double[][] {{324, 342}, {315, 351}, {174, 222, 270}, {78, 222, 366}};

        for (int i = 0; i < dimsToSum.length; i++) {
            int[] d = dimsToSum[i];

            INDArray outC = arrC.sum(d);
            INDArray outF = arrF.sum(d);
            INDArray exp = Nd4j.create(expD[i], outC.shape());

            assertEquals(exp, outC);
            assertEquals(exp, outF);

            //System.out.println(Arrays.toString(d) + "\t" + outC + "\t" + outF);
        }
    }

    @Test
    public void testToFlattened() {
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        List<INDArray> concat = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            concat.add(arr.dup());
        }

        INDArray assertion = Nd4j.create(new double[] {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});
        INDArray flattened = Nd4j.toFlattened(concat);
        assertEquals(assertion, flattened);

    }



    @Test
    public void testDup() {
        for (int x = 0; x < 100; x++) {
            INDArray orig = Nd4j.linspace(1, 4, 4);
            INDArray dup = orig.dup();
            assertEquals(orig, dup);

            INDArray matrix = Nd4j.create(new float[] {1, 2, 3, 4}, new long[] {2, 2});
            INDArray dup2 = matrix.dup();
            assertEquals(matrix, dup2);

            INDArray row1 = matrix.getRow(1);
            INDArray dupRow = row1.dup();
            assertEquals(row1, dupRow);


            INDArray columnSorted = Nd4j.create(new float[] {2, 1, 4, 3}, new long[] {2, 2});
            INDArray dup3 = columnSorted.dup();
            assertEquals(columnSorted, dup3);
        }
    }

    @Test
    public void testSortWithIndicesDescending() {
        INDArray toSort = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        //indices,data
        INDArray[] sorted = Nd4j.sortWithIndices(toSort.dup(), 1, false);
        INDArray sorted2 = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(sorted[1], sorted2);
        INDArray shouldIndex = Nd4j.create(new float[] {1, 0, 1, 0}, new long[] {2, 2});
        assertEquals(shouldIndex, sorted[0]);


    }

    @Test
    public void testGetFromRowVector() {
        INDArray matrix = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray rowGet = matrix.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 2));
        assertArrayEquals(new long[] {1, 2}, rowGet.shape());
    }

    @Test
    public void testSubRowVector() {
        INDArray matrix = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray row = Nd4j.linspace(1, 3, 3);
        INDArray test = matrix.subRowVector(row);
        INDArray assertion = Nd4j.create(new double[][] {{0, 0, 0}, {3, 3, 3}});
        assertEquals(assertion, test);

        INDArray threeByThree = Nd4j.linspace(1, 9, 9).reshape(3, 3);
        INDArray offsetTest = threeByThree.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        assertEquals(2, offsetTest.rows());
        INDArray offsetAssertion = Nd4j.create(new double[][] {{3, 3, 3}, {6, 6, 6}});
        INDArray offsetSub = offsetTest.subRowVector(row);
        assertEquals(offsetAssertion, offsetSub);

    }



    @Test
    public void testDimShuffle() {
        INDArray n = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray twoOneTwo = n.dimShuffle(new Object[] {0, 'x', 1}, new int[] {0, 1}, new boolean[] {false, false});
        assertTrue(Arrays.equals(new long[] {2, 1, 2}, twoOneTwo.shape()));

        INDArray reverse = n.dimShuffle(new Object[] {1, 'x', 0}, new int[] {1, 0}, new boolean[] {false, false});
        assertTrue(Arrays.equals(new long[] {2, 1, 2}, reverse.shape()));

    }

    @Test
    public void testGetVsGetScalar() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        float element = a.getFloat(0, 1);
        double element2 = a.getDouble(0, 1);
        assertEquals(element, element2, 1e-1);
        INDArray a2 = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        float element23 = a2.getFloat(0, 1);
        double element22 = a2.getDouble(0, 1);
        assertEquals(element23, element22, 1e-1);

    }

    @Test
    public void testDivide() {
        INDArray two = Nd4j.create(new float[] {2, 2, 2, 2});
        INDArray div = two.div(two);
        assertEquals(Nd4j.ones(4), div);

        INDArray half = Nd4j.create(new float[] {0.5f, 0.5f, 0.5f, 0.5f}, new long[] {2, 2});
        INDArray divi = Nd4j.create(new float[] {0.3f, 0.6f, 0.9f, 0.1f}, new long[] {2, 2});
        INDArray assertion = Nd4j.create(new float[] {1.6666666f, 0.8333333f, 0.5555556f, 5}, new long[] {2, 2});
        INDArray result = half.div(divi);
        assertEquals(assertion, result);
    }


    @Test
    public void testSigmoid() {
        INDArray n = Nd4j.create(new float[] {1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new float[] {0.73105858f, 0.88079708f, 0.95257413f, 0.98201379f});
        INDArray sigmoid = Transforms.sigmoid(n, false);
        assertEquals(assertion, sigmoid);
    }

    @Test
    public void testNeg() {
        INDArray n = Nd4j.create(new float[] {1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new float[] {-1, -2, -3, -4});
        INDArray neg = Transforms.neg(n);
        assertEquals(getFailureMessage(), assertion, neg);

    }

    @Test
    public void testNorm2Double() {
        DataBuffer.Type initialType = Nd4j.dataType();
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        INDArray n = Nd4j.create(new double[] {1, 2, 3, 4});
        double assertion = 5.47722557505;
        double norm3 = n.norm2Number().doubleValue();
        assertEquals(getFailureMessage(), assertion, norm3, 1e-1);

        INDArray row = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray row1 = row.getRow(1);
        double norm2 = row1.norm2Number().doubleValue();
        double assertion2 = 5.0f;
        assertEquals(getFailureMessage(), assertion2, norm2, 1e-1);
        DataTypeUtil.setDTypeForContext(initialType);
    }


    @Test
    public void testNorm2() {
        INDArray n = Nd4j.create(new float[] {1, 2, 3, 4});
        float assertion = 5.47722557505f;
        float norm3 = n.norm2Number().floatValue();
        assertEquals(getFailureMessage(), assertion, norm3, 1e-1);


        INDArray row = Nd4j.create(new float[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray row1 = row.getRow(1);
        float norm2 = row1.norm2Number().floatValue();
        float assertion2 = 5.0f;
        assertEquals(getFailureMessage(), assertion2, norm2, 1e-1);

    }



    @Test
    public void testCosineSim() {
        INDArray vec1 = Nd4j.create(new double[] {1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new double[] {1, 2, 3, 4});
        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(getFailureMessage(), 1, sim, 1e-1);

        INDArray vec3 = Nd4j.create(new float[] {0.2f, 0.3f, 0.4f, 0.5f});
        INDArray vec4 = Nd4j.create(new float[] {0.6f, 0.7f, 0.8f, 0.9f});
        sim = Transforms.cosineSim(vec3, vec4);
        assertEquals(0.98, sim, 1e-1);

    }


    @Test
    public void testScal() {
        double assertion = 2;
        INDArray answer = Nd4j.create(new double[] {2, 4, 6, 8});
        INDArray scal = Nd4j.getBlasWrapper().scal(assertion, answer);
        assertEquals(getFailureMessage(), answer, scal);

        INDArray row = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray row1 = row.getRow(1);
        double assertion2 = 5.0;
        INDArray answer2 = Nd4j.create(new double[] {15, 20});
        INDArray scal2 = Nd4j.getBlasWrapper().scal(assertion2, row1);
        assertEquals(getFailureMessage(), answer2, scal2);

    }

    @Test
    public void testExp() {
        INDArray n = Nd4j.create(new double[] {1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new double[] {2.71828183f, 7.3890561f, 20.08553692f, 54.59815003f});
        INDArray exped = Transforms.exp(n);
        assertEquals(assertion, exped);

        assertArrayEquals(new double[] {2.71828183f, 7.3890561f, 20.08553692f, 54.59815003f}, exped.toDoubleVector(), 1e-5);
        assertArrayEquals(new double[] {2.71828183f, 7.3890561f, 20.08553692f, 54.59815003f}, assertion.toDoubleVector(), 1e-5);
    }


    @Test
    public void testSlices() {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new long[] {4, 3, 2});
        for (int i = 0; i < arr.slices(); i++) {
            assertEquals(2, arr.slice(i).slice(1).slices());
        }

    }


    @Test
    public void testScalar() {
        INDArray a = Nd4j.scalar(1.0);
        assertEquals(true, a.isScalar());

        INDArray n = Nd4j.create(new float[] {1.0f}, new long[] {1, 1});
        assertEquals(n, a);
        assertTrue(n.isScalar());
    }

    @Test
    public void testWrap() throws Exception {
        int[] shape = {2, 4};
        INDArray d = Nd4j.linspace(1, 8, 8).reshape(shape[0], shape[1]);
        INDArray n = d;
        assertEquals(d.rows(), n.rows());
        assertEquals(d.columns(), n.columns());

        INDArray vector = Nd4j.linspace(1, 3, 3);
        INDArray testVector = vector;
        for (int i = 0; i < vector.length(); i++)
            assertEquals(vector.getDouble(i), testVector.getDouble(i), 1e-1);
        assertEquals(3, testVector.length());
        assertEquals(true, testVector.isVector());
        assertEquals(true, Shape.shapeEquals(new long[] {3}, testVector.shape()));

        INDArray row12 = Nd4j.linspace(1, 2, 2).reshape(2, 1);
        INDArray row22 = Nd4j.linspace(3, 4, 2).reshape(1, 2);

        assertEquals(row12.rows(), 2);
        assertEquals(row12.columns(), 1);
        assertEquals(row22.rows(), 1);
        assertEquals(row22.columns(), 2);
    }



    @Test
    public void testVectorInit() {
        DataBuffer data = Nd4j.linspace(1, 4, 4).data();
        INDArray arr = Nd4j.create(data, new long[] {1, 4});
        assertEquals(true, arr.isRowVector());
        INDArray arr2 = Nd4j.create(data, new long[] {1, 4});
        assertEquals(true, arr2.isRowVector());

        INDArray columnVector = Nd4j.create(data, new long[] {4, 1});
        assertEquals(true, columnVector.isColumnVector());
    }


    @Test
    public void testColumns() {
        INDArray arr = Nd4j.create(new long[] {3, 2});
        INDArray column2 = arr.getColumn(0);
        //assertEquals(true, Shape.shapeEquals(new long[]{3, 1}, column2.shape()));
        INDArray column = Nd4j.create(new double[] {1, 2, 3}, new long[] {1, 3});
        arr.putColumn(0, column);

        INDArray firstColumn = arr.getColumn(0);

        assertEquals(column, firstColumn);


        INDArray column1 = Nd4j.create(new double[] {4, 5, 6}, new long[] {1, 3});
        arr.putColumn(1, column1);
        //assertEquals(true, Shape.shapeEquals(new long[]{3, 1}, arr.getColumn(1).shape()));
        INDArray testRow1 = arr.getColumn(1);
        assertEquals(column1, testRow1);


        INDArray evenArr = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray put = Nd4j.create(new double[] {5, 6}, new long[] {1, 2});
        evenArr.putColumn(1, put);
        INDArray testColumn = evenArr.getColumn(1);
        assertEquals(put, testColumn);


        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new long[] {2, 2});
        INDArray column23 = n.getColumn(0);
        INDArray column12 = Nd4j.create(new double[] {1, 3}, new long[] {1, 2});
        assertEquals(column23, column12);


        INDArray column0 = n.getColumn(1);
        INDArray column01 = Nd4j.create(new double[] {2, 4}, new long[] {1, 2});
        assertEquals(column0, column01);


    }


    @Test
    public void testPutRow() {
        INDArray d = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray slice1 = d.slice(1);
        INDArray n = d.dup();

        //works fine according to matlab, let's go with it..
        //reproduce with:  A = newShapeNoCopy(linspace(1,4,4),[2 2 ]);
        //A(1,2) % 1 index based
        float nFirst = 2;
        float dFirst = d.getFloat(0, 1);
        assertEquals(nFirst, dFirst, 1e-1);
        assertEquals(d, n);
        assertEquals(true, Arrays.equals(new long[] {2, 2}, n.shape()));

        INDArray newRow = Nd4j.linspace(5, 6, 2);
        n.putRow(0, newRow);
        d.putRow(0, newRow);


        INDArray testRow = n.getRow(0);
        assertEquals(newRow.length(), testRow.length());
        assertEquals(true, Shape.shapeEquals(new long[] {1, 2}, testRow.shape()));


        INDArray nLast = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new long[] {2, 2});
        INDArray row = nLast.getRow(1);
        INDArray row1 = Nd4j.create(new double[] {3, 4}, new long[] {1, 2});
        assertEquals(row, row1);


        INDArray arr = Nd4j.create(new long[] {3, 2});
        INDArray evenRow = Nd4j.create(new double[] {1, 2}, new long[] {1, 2});
        arr.putRow(0, evenRow);
        INDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new long[] {1, 2}, firstRow.shape()));
        INDArray testRowEven = arr.getRow(0);
        assertEquals(evenRow, testRowEven);


        INDArray row12 = Nd4j.create(new double[] {5, 6}, new long[] {1, 2});
        arr.putRow(1, row12);
        assertEquals(true, Shape.shapeEquals(new long[] {1, 2}, arr.getRow(0).shape()));
        INDArray testRow1 = arr.getRow(1);
        assertEquals(row12, testRow1);


        INDArray multiSliceTest = Nd4j.create(Nd4j.linspace(1, 16, 16).data(), new long[] {4, 2, 2});
        INDArray test = Nd4j.create(new double[] {5, 6}, new long[] {1, 2});
        INDArray test2 = Nd4j.create(new double[] {7, 8}, new long[] {1, 2});

        INDArray multiSliceRow1 = multiSliceTest.slice(1).getRow(0);
        INDArray multiSliceRow2 = multiSliceTest.slice(1).getRow(1);

        assertEquals(test, multiSliceRow1);
        assertEquals(test2, multiSliceRow2);



        INDArray threeByThree = Nd4j.create(3, 3);
        INDArray threeByThreeRow1AndTwo = threeByThree.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        threeByThreeRow1AndTwo.putRow(1, Nd4j.ones(3));
        assertEquals(Nd4j.ones(3), threeByThreeRow1AndTwo.getRow(1));

    }


    @Test
    public void testMulRowVector() {
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        arr.muliRowVector(Nd4j.linspace(1, 2, 2));
        INDArray assertion = Nd4j.create(new double[][] {{1, 4}, {3, 8}});

        assertEquals(assertion, arr);
    }



    @Test
    public void testSum() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new long[] {2, 2, 2});
        INDArray test = Nd4j.create(new float[] {3, 7, 11, 15}, new long[] {2, 2});
        INDArray sum = n.sum(-1);
        assertEquals(test, sum);

    }



    @Test
    public void testInplaceTranspose() {
        INDArray test = Nd4j.rand(34, 484);
        INDArray transposei = test.transposei();

        for (int i = 0; i < test.rows(); i++) {
            for (int j = 0; j < test.columns(); j++) {
                assertEquals(test.getDouble(i, j), transposei.getDouble(j, i), 1e-1);
            }
        }

    }

    @Test
    public void testTADMMul() {
        Nd4j.getRandom().setSeed(12345);
        val shape = new long[] {4, 5, 7};
        INDArray arr = Nd4j.rand(shape);

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
        INDArray first = Nd4j.rand(new long[] {2, 7});
        INDArray mmul = first.mmul(tad);
        INDArray mmulCopy = first.mmul(copy);

        assertEquals(mmul, mmulCopy);
    }

    @Test
    public void testTADMMulLeadingOne() {
        Nd4j.getRandom().setSeed(12345);
        val shape = new long[] {1, 5, 7};
        INDArray arr = Nd4j.rand(shape);

        INDArray tad = arr.tensorAlongDimension(0, 1, 2);
        boolean order = Shape.cOrFortranOrder(tad.shape(), tad.stride(), tad.elementStride());
        assertArrayEquals(tad.shape(), new long[] {5, 7});


        INDArray copy = Nd4j.zeros(5, 7);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 7; j++) {
                copy.putScalar(new long[] {i, j}, tad.getDouble(i, j));
            }
        }

        assertTrue(tad.equals(copy));

        tad = tad.reshape(7, 5);
        copy = copy.reshape(7, 5);
        INDArray first = Nd4j.rand(new long[] {2, 7});
        INDArray mmul = first.mmul(tad);
        INDArray mmulCopy = first.mmul(copy);

        assertTrue(mmul.equals(mmulCopy));
    }


    @Test
    public void testSum2() {
        INDArray test = Nd4j.create(new float[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray sum = test.sum(1);
        INDArray assertion = Nd4j.create(new float[] {3, 7});
        assertEquals(assertion, sum);
        INDArray sum0 = Nd4j.create(new double[] {4, 6});
        assertEquals(sum0, test.sum(0));
    }


    @Test
    public void testGetIntervalEdgeCase() {
        Nd4j.getRandom().setSeed(12345);

        int[] shape = {3, 2, 4};
        INDArray arr3d = Nd4j.rand(shape);

        INDArray get0 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 1));
        INDArray getPoint0 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
        get0 = get0.reshape(getPoint0.shape());
        INDArray tad0 = arr3d.tensorAlongDimension(0, 1, 0);

        assertTrue(get0.equals(getPoint0)); //OK
        assertTrue(getPoint0.equals(tad0)); //OK

        INDArray get1 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 2));
        INDArray getPoint1 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1));
        get1 = get1.reshape(getPoint1.shape());
        INDArray tad1 = arr3d.tensorAlongDimension(1, 1, 0);

        assertTrue(getPoint1.equals(tad1)); //OK
        assertTrue(get1.equals(getPoint1)); //Fails
        assertTrue(get1.equals(tad1));

        INDArray get2 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(2, 3));
        INDArray getPoint2 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(2));
        get2 = get2.reshape(getPoint2.shape());
        INDArray tad2 = arr3d.tensorAlongDimension(2, 1, 0);

        assertTrue(getPoint2.equals(tad2)); //OK
        assertTrue(get2.equals(getPoint2)); //Fails
        assertTrue(get2.equals(tad2));

        INDArray get3 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(3, 4));
        INDArray getPoint3 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(3));
        get3 = get3.reshape(getPoint3.shape());
        INDArray tad3 = arr3d.tensorAlongDimension(3, 1, 0);

        assertTrue(getPoint3.equals(tad3)); //OK
        assertTrue(get3.equals(getPoint3)); //Fails
        assertTrue(get3.equals(tad3));
    }


    @Test
    public void testGetIntervalEdgeCase2() {
        Nd4j.getRandom().setSeed(12345);

        int[] shape = {3, 2, 4};
        INDArray arr3d = Nd4j.rand(shape);

        for (int x = 0; x < 4; x++) {
            INDArray getInterval = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(x, x + 1)); //3d
            INDArray getPoint = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(x)); //2d
            INDArray tad = arr3d.tensorAlongDimension(x, 1, 0); //2d

            assertEquals(getPoint, tad);
            //assertTrue(getPoint.equals(tad));   //OK, comparing 2d with 2d
            assertArrayEquals(getInterval.shape(), new long[] {3, 2, 1});
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(getInterval.getDouble(i, j, 0), getPoint.getDouble(i, j), 1e-1);
                }
            }
        }
    }


    @Test
    public void testMmul() {
        DataBuffer data = Nd4j.linspace(1, 10, 10).data();
        INDArray n = Nd4j.create(data, new long[] {1, 10});
        INDArray transposed = n.transpose();
        assertEquals(true, n.isRowVector());
        assertEquals(true, transposed.isColumnVector());

        INDArray d = Nd4j.create(n.rows(), n.columns());
        d.setData(n.data());


        INDArray d3 = Nd4j.create(new double[] {1, 2}).reshape(2, 1);
        INDArray d4 = Nd4j.create(new double[] {3, 4});
        INDArray resultNDArray = d3.mmul(d4);
        INDArray result = Nd4j.create(new double[][] {{3, 4}, {6, 8}});
        assertEquals(result, resultNDArray);


        INDArray innerProduct = n.mmul(transposed);

        INDArray scalar = Nd4j.scalar(385);
        assertEquals(getFailureMessage(), scalar, innerProduct);

        INDArray outerProduct = transposed.mmul(n);
        assertEquals(getFailureMessage(), true, Shape.shapeEquals(new long[] {10, 10}, outerProduct.shape()));



        INDArray three = Nd4j.create(new double[] {3, 4}, new long[] {1, 2});
        INDArray test = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new long[] {3, 5, 2});
        INDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(getFailureMessage(), three, sliceRow);

        INDArray twoSix = Nd4j.create(new double[] {2, 6}, new long[] {2, 1});
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
                225}, new long[] {16, 16});


        INDArray n1 = Nd4j.create(Nd4j.linspace(0, 15, 16).data(), new long[] {1, 16});
        INDArray k1 = n1.transpose();

        INDArray testVectorVector = k1.mmul(n1);
        assertEquals(getFailureMessage(), vectorVector, testVectorVector);


    }



    @Test
    public void testRowsColumns() {
        DataBuffer data = Nd4j.linspace(1, 6, 6).data();
        INDArray rows = Nd4j.create(data, new long[] {2, 3});
        assertEquals(2, rows.rows());
        assertEquals(3, rows.columns());

        INDArray columnVector = Nd4j.create(data, new long[] {6, 1});
        assertEquals(6, columnVector.rows());
        assertEquals(1, columnVector.columns());
        INDArray rowVector = Nd4j.create(data, new long[] {1, 6});
        assertEquals(1, rowVector.rows());
        assertEquals(6, rowVector.columns());
    }


    @Test
    public void testTranspose() {
        INDArray n = Nd4j.create(Nd4j.ones(100).data(), new long[] {5, 5, 4});
        INDArray transpose = n.transpose();
        assertEquals(n.length(), transpose.length());
        assertEquals(true, Arrays.equals(new long[] {4, 5, 5}, transpose.shape()));

        INDArray rowVector = Nd4j.linspace(1, 10, 10);
        assertTrue(rowVector.isRowVector());
        INDArray columnVector = rowVector.transpose();
        assertTrue(columnVector.isColumnVector());


        INDArray linspaced = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray transposed = Nd4j.create(new float[] {1, 3, 2, 4}, new long[] {2, 2});
        INDArray linSpacedT = linspaced.transpose();
        assertEquals(transposed, linSpacedT);



    }


    @Test
    public void testLogX1() {
        INDArray x = Nd4j.create(10).assign(7);

        INDArray logX5 = Transforms.log(x, 5, true);

        INDArray exp = Transforms.log(x, true).div(Transforms.log(Nd4j.create(10).assign(5)));

        assertEquals(exp, logX5);
    }

    @Test
    public void testAddMatrix() {
        INDArray five = Nd4j.ones(5);
        five.addi(five);
        INDArray twos = Nd4j.valueArrayOf(5, 2);
        assertEquals(twos, five);
    }



    @Test
    public void testPutSlice() {
        INDArray n = Nd4j.linspace(1, 27, 27).reshape(3, 3, 3);
        INDArray newSlice = Nd4j.zeros(3, 3);
        n.putSlice(0, newSlice);
        assertEquals(newSlice, n.slice(0));

        INDArray firstDimensionAs1 = newSlice.reshape(1, 3, 3);
        n.putSlice(0, firstDimensionAs1);


    }



    @Test
    public void testRowVectorMultipleIndices() {
        INDArray linear = Nd4j.create(1, 4);
        linear.putScalar(new long[] {0, 1}, 1);
        assertEquals(linear.getDouble(0, 1), 1, 1e-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSize() {
        INDArray arr = Nd4j.create(4, 5);

        for (int i = 0; i < 6; i++) {
            //This should fail for i >= 2, but doesn't
            System.out.println(arr.size(i));
        }
    }

    @Test
    public void testNullPointerDataBuffer() {
        DataBuffer.Type initialType = Nd4j.dataType();

        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);

        ByteBuffer allocate = ByteBuffer.allocateDirect(10 * 4).order(ByteOrder.nativeOrder());
        allocate.asFloatBuffer().put(new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        DataBuffer buff = Nd4j.createBuffer(allocate, DataBuffer.Type.FLOAT, 10);
        float sum = Nd4j.create(buff).sumNumber().floatValue();
        System.out.println(sum);
        assertEquals(55f, sum, 0.001f);

        DataTypeUtil.setDTypeForContext(initialType);
    }

    @Test
    public void testEps() {
        INDArray ones = Nd4j.ones(5);
        double sum = Nd4j.getExecutioner().exec(new Eps(ones, ones, ones, ones.length())).z().sumNumber().doubleValue();
        assertEquals(5, sum, 1e-1);
    }

    @Test
    public void testEps2() {

        INDArray first = Nd4j.valueArrayOf(10, 1e-2); //0.01
        INDArray second = Nd4j.zeros(10); //0.0

        INDArray expAllZeros1 = Nd4j.getExecutioner()
                .execAndReturn(new Eps(first, second, Nd4j.create(new long[] {1, 10}, 'f'), 10));
        INDArray expAllZeros2 = Nd4j.getExecutioner()
                .execAndReturn(new Eps(second, first, Nd4j.create(new long[] {1, 10}, 'f'), 10));

        System.out.println(expAllZeros1);
        System.out.println(expAllZeros2);

        assertEquals(0, expAllZeros1.sumNumber().doubleValue(), 0.0);
        assertEquals(0, expAllZeros2.sumNumber().doubleValue(), 0.0);
    }

    @Test
    public void testLogDouble() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        INDArray log = Transforms.log(linspace);
        INDArray assertion = Nd4j.create(new double[] {0, 0.6931471805599453, 1.0986122886681098, 1.3862943611198906,
                1.6094379124341005, 1.791759469228055});
        assertEquals(assertion, log);
    }

    @Test
    public void testDupDimension() {
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        assertEquals(arr.tensorAlongDimension(0, 1), arr.tensorAlongDimension(0, 1));
    }


    @Test
    public void testIterator() {
        INDArray x = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray repeated = x.repeat(1, new int[] {2});
        assertEquals(8, repeated.length());
        Iterator<Double> arrayIter = new INDArrayIterator(x);
        double[] vals = Nd4j.linspace(1, 4, 4).data().asDouble();
        for (int i = 0; i < vals.length; i++)
            assertEquals(vals[i], arrayIter.next().doubleValue(), 1e-1);
    }

    @Test
    public void testTile() {
        INDArray x = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray repeated = x.repeat(0, new int[] {2});
        assertEquals(8, repeated.length());
        INDArray repeatAlongDimension = x.repeat(1, new long[] {2});
        INDArray assertionRepeat = Nd4j.create(new double[][] {{1, 1, 2, 2}, {3, 3, 4, 4}});
        assertArrayEquals(new long[] {2, 4}, assertionRepeat.shape());
        assertEquals(assertionRepeat, repeatAlongDimension);
        System.out.println(repeatAlongDimension);
        INDArray ret = Nd4j.create(new double[] {0, 1, 2});
        INDArray tile = Nd4j.tile(ret, 2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{0, 1, 2, 0, 1, 2}, {0, 1, 2, 0, 1, 2}});
        assertEquals(assertion, tile);
    }

    @Test
    public void testNegativeOneReshape() {
        INDArray arr = Nd4j.create(new double[] {0, 1, 2});
        INDArray newShape = arr.reshape(-1, 3);
        assertEquals(newShape, arr);
    }


    @Test
    public void testSmallSum() {
        INDArray base = Nd4j.create(new double[] {5.843333333333335, 3.0540000000000007});
        base.addi(1e-12);
        INDArray assertion = Nd4j.create(new double[] {5.84333433, 3.054001});
        assertEquals(assertion, base);

    }


    @Test
    public void test2DArraySlice() {
        INDArray array2D = Nd4j.ones(5, 7);
        /**
         * This should be reverse.
         * This is compatibility with numpy.
         *
         * If you do numpy.sum along dimension
         * 1 you will find its row sums.
         *
         * 0 is columns sums.
         *
         * slice(0,axis)
         * should be consistent with this behavior
         */
        for (int i = 0; i < 7; i++) {
            INDArray slice = array2D.slice(i, 1);
            assertTrue(Arrays.equals(slice.shape(), new long[] {5, 1}));
        }

        for (int i = 0; i < 5; i++) {
            INDArray slice = array2D.slice(i, 0);
            assertTrue(Arrays.equals(slice.shape(), new long[] {1, 7}));
        }
    }

    @Test
    public void testTensorDot() {
        INDArray oneThroughSixty = Nd4j.arange(60).reshape(3, 4, 5);
        INDArray oneThroughTwentyFour = Nd4j.arange(24).reshape(4, 3, 2);
        INDArray result = Nd4j.tensorMmul(oneThroughSixty, oneThroughTwentyFour, new int[][] {{1, 0}, {0, 1}});
        assertArrayEquals(new long[] {5, 2}, result.shape());
        INDArray assertion = Nd4j
                .create(new double[][] {{4400, 4730}, {4532, 4874}, {4664, 5018}, {4796, 5162}, {4928, 5306}});
        assertEquals(assertion, result);

        INDArray w = Nd4j.valueArrayOf(new long[] {2, 1, 2, 2}, 0.5);
        INDArray col = Nd4j.create(new double[] {1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3,
                1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4,
                2, 2, 2, 2, 4, 4, 4, 4}, new long[] {1, 1, 2, 2, 4, 4});

        INDArray test = Nd4j.tensorMmul(col, w, new int[][] {{1, 2, 3}, {1, 2, 3}});
        INDArray assertion2 = Nd4j.create(
                new double[] {3., 3., 3., 3., 3., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7., 3., 3., 3., 3.,
                        3., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7.},
                new long[] {1, 4, 4, 2}, new long[] {16, 8, 2, 1}, 0, 'f');
        //        assertion2.setOrder('f');
        assertEquals(assertion2, test);
    }



    @Test
    public void testGetRow() {
        INDArray arr = Nd4j.ones(10, 4);
        for (int i = 0; i < 10; i++) {
            INDArray row = arr.getRow(i);
            assertArrayEquals(row.shape(), new long[] {1, 4});
        }
    }


    @Test
    public void testGetPermuteReshapeSub() {
        Nd4j.getRandom().setSeed(12345);

        INDArray first = Nd4j.rand(new long[] {10, 4});

        //Reshape, as per RnnOutputLayer etc on labels
        INDArray orig3d = Nd4j.rand(new long[] {2, 4, 15});
        INDArray subset3d = orig3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(5, 10));
        INDArray permuted = subset3d.permute(0, 2, 1);
        val newShape = new long []{subset3d.size(0) * subset3d.size(2), subset3d.size(1)};
        INDArray second = permuted.reshape(newShape);

        assertArrayEquals(first.shape(), second.shape());
        assertEquals(first.length(), second.length());
        assertArrayEquals(first.stride(), second.stride());

        first.sub(second); //Exception
    }


    @Test
    public void testPutAtIntervalIndexWithStride() {
        INDArray n1 = Nd4j.create(3, 3).assign(0.0);
        INDArrayIndex[] indices = {NDArrayIndex.interval(0, 2, 3), NDArrayIndex.all()};
        n1.put(indices, 1);
        INDArray expected = Nd4j.create(new double[][] {{1d, 1d, 1d}, {0d, 0d, 0d}, {1d, 1d, 1d}});
        assertEquals(expected, n1);
    }

    @Test
    public void testMMulMatrixTimesColVector() {
        //[1 1 1 1 1; 10 10 10 10 10; 100 100 100 100 100] x [1; 1; 1; 1; 1] = [5; 50; 500]
        INDArray matrix = Nd4j.ones(3, 5);
        matrix.getRow(1).muli(10);
        matrix.getRow(2).muli(100);

        INDArray colVector = Nd4j.ones(5, 1);
        INDArray out = matrix.mmul(colVector);

        INDArray expected = Nd4j.create(new double[] {5, 50, 500}, new long[] {3, 1});
        assertEquals(expected, out);
    }


    @Test
    public void testMMulMixedOrder() {
        INDArray first = Nd4j.ones(5, 2);
        INDArray second = Nd4j.ones(2, 3);
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


    @Test
    public void testFTimesCAddiRow() {

        INDArray arrF = Nd4j.create(2, 3, 'f').assign(1.0);
        INDArray arrC = Nd4j.create(2, 3, 'c').assign(1.0);
        INDArray arr2 = Nd4j.create(new long[] {3, 4}, 'c').assign(1.0);

        INDArray mmulC = arrC.mmul(arr2); //[2,4] with elements 3.0
        INDArray mmulF = arrF.mmul(arr2); //[2,4] with elements 3.0
        assertArrayEquals(mmulC.shape(), new long[] {2, 4});
        assertArrayEquals(mmulF.shape(), new long[] {2, 4});
        assertTrue(arrC.equals(arrF));

        INDArray row = Nd4j.zeros(1, 4).assign(0.0).addi(0.5);
        mmulC.addiRowVector(row); //OK
        mmulF.addiRowVector(row); //Exception

        assertTrue(mmulC.equals(mmulF));

        for (int i = 0; i < mmulC.length(); i++)
            assertEquals(mmulC.getDouble(i), 3.5, 1e-1); //OK
        for (int i = 0; i < mmulF.length(); i++)
            assertEquals(mmulF.getDouble(i), 3.5, 1e-1); //Exception
    }



    @Test
    public void testMmulGet() {
        Nd4j.getRandom().setSeed(12345L);
        INDArray elevenByTwo = Nd4j.rand(new long[] {11, 2});
        INDArray twoByEight = Nd4j.rand(new long[] {2, 8});

        INDArray view = twoByEight.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        INDArray viewCopy = view.dup();
        assertTrue(view.equals(viewCopy));

        INDArray mmul1 = elevenByTwo.mmul(view);
        INDArray mmul2 = elevenByTwo.mmul(viewCopy);

        assertTrue(mmul1.equals(mmul2));
    }


    @Test
    public void testMMulRowColVectorMixedOrder() {
        INDArray colVec = Nd4j.ones(5, 1);
        INDArray rowVec = Nd4j.ones(1, 3);
        INDArray out = colVec.mmul(rowVec);
        assertArrayEquals(out.shape(), new long[] {5, 3});
        assertTrue(out.equals(Nd4j.ones(5, 3)));
        //Above: OK

        INDArray colVectorC = Nd4j.create(new long[] {5, 1}, 'c');
        INDArray rowVectorF = Nd4j.create(new long[] {1, 3}, 'f');
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

    @Test
    public void testMMulFTimesC() {
        int nRows = 3;
        int nCols = 3;
        java.util.Random r = new java.util.Random(12345);

        INDArray arrC = Nd4j.create(new long[] {nRows, nCols}, 'c');
        INDArray arrF = Nd4j.create(new long[] {nRows, nCols}, 'f');
        INDArray arrC2 = Nd4j.create(new long[] {nRows, nCols}, 'c');
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

    @Test
    public void testMMulColVectorRowVectorMixedOrder() {
        INDArray colVec = Nd4j.ones(5, 1);
        INDArray rowVec = Nd4j.ones(1, 5);
        INDArray out = rowVec.mmul(colVec);
        assertArrayEquals(out.shape(), new long[] {1, 1});
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

    @Test
    public void testPermute() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 20, 20).data(), new long[] {5, 4});
        INDArray transpose = n.transpose();
        INDArray permute = n.permute(1, 0);
        assertEquals(permute, transpose);
        assertEquals(transpose.length(), permute.length(), 1e-1);


        INDArray toPermute = Nd4j.create(Nd4j.linspace(0, 7, 8).data(), new long[] {2, 2, 2});
        INDArray permuted = toPermute.permute(2, 1, 0);
        INDArray assertion = Nd4j.create(new float[] {0, 4, 2, 6, 1, 5, 3, 7}, new long[] {2, 2, 2});
        assertEquals(permuted, assertion);
    }

    @Test
    public void testPermutei() {
        //Check in-place permute vs. copy array permute

        //2d:
        INDArray orig = Nd4j.linspace(1, 3 * 4, 3 * 4).reshape('c', 3, 4);
        INDArray exp01 = orig.permute(0, 1);
        INDArray exp10 = orig.permute(1, 0);
        List<Pair<INDArray, String>> list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(3, 4, 12345);
        List<Pair<INDArray, String>> list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(3, 4, 12345);
        for (int i = 0; i < list1.size(); i++) {
            INDArray p1 = list1.get(i).getFirst().assign(orig).permutei(0, 1);
            INDArray p2 = list2.get(i).getFirst().assign(orig).permutei(1, 0);

            assertEquals(exp01, p1);
            assertEquals(exp10, p2);

            assertEquals(3, p1.rows());
            assertEquals(4, p1.columns());

            assertEquals(4, p2.rows());
            assertEquals(3, p2.columns());
        }

        //2d, v2
        orig = Nd4j.linspace(1, 4, 4).reshape('c', 1, 4);
        exp01 = orig.permute(0, 1);
        exp10 = orig.permute(1, 0);
        list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(1, 4, 12345);
        list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(1, 4, 12345);
        for (int i = 0; i < list1.size(); i++) {
            INDArray p1 = list1.get(i).getFirst().assign(orig).permutei(0, 1);
            INDArray p2 = list2.get(i).getFirst().assign(orig).permutei(1, 0);

            assertEquals(exp01, p1);
            assertEquals(exp10, p2);

            assertEquals(1, p1.rows());
            assertEquals(4, p1.columns());
            assertEquals(4, p2.rows());
            assertEquals(1, p2.columns());
            assertTrue(p1.isRowVector());
            assertFalse(p1.isColumnVector());
            assertFalse(p2.isRowVector());
            assertTrue(p2.isColumnVector());
        }

        //3d:
        INDArray orig3d = Nd4j.linspace(1, 3 * 4 * 5, 3 * 4 * 5).reshape('c', 3, 4, 5);
        INDArray exp012 = orig3d.permute(0, 1, 2);
        INDArray exp021 = orig3d.permute(0, 2, 1);
        INDArray exp120 = orig3d.permute(1, 2, 0);
        INDArray exp102 = orig3d.permute(1, 0, 2);
        INDArray exp201 = orig3d.permute(2, 0, 1);
        INDArray exp210 = orig3d.permute(2, 1, 0);

        List<Pair<INDArray, String>> list012 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, 3, 4, 5);
        List<Pair<INDArray, String>> list021 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, 3, 4, 5);
        List<Pair<INDArray, String>> list120 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, 3, 4, 5);
        List<Pair<INDArray, String>> list102 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, 3, 4, 5);
        List<Pair<INDArray, String>> list201 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, 3, 4, 5);
        List<Pair<INDArray, String>> list210 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, 3, 4, 5);

        for (int i = 0; i < list012.size(); i++) {
            INDArray p1 = list012.get(i).getFirst().assign(orig3d).permutei(0, 1, 2);
            INDArray p2 = list021.get(i).getFirst().assign(orig3d).permutei(0, 2, 1);
            INDArray p3 = list120.get(i).getFirst().assign(orig3d).permutei(1, 2, 0);
            INDArray p4 = list102.get(i).getFirst().assign(orig3d).permutei(1, 0, 2);
            INDArray p5 = list201.get(i).getFirst().assign(orig3d).permutei(2, 0, 1);
            INDArray p6 = list210.get(i).getFirst().assign(orig3d).permutei(2, 1, 0);

            assertEquals(exp012, p1);
            assertEquals(exp021, p2);
            assertEquals(exp120, p3);
            assertEquals(exp102, p4);
            assertEquals(exp201, p5);
            assertEquals(exp210, p6);
        }
    }


    @Test
    public void testPermuteiShape() {

        INDArray row = Nd4j.create(1, 10);

        INDArray permutedCopy = row.permute(1, 0);
        INDArray permutedInplace = row.permutei(1, 0);

        assertArrayEquals(new long[] {10, 1}, permutedCopy.shape());
        assertArrayEquals(new long[] {10, 1}, permutedInplace.shape());

        assertEquals(10, permutedCopy.rows());
        assertEquals(10, permutedInplace.rows());

        assertEquals(1, permutedCopy.columns());
        assertEquals(1, permutedInplace.columns());


        INDArray col = Nd4j.create(10, 1);
        INDArray cPermutedCopy = col.permute(1, 0);
        INDArray cPermutedInplace = col.permutei(1, 0);

        assertArrayEquals(new long[] {1, 10}, cPermutedCopy.shape());
        assertArrayEquals(new long[] {1, 10}, cPermutedInplace.shape());

        assertEquals(1, cPermutedCopy.rows());
        assertEquals(1, cPermutedInplace.rows());

        assertEquals(10, cPermutedCopy.columns());
        assertEquals(10, cPermutedInplace.columns());
    }



    @Test
    public void testSwapAxes() {
        INDArray n = Nd4j.create(Nd4j.linspace(0, 7, 8).data(), new long[] {2, 2, 2});
        INDArray assertion = n.permute(2, 1, 0);
        INDArray permuteTranspose = assertion.slice(1).slice(1);
        INDArray validate = Nd4j.create(new float[] {0, 4, 2, 6, 1, 5, 3, 7}, new long[] {2, 2, 2});
        assertEquals(validate, assertion);

        INDArray thirty = Nd4j.linspace(1, 30, 30).reshape(3, 5, 2);
        INDArray swapped = thirty.swapAxes(2, 1);
        INDArray slice = swapped.slice(0).slice(0);
        INDArray assertion2 = Nd4j.create(new double[] {1, 3, 5, 7, 9});
        assertEquals(assertion2, slice);


    }


    @Test
    public void testMuliRowVector() {
        INDArray arrC = Nd4j.linspace(1, 6, 6).reshape('c', 3, 2);
        INDArray arrF = Nd4j.create(new long[] {3, 2}, 'f').assign(arrC);

        INDArray temp = Nd4j.create(new long[] {2, 11}, 'c');
        INDArray vec = temp.get(NDArrayIndex.all(), NDArrayIndex.interval(9, 10)).transpose();
        vec.assign(Nd4j.linspace(1, 2, 2));

        //Passes if we do one of these...
        //        vec = vec.dup('c');
        //        vec = vec.dup('f');

        System.out.println("Vec: " + vec);

        INDArray outC = arrC.muliRowVector(vec);
        INDArray outF = arrF.muliRowVector(vec);

        double[][] expD = new double[][] {{1, 4}, {3, 8}, {5, 12}};
        INDArray exp = Nd4j.create(expD);

        assertEquals(exp, outC);
        assertEquals(exp, outF);
    }

    @Test
    public void testSliceConstructor() throws Exception {
        List<INDArray> testList = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            testList.add(Nd4j.scalar(i + 1));

        INDArray test = Nd4j.create(testList, new long[] {1, testList.size()}).reshape(1, 5);
        INDArray expected = Nd4j.create(new float[] {1, 2, 3, 4, 5}, new long[] {1, 5});
        assertEquals(expected, test);
    }



    @Test
    public void testStdev0() {
        double[][] ind = {{5.1, 3.5, 1.4}, {4.9, 3.0, 1.4}, {4.7, 3.2, 1.3}};
        INDArray in = Nd4j.create(ind);
        INDArray stdev = in.std(0);
        INDArray exp = Nd4j.create(new double[] {0.19999999999999973, 0.2516611478423583, 0.057735026918962505});

        assertEquals(exp, stdev);
    }

    @Test
    public void testStdev1() {
        double[][] ind = {{5.1, 3.5, 1.4}, {4.9, 3.0, 1.4}, {4.7, 3.2, 1.3}};
        INDArray in = Nd4j.create(ind);
        INDArray stdev = in.std(1);
        log.info("StdDev: {}", stdev.toDoubleVector());
        INDArray exp = Nd4j.create(new double[] {1.8556220879622372, 1.7521415467935233, 1.7039170558842744});
        assertEquals(exp, stdev);
    }


    @Test
    public void testSignXZ() {
        double[] d = {1.0, -1.1, 1.2, 1.3, -1.4, -1.5, 1.6, -1.7, -1.8, -1.9, -1.01, -1.011};
        double[] e = {1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

        INDArray arrF = Nd4j.create(d, new long[] {4, 3}, 'f');
        INDArray arrC = Nd4j.create(new long[] {4, 3}, 'c').assign(arrF);

        INDArray exp = Nd4j.create(e, new long[] {4, 3}, 'f');

        //First: do op with just x (inplace)
        INDArray arrFCopy = arrF.dup('f');
        INDArray arrCCopy = arrC.dup('c');
        Nd4j.getExecutioner().exec(new Sign(arrFCopy));
        Nd4j.getExecutioner().exec(new Sign(arrCCopy));
        assertEquals(exp, arrFCopy);
        assertEquals(exp, arrCCopy);

        //Second: do op with both x and z:
        INDArray zOutFC = Nd4j.create(new long[] {4, 3}, 'c');
        INDArray zOutFF = Nd4j.create(new long[] {4, 3}, 'f');
        INDArray zOutCC = Nd4j.create(new long[] {4, 3}, 'c');
        INDArray zOutCF = Nd4j.create(new long[] {4, 3}, 'f');
        Nd4j.getExecutioner().exec(new Sign(arrF, zOutFC));
        Nd4j.getExecutioner().exec(new Sign(arrF, zOutFF));
        Nd4j.getExecutioner().exec(new Sign(arrC, zOutCC));
        Nd4j.getExecutioner().exec(new Sign(arrC, zOutCF));

        assertEquals(exp, zOutFC); //fails
        assertEquals(exp, zOutFF); //pass
        assertEquals(exp, zOutCC); //pass
        assertEquals(exp, zOutCF); //fails
    }

    @Test
    public void testTanhXZ() {
        INDArray arrC = Nd4j.linspace(-6, 6, 12).reshape('c', 4, 3);
        INDArray arrF = Nd4j.create(new long[] {4, 3}, 'f').assign(arrC);
        double[] d = arrC.data().asDouble();
        double[] e = new double[d.length];
        for (int i = 0; i < e.length; i++)
            e[i] = Math.tanh(d[i]);

        INDArray exp = Nd4j.create(e, new long[] {4, 3}, 'c');

        //First: do op with just x (inplace)
        INDArray arrFCopy = arrF.dup('f');
        INDArray arrCCopy = arrF.dup('c');
        Nd4j.getExecutioner().exec(new Tanh(arrFCopy));
        Nd4j.getExecutioner().exec(new Tanh(arrCCopy));
        assertEquals(exp, arrFCopy);
        assertEquals(exp, arrCCopy);

        //Second: do op with both x and z:
        INDArray zOutFC = Nd4j.create(new long[] {4, 3}, 'c');
        INDArray zOutFF = Nd4j.create(new long[] {4, 3}, 'f');
        INDArray zOutCC = Nd4j.create(new long[] {4, 3}, 'c');
        INDArray zOutCF = Nd4j.create(new long[] {4, 3}, 'f');
        Nd4j.getExecutioner().exec(new Tanh(arrF, zOutFC));
        Nd4j.getExecutioner().exec(new Tanh(arrF, zOutFF));
        Nd4j.getExecutioner().exec(new Tanh(arrC, zOutCC));
        Nd4j.getExecutioner().exec(new Tanh(arrC, zOutCF));

        assertEquals(exp, zOutFC); //fails
        assertEquals(exp, zOutFF); //pass
        assertEquals(exp, zOutCC); //pass
        assertEquals(exp, zOutCF); //fails
    }


    @Test
    public void testBroadcastDiv() {
        INDArray num = Nd4j.create(new double[] {1.00, 1.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 1.00, 1.00, 1.00, 1.00,
                2.00, 2.00, 2.00, 2.00, -1.00, -1.00, -1.00, -1.00, -2.00, -2.00, -2.00, -2.00, -1.00, -1.00,
                -1.00, -1.00, -2.00, -2.00, -2.00, -2.00}).reshape(2, 16);

        INDArray denom = Nd4j.create(new double[] {1.00, 1.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 1.00, 1.00, 1.00,
                1.00, 2.00, 2.00, 2.00, 2.00});

        INDArray expected = Nd4j.create(
                new double[] {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., -1., -1.,
                        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,},
                new long[] {2, 16});

        INDArray actual = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(num, denom, num.dup(), -1));
        assertEquals(expected, actual);
    }


    @Test
    public void testBroadcastMult() {
        INDArray num = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, -1.00, -2.00, -3.00,
                -4.00, -5.00, -6.00, -7.00, -8.00}).reshape(2, 8);

        INDArray denom = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00});

        INDArray expected = Nd4j.create(new double[] {1, 4, 9, 16, 25, 36, 49, 64, -1, -4, -9, -16, -25, -36, -49, -64},
                new long[] {2, 8});

        INDArray actual = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(num, denom, num.dup(), -1));
        assertEquals(expected, actual);
    }

    @Test
    public void testBroadcastSub() {
        INDArray num = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, -1.00, -2.00, -3.00,
                -4.00, -5.00, -6.00, -7.00, -8.00}).reshape(2, 8);

        INDArray denom = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00});

        INDArray expected = Nd4j.create(new double[] {0, 0, 0, 0, 0, 0, 0, 0, -2, -4, -6, -8, -10, -12, -14, -16},
                new long[] {2, 8});

        INDArray actual = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(num, denom, num.dup(), -1));
        assertEquals(expected, actual);
    }

    @Test
    public void testBroadcastAdd() {
        INDArray num = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, -1.00, -2.00, -3.00,
                -4.00, -5.00, -6.00, -7.00, -8.00}).reshape(2, 8);

        INDArray denom = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00});

        INDArray expected = Nd4j.create(new double[] {2, 4, 6, 8, 10, 12, 14, 16, 0, 0, 0, 0, 0, 0, 0, 0,},
                new long[] {2, 8});
        INDArray dup = num.dup();
        INDArray actual = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(num, denom, dup, -1));
        assertEquals(expected, actual);
    }


    @Test
    public void testDimension() {
        INDArray test = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new long[] {2, 2});
        //row
        INDArray slice0 = test.slice(0, 1);
        INDArray slice02 = test.slice(1, 1);

        INDArray assertSlice0 = Nd4j.create(new float[] {1, 3});
        INDArray assertSlice02 = Nd4j.create(new float[] {2, 4});
        assertEquals(assertSlice0, slice0);
        assertEquals(assertSlice02, slice02);

        //column
        INDArray assertSlice1 = Nd4j.create(new float[] {1, 2});
        INDArray assertSlice12 = Nd4j.create(new float[] {3, 4});


        INDArray slice1 = test.slice(0, 0);
        INDArray slice12 = test.slice(1, 0);


        assertEquals(assertSlice1, slice1);
        assertEquals(assertSlice12, slice12);


        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new long[] {4, 3, 2});
        INDArray secondSliceFirstDimension = arr.slice(1, 1);
        assertEquals(secondSliceFirstDimension, secondSliceFirstDimension);


    }



    @Test
    public void testReshape() {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new long[] {4, 3, 2});
        INDArray reshaped = arr.reshape(2, 3, 4);
        assertEquals(arr.length(), reshaped.length());
        assertEquals(true, Arrays.equals(new long[] {4, 3, 2}, arr.shape()));
        assertEquals(true, Arrays.equals(new long[] {2, 3, 4}, reshaped.shape()));

    }



    @Test
    public void testDot() {
        INDArray vec1 = Nd4j.create(new float[] {1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new float[] {1, 2, 3, 4});
        assertEquals(30, Nd4j.getBlasWrapper().dot(vec1, vec2), 1e-1);

        INDArray matrix = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row = matrix.getRow(1);
        assertEquals(25, Nd4j.getBlasWrapper().dot(row, row), 1e-1);

    }


    @Test
    public void testIdentity() {
        INDArray eye = Nd4j.eye(5);
        assertTrue(Arrays.equals(new long[] {5, 5}, eye.shape()));
        eye = Nd4j.eye(5);
        assertTrue(Arrays.equals(new long[] {5, 5}, eye.shape()));


    }

    @Test
    public void testTemp() {
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(new long[] {2, 2, 2});
        System.out.println("In:\n" + in);
        INDArray permuted = in.permute(0, 2, 1); //Permute, so we get correct order after reshaping
        INDArray out = permuted.reshape(4, 2);
        System.out.println("Out:\n" + out);

        int countZero = 0;
        for (int i = 0; i < 8; i++)
            if (out.getDouble(i) == 0.0)
                countZero++;
        assertEquals(countZero, 0);
    }


    @Test
    public void testMeans() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray mean1 = a.mean(1);
        assertEquals(getFailureMessage(), Nd4j.create(new double[] {1.5, 3.5}), mean1);
        assertEquals(getFailureMessage(), Nd4j.create(new double[] {2, 3}), a.mean(0));
        assertEquals(getFailureMessage(), 2.5, Nd4j.linspace(1, 4, 4).meanNumber().doubleValue(), 1e-1);
        assertEquals(getFailureMessage(), 2.5, a.meanNumber().doubleValue(), 1e-1);

    }


    @Test
    public void testSums() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        assertEquals(getFailureMessage(), Nd4j.create(new float[] {3, 7}), a.sum(1));
        assertEquals(getFailureMessage(), Nd4j.create(new float[] {4, 6}), a.sum(0));
        assertEquals(getFailureMessage(), 10, a.sumNumber().doubleValue(), 1e-1);


    }

    @Test
    public void testRSubi() {
        INDArray n2 = Nd4j.ones(2);
        INDArray n2Assertion = Nd4j.zeros(2);
        INDArray nRsubi = n2.rsubi(1);
        assertEquals(n2Assertion, nRsubi);
    }


    @Test
    public void testConcat() {
        INDArray A = Nd4j.linspace(1, 8, 8).reshape(2, 2, 2);
        INDArray B = Nd4j.linspace(1, 12, 12).reshape(3, 2, 2);
        INDArray concat = Nd4j.concat(0, A, B);
        assertTrue(Arrays.equals(new long[] {5, 2, 2}, concat.shape()));

        INDArray columnConcat = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray concatWith = Nd4j.zeros(2, 3);
        INDArray columnWiseConcat = Nd4j.concat(0, columnConcat, concatWith);
        System.out.println(columnConcat);

    }

    @Test
    public void testConcatHorizontally() {
        INDArray rowVector = Nd4j.ones(5);
        INDArray other = Nd4j.ones(5);
        INDArray concat = Nd4j.hstack(other, rowVector);
        assertEquals(rowVector.rows(), concat.rows());
        assertEquals(rowVector.columns() * 2, concat.columns());

    }



    @Test
    public void testArgMaxSameValues() {
        //Here: assume that by convention, argmax returns the index of the FIRST maximum value
        //Thus, argmax(ones(...)) = 0 by convention
        INDArray arr = Nd4j.ones(10);

        for (int i = 0; i < 10; i++) {
            double argmax = Nd4j.argMax(arr, 1).getDouble(0);
            //System.out.println(argmax);
            assertEquals(0.0, argmax, 0.0);
        }
    }


    @Test
    public void testSoftmaxStability() {
        INDArray input = Nd4j.create(new double[] {-0.75, 0.58, 0.42, 1.03, -0.61, 0.19, -0.37, -0.40, -1.42, -0.04})
                .transpose();
        System.out.println("Input transpose " + Shape.shapeToString(input.shapeInfo()));
        INDArray output = Nd4j.create(10, 1);
        System.out.println("Element wise stride of output " + output.elementWiseStride());
        Nd4j.getExecutioner().exec(new OldSoftMax(input, output));
    }

    @Test
    public void testAssignOffset() {
        INDArray arr = Nd4j.ones(5, 5);
        INDArray row = arr.slice(1);
        row.assign(1);
        assertEquals(Nd4j.ones(5), row);
    }

    @Test
    public void testAddScalar() {
        INDArray div = Nd4j.valueArrayOf(new long[] {1, 4}, 4);
        INDArray rdiv = div.add(1);
        INDArray answer = Nd4j.valueArrayOf(new long[] {1, 4}, 5);
        assertEquals(answer, rdiv);
    }

    @Test
    public void testRdivScalar() {
        INDArray div = Nd4j.valueArrayOf(new long[] {1, 4}, 4);
        INDArray rdiv = div.rdiv(1);
        INDArray answer = Nd4j.valueArrayOf(new long[] {1, 4}, 0.25);
        assertEquals(rdiv, answer);
    }

    @Test
    public void testRDivi() {
        INDArray n2 = Nd4j.valueArrayOf(new long[] {1, 2}, 4);
        INDArray n2Assertion = Nd4j.valueArrayOf(new long[] {1, 2}, 0.5);
        INDArray nRsubi = n2.rdivi(2);
        assertEquals(n2Assertion, nRsubi);
    }



    @Test
    public void testElementWiseAdd() {
        INDArray linspace = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray linspace2 = linspace.dup();
        INDArray assertion = Nd4j.create(new double[][] {{2, 4}, {6, 8}});
        linspace.addi(linspace2);
        assertEquals(assertion, linspace);
    }

    @Test
    public void testSquareMatrix() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new long[] {2, 2, 2});
        INDArray eightFirstTest = n.vectorAlongDimension(0, 2);
        INDArray eightFirstAssertion = Nd4j.create(new float[] {1, 2}, new long[] {1, 2});
        assertEquals(eightFirstAssertion, eightFirstTest);

        INDArray eightFirstTestSecond = n.vectorAlongDimension(1, 2);
        INDArray eightFirstTestSecondAssertion = Nd4j.create(new float[] {3, 4});
        assertEquals(eightFirstTestSecondAssertion, eightFirstTestSecond);

    }

    @Test
    public void testNumVectorsAlongDimension() {
        INDArray arr = Nd4j.linspace(1, 24, 24).reshape(4, 3, 2);
        assertEquals(12, arr.vectorsAlongDimension(2));
    }



    @Test
    public void testBroadCast() {
        INDArray n = Nd4j.linspace(1, 4, 4);
        INDArray broadCasted = n.broadcast(5, 4);
        for (int i = 0; i < broadCasted.rows(); i++) {
            INDArray row = broadCasted.getRow(i);
            assertEquals(n, broadCasted.getRow(i));
        }

        INDArray broadCast2 = broadCasted.getRow(0).broadcast(5, 4);
        assertEquals(broadCasted, broadCast2);


        INDArray columnBroadcast = n.transpose().broadcast(4, 5);
        for (int i = 0; i < columnBroadcast.columns(); i++) {
            INDArray column = columnBroadcast.getColumn(i);
            assertEquals(column, n.transpose());
        }

        INDArray fourD = Nd4j.create(1, 2, 1, 1);
        INDArray broadCasted3 = fourD.broadcast(1, 2, 36, 36);
        assertTrue(Arrays.equals(new long[] {1, 2, 36, 36}, broadCasted3.shape()));



        INDArray ones = Nd4j.ones(1, 1, 1).broadcast(2, 1, 1);
        assertArrayEquals(new long[] {2, 1, 1}, ones.shape());
    }

    @Test
    public void testScalarBroadcast() {
        INDArray fiveThree = Nd4j.ones(5, 3);
        INDArray fiveThreeTest = Nd4j.scalar(1.0).broadcast(5, 3);
        assertEquals(fiveThree, fiveThreeTest);

    }


    @Test
    public void testPutRowGetRowOrdering() {
        INDArray row1 = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray put = Nd4j.create(new double[] {5, 6});
        row1.putRow(1, put);


        INDArray row1Fortran = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray putFortran = Nd4j.create(new double[] {5, 6});
        row1Fortran.putRow(1, putFortran);
        assertEquals(row1, row1Fortran);
        INDArray row1CTest = row1.getRow(1);
        INDArray row1FortranTest = row1Fortran.getRow(1);
        assertEquals(row1CTest, row1FortranTest);



    }



    @Test
    public void testElementWiseOps() {
        INDArray n1 = Nd4j.scalar(1);
        INDArray n2 = Nd4j.scalar(2);
        INDArray nClone = n1.add(n2);
        assertEquals(Nd4j.scalar(3), nClone);
        assertFalse(n1.add(n2).equals(n1));

        INDArray n3 = Nd4j.scalar(3);
        INDArray n4 = Nd4j.scalar(4);
        INDArray subbed = n4.sub(n3);
        INDArray mulled = n4.mul(n3);
        INDArray div = n4.div(n3);

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));
        assertEquals(Nd4j.scalar(1), subbed);
        assertEquals(Nd4j.scalar(12), mulled);
        assertEquals(Nd4j.scalar(1.333333333333333333333), div);
    }

    @Test
    public void testNdArrayCreation() {
        double delta = 1e-1;
        INDArray n1 = Nd4j.create(new double[] {0d, 1d, 2d, 3d}, new long[] {2, 2}, 'c');
        INDArray lv = n1.linearView();
        assertEquals(0d, lv.getDouble(0), delta);
        assertEquals(1d, lv.getDouble(1), delta);
        assertEquals(2d, lv.getDouble(2), delta);
        assertEquals(3d, lv.getDouble(3), delta);
    }

    @Test
    public void testToFlattenedWithOrder() {
        int[] firstShape = {10, 3};
        int firstLen = ArrayUtil.prod(firstShape);
        int[] secondShape = {2, 7};
        int secondLen = ArrayUtil.prod(secondShape);
        int[] thirdShape = {3, 3};
        int thirdLen = ArrayUtil.prod(thirdShape);
        INDArray firstC = Nd4j.linspace(1, firstLen, firstLen).reshape('c', firstShape);
        INDArray firstF = Nd4j.create(firstShape, 'f').assign(firstC);
        INDArray secondC = Nd4j.linspace(1, secondLen, secondLen).reshape('c', secondShape);
        INDArray secondF = Nd4j.create(secondShape, 'f').assign(secondC);
        INDArray thirdC = Nd4j.linspace(1, thirdLen, thirdLen).reshape('c', thirdShape);
        INDArray thirdF = Nd4j.create(thirdShape, 'f').assign(thirdC);


        assertEquals(firstC, firstF);
        assertEquals(secondC, secondF);
        assertEquals(thirdC, thirdF);

        INDArray cc = Nd4j.toFlattened('c', firstC, secondC, thirdC);
        INDArray cf = Nd4j.toFlattened('c', firstF, secondF, thirdF);
        assertEquals(cc, cf);

        INDArray cmixed = Nd4j.toFlattened('c', firstC, secondF, thirdF);
        assertEquals(cc, cmixed);

        INDArray fc = Nd4j.toFlattened('f', firstC, secondC, thirdC);
        assertNotEquals(cc, fc);

        INDArray ff = Nd4j.toFlattened('f', firstF, secondF, thirdF);
        assertEquals(fc, ff);

        INDArray fmixed = Nd4j.toFlattened('f', firstC, secondF, thirdF);
        assertEquals(fc, fmixed);
    }


    @Test
    public void testLeakyRelu() {
        INDArray arr = Nd4j.linspace(-1, 1, 10);
        double[] expected = new double[10];
        for (int i = 0; i < 10; i++) {
            double in = arr.getDouble(i);
            expected[i] = (in <= 0.0 ? 0.01 * in : in);
        }

        INDArray out = Nd4j.getExecutioner().execAndReturn(new LeakyReLU(arr, 0.01));

        INDArray exp = Nd4j.create(expected);
        assertEquals(exp, out);
    }

    @Test
    public void testSoftmaxRow() {
        for (int i = 0; i < 20; i++) {
            INDArray arr1 = Nd4j.zeros(100);
            Nd4j.getExecutioner().execAndReturn(new OldSoftMax(arr1));
            System.out.println(Arrays.toString(arr1.data().asFloat()));
        }
    }

    @Test
    public void testLeakyRelu2() {
        INDArray arr = Nd4j.linspace(-1, 1, 10);
        double[] expected = new double[10];
        for (int i = 0; i < 10; i++) {
            double in = arr.getDouble(i);
            expected[i] = (in <= 0.0 ? 0.01 * in : in);
        }

        INDArray out = Nd4j.getExecutioner().execAndReturn(new LeakyReLU(arr, 0.01));

        System.out.println("Expected: " + Arrays.toString(expected));
        System.out.println("Actual:   " + Arrays.toString(out.data().asDouble()));

        INDArray exp = Nd4j.create(expected);
        assertEquals(exp, out);
    }

    @Test
    public void testDupAndDupWithOrder() {
        List<Pair<INDArray, String>> testInputs =
                NDArrayCreationUtil.getAllTestMatricesWithShape(ordering(), 4, 5, 123);
        for (Pair<INDArray, String> pair : testInputs) {

            String msg = pair.getSecond();
            INDArray in = pair.getFirst();
            INDArray dup = in.dup();
            INDArray dupc = in.dup('c');
            INDArray dupf = in.dup('f');

            assertEquals(dup.ordering(), ordering());
            assertEquals(dupc.ordering(), 'c');
            assertEquals(dupf.ordering(), 'f');
            assertEquals(msg, in, dupc);
            assertEquals(msg, in, dupf);
        }
    }

    @Test
    public void testToOffsetZeroCopy() {
        List<Pair<INDArray, String>> testInputs =
                NDArrayCreationUtil.getAllTestMatricesWithShape(ordering(), 4, 5, 123);

        for (int i = 0; i < testInputs.size(); i++) {
            Pair<INDArray, String> pair = testInputs.get(i);
            String msg = pair.getSecond();
            msg += "Failed on " + i;
            INDArray in = pair.getFirst();
            INDArray dup = Shape.toOffsetZeroCopy(in, ordering());
            INDArray dupc = Shape.toOffsetZeroCopy(in, 'c');
            INDArray dupf = Shape.toOffsetZeroCopy(in, 'f');
            INDArray dupany = Shape.toOffsetZeroCopyAnyOrder(in);

            assertEquals(msg, in, dup);
            assertEquals(msg, in, dupc);
            assertEquals(msg, in, dupf);
            assertEquals(msg, dupc.ordering(), 'c');
            assertEquals(msg, dupf.ordering(), 'f');
            assertEquals(msg, in, dupany);

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

    @Test
    public void testTensorStats() {
        List<Pair<INDArray, String>> testInputs = NDArrayCreationUtil.getAllTestMatricesWithShape(9, 13, 123);

        for (Pair<INDArray, String> pair : testInputs) {
            INDArray arr = pair.getFirst();
            String msg = pair.getSecond();

            val nTAD0 = arr.tensorssAlongDimension(0);
            val nTAD1 = arr.tensorssAlongDimension(1);

            OpExecutionerUtil.Tensor1DStats t0 = OpExecutionerUtil.get1DTensorStats(arr, 0);
            OpExecutionerUtil.Tensor1DStats t1 = OpExecutionerUtil.get1DTensorStats(arr, 1);

            assertEquals(nTAD0, t0.getNumTensors());
            assertEquals(nTAD1, t1.getNumTensors());

            INDArray tFirst0 = arr.tensorAlongDimension(0, 0);
            INDArray tSecond0 = arr.tensorAlongDimension(1, 0);

            INDArray tFirst1 = arr.tensorAlongDimension(0, 1);
            INDArray tSecond1 = arr.tensorAlongDimension(1, 1);

            assertEquals(tFirst0.offset(), t0.getFirstTensorOffset());
            assertEquals(tFirst1.offset(), t1.getFirstTensorOffset());
            long separation0 = tSecond0.offset() - tFirst0.offset();
            long separation1 = tSecond1.offset() - tFirst1.offset();
            assertEquals(separation0, t0.getTensorStartSeparation());
            assertEquals(separation1, t1.getTensorStartSeparation());

            for (int i = 0; i < nTAD0; i++) {
                INDArray tad0 = arr.tensorAlongDimension(i, 0);
                assertEquals(tad0.length(), t0.getTensorLength());
                assertEquals(tad0.elementWiseStride(), t0.getElementWiseStride());

                long offset = tad0.offset();
                long calcOffset = t0.getFirstTensorOffset() + i * t0.getTensorStartSeparation();
                assertEquals(offset, calcOffset);
            }

            for (int i = 0; i < nTAD1; i++) {
                INDArray tad1 = arr.tensorAlongDimension(i, 1);
                assertEquals(tad1.length(), t1.getTensorLength());
                assertEquals(tad1.elementWiseStride(), t1.getElementWiseStride());

                long offset = tad1.offset();
                long calcOffset = t1.getFirstTensorOffset() + i * t1.getTensorStartSeparation();
                assertEquals(offset, calcOffset);
            }
        }
    }



    @Test
    @Ignore
    public void largeInstantiation() {
        Nd4j.ones((1024 * 1024 * 511) + (1024 * 1024 - 1)); // Still works; this can even be called as often as I want, allowing me even to spill over on disk
        Nd4j.ones((1024 * 1024 * 511) + (1024 * 1024)); // Crashes
    }

    @Test
    public void testAssignNumber() {
        int nRows = 10;
        int nCols = 20;
        INDArray in = Nd4j.linspace(1, nRows * nCols, nRows * nCols).reshape('c', new long[] {nRows, nCols});

        INDArray subset1 = in.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(0, nCols / 2));
        subset1.assign(1.0);

        INDArray subset2 = in.get(NDArrayIndex.interval(5, 8), NDArrayIndex.interval(nCols / 2, nCols));
        subset2.assign(2.0);
        INDArray assertion = Nd4j.create(new double[] {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
                29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0,
                45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0,
                77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0,
                93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0,
                107.0, 108.0, 109.0, 110.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 121.0, 122.0,
                123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0,
                169.0, 170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, 180.0, 181.0,
                182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0, 191.0, 192.0, 193.0, 194.0,
                195.0, 196.0, 197.0, 198.0, 199.0, 200.0}, in.shape(), 0, 'c');
        assertEquals(assertion, in);
    }


    @Test
    public void testSumDifferentOrdersSquareMatrix() {
        INDArray arrc = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray arrf = Nd4j.create(new long[] {2, 2}, 'f').assign(arrc);

        INDArray cSum = arrc.sum(0);
        INDArray fSum = arrf.sum(0);
        assertEquals(arrc, arrf);
        assertEquals(cSum, fSum); //Expect: 4,6. Getting [4, 4] for f order
    }

    @Test
    public void testAssignMixedC() {
        int[] shape1 = {3, 2, 2, 2, 2, 2};
        int[] shape2 = {12, 8};
        int length = ArrayUtil.prod(shape1);

        assertEquals(ArrayUtil.prod(shape1), ArrayUtil.prod(shape2));

        INDArray arr = Nd4j.linspace(1, length, length).reshape('c', shape1);
        INDArray arr2c = Nd4j.create(shape2, 'c');
        INDArray arr2f = Nd4j.create(shape2, 'f');

        log.info("2f data: {}", Arrays.toString(arr2f.data().asFloat()));

        arr2c.assign(arr);
        System.out.println("--------------");
        arr2f.assign(arr);

        INDArray exp = Nd4j.linspace(1, length, length).reshape('c', shape2);

        log.info("arr data: {}", Arrays.toString(arr.data().asFloat()));
        log.info("2c data: {}", Arrays.toString(arr2c.data().asFloat()));
        log.info("2f data: {}", Arrays.toString(arr2f.data().asFloat()));
        log.info("2c shape: {}", Arrays.toString(arr2c.shapeInfoDataBuffer().asInt()));
        log.info("2f shape: {}", Arrays.toString(arr2f.shapeInfoDataBuffer().asInt()));
        assertEquals(exp, arr2c);
        assertEquals(exp, arr2f);
    }

    @Test
    public void testDummy() {
        INDArray arr2f = Nd4j.create(new double[] {1.0, 13.0, 25.0, 37.0, 49.0, 61.0, 73.0, 85.0, 2.0, 14.0, 26.0, 38.0,
                50.0, 62.0, 74.0, 86.0, 3.0, 15.0, 27.0, 39.0, 51.0, 63.0, 75.0, 87.0, 4.0, 16.0, 28.0, 40.0,
                52.0, 64.0, 76.0, 88.0, 5.0, 17.0, 29.0, 41.0, 53.0, 65.0, 77.0, 89.0, 6.0, 18.0, 30.0, 42.0,
                54.0, 66.0, 78.0, 90.0, 7.0, 19.0, 31.0, 43.0, 55.0, 67.0, 79.0, 91.0, 8.0, 20.0, 32.0, 44.0,
                56.0, 68.0, 80.0, 92.0, 9.0, 21.0, 33.0, 45.0, 57.0, 69.0, 81.0, 93.0, 10.0, 22.0, 34.0, 46.0,
                58.0, 70.0, 82.0, 94.0, 11.0, 23.0, 35.0, 47.0, 59.0, 71.0, 83.0, 95.0, 12.0, 24.0, 36.0, 48.0,
                60.0, 72.0, 84.0, 96.0}, new long[] {12, 8}, 'f');
        log.info("arr2f shape: {}", Arrays.toString(arr2f.shapeInfoDataBuffer().asInt()));
        log.info("arr2f data: {}", Arrays.toString(arr2f.data().asFloat()));
        log.info("render: {}", arr2f);

        log.info("----------------------");

        INDArray array = Nd4j.linspace(1, 96, 96).reshape('c', 12, 8);
        log.info("array render: {}", array);

        log.info("----------------------");

        INDArray arrayf = array.dup('f');
        log.info("arrayf render: {}", arrayf);
        log.info("arrayf shape: {}", Arrays.toString(arrayf.shapeInfoDataBuffer().asInt()));
        log.info("arrayf data: {}", Arrays.toString(arrayf.data().asFloat()));
    }

    @Test
    public void testPairwiseMixedC() {
        int[] shape2 = {12, 8};
        int length = ArrayUtil.prod(shape2);


        INDArray arr = Nd4j.linspace(1, length, length).reshape('c', shape2);
        INDArray arr2c = arr.dup('c');
        INDArray arr2f = arr.dup('f');

        arr2c.addi(arr);
        System.out.println("--------------");
        arr2f.addi(arr);

        INDArray exp = Nd4j.linspace(1, length, length).reshape('c', shape2).mul(2.0);

        assertEquals(exp, arr2c);
        assertEquals(exp, arr2f);

        log.info("2c data: {}", Arrays.toString(arr2c.data().asFloat()));
        log.info("2f data: {}", Arrays.toString(arr2f.data().asFloat()));

        assertTrue(arrayNotEquals(arr2c.data().asFloat(), arr2f.data().asFloat(), 1e-5f));
    }

    @Test
    public void testPairwiseMixedF() {
        int[] shape2 = {12, 8};
        int length = ArrayUtil.prod(shape2);


        INDArray arr = Nd4j.linspace(1, length, length).reshape('c', shape2).dup('f');
        INDArray arr2c = arr.dup('c');
        INDArray arr2f = arr.dup('f');

        arr2c.addi(arr);
        System.out.println("--------------");
        arr2f.addi(arr);

        INDArray exp = Nd4j.linspace(1, length, length).reshape('c', shape2).dup('f').mul(2.0);

        assertEquals(exp, arr2c);
        assertEquals(exp, arr2f);

        log.info("2c data: {}", Arrays.toString(arr2c.data().asFloat()));
        log.info("2f data: {}", Arrays.toString(arr2f.data().asFloat()));

        assertTrue(arrayNotEquals(arr2c.data().asFloat(), arr2f.data().asFloat(), 1e-5f));
    }

    @Test
    public void testAssign2D() {
        int[] shape2 = {8, 4};

        int length = ArrayUtil.prod(shape2);

        INDArray arr = Nd4j.linspace(1, length, length).reshape('c', shape2);
        INDArray arr2c = Nd4j.create(shape2, 'c');
        INDArray arr2f = Nd4j.create(shape2, 'f');

        arr2c.assign(arr);
        System.out.println("--------------");
        arr2f.assign(arr);

        INDArray exp = Nd4j.linspace(1, length, length).reshape('c', shape2);

        assertEquals(exp, arr2c);
        assertEquals(exp, arr2f);
    }

    @Test
    public void testAssign2D_2() {
        int[] shape2 = {8, 4};

        int length = ArrayUtil.prod(shape2);

        INDArray arr = Nd4j.linspace(1, length, length).reshape('c', shape2);
        INDArray arr2c = Nd4j.create(shape2, 'c');
        INDArray arr2f = Nd4j.create(shape2, 'f');
        INDArray z_f = Nd4j.create(shape2, 'f');
        INDArray z_c = Nd4j.create(shape2, 'c');

        Nd4j.getExecutioner().exec(new Set(arr2f, arr, z_f, arr2c.length()));

        Nd4j.getExecutioner().commit();

        Nd4j.getExecutioner().exec(new Set(arr2f, arr, z_c, arr2c.length()));

        INDArray exp = Nd4j.linspace(1, length, length).reshape('c', shape2);


        System.out.println("Zf data: " + Arrays.toString(z_f.data().asFloat()));
        System.out.println("Zc data: " + Arrays.toString(z_c.data().asFloat()));

        assertEquals(exp, z_f);
        assertEquals(exp, z_c);
    }

    @Test
    public void testAssign3D_2() {
        int[] shape3 = {8, 4, 8};

        int length = ArrayUtil.prod(shape3);

        INDArray arr = Nd4j.linspace(1, length, length).reshape('c', shape3).dup('f');
        INDArray arr3c = Nd4j.create(shape3, 'c');
        INDArray arr3f = Nd4j.create(shape3, 'f');

        Nd4j.getExecutioner().exec(new Set(arr3c, arr, arr3f, arr3c.length()));

        Nd4j.getExecutioner().commit();

        Nd4j.getExecutioner().exec(new Set(arr3f, arr, arr3c, arr3c.length()));

        INDArray exp = Nd4j.linspace(1, length, length).reshape('c', shape3);

        assertEquals(exp, arr3c);
        assertEquals(exp, arr3f);
    }

    @Test
    public void testSumDifferentOrders() {
        INDArray arrc = Nd4j.linspace(1, 6, 6).reshape('c', 3, 2);
        INDArray arrf = Nd4j.create(new double[6], new long[] {3, 2}, 'f').assign(arrc);

        assertEquals(arrc, arrf);
        INDArray cSum = arrc.sum(0);
        INDArray fSum = arrf.sum(0);
        assertEquals(cSum, fSum); //Expect: 0.51, 1.79; getting [0.51,1.71] for f order
    }

    @Test
    public void testCreateUnitialized() {

        INDArray arrC = Nd4j.createUninitialized(new long[] {10, 10}, 'c');
        INDArray arrF = Nd4j.createUninitialized(new long[] {10, 10}, 'f');

        assertEquals('c', arrC.ordering());
        assertArrayEquals(new long[] {10, 10}, arrC.shape());
        assertEquals('f', arrF.ordering());
        assertArrayEquals(new long[] {10, 10}, arrF.shape());

        //Can't really test that it's *actually* uninitialized...
        arrC.assign(0);
        arrF.assign(0);

        assertEquals(Nd4j.create(new long[] {10, 10}), arrC);
        assertEquals(Nd4j.create(new long[] {10, 10}), arrF);
    }

    @Test
    public void testVarConst() {
        INDArray x = Nd4j.linspace(1, 100, 100).reshape(10, 10);
        System.out.println(x);
        assertFalse(Double.isNaN(x.var(0).sumNumber().doubleValue()));
        System.out.println(x.var(0));
        assertFalse(Double.isNaN(x.var(1).sumNumber().doubleValue()));
        System.out.println(x.var(1));

        System.out.println("=================================");
        // 2d array - all elements are the same
        INDArray a = Nd4j.ones(10, 10).mul(10);
        System.out.println(a);
        assertFalse(Double.isNaN(a.var(0).sumNumber().doubleValue()));
        System.out.println(a.var(0));
        assertFalse(Double.isNaN(a.var(1).sumNumber().doubleValue()));
        System.out.println(a.var(1));

        // 2d array - constant in one dimension
        System.out.println("=================================");
        INDArray nums = Nd4j.linspace(1, 10, 10);
        INDArray b = Nd4j.ones(10, 10).mulRowVector(nums);
        System.out.println(b);
        assertFalse(Double.isNaN((Double) b.var(0).sumNumber()));
        System.out.println(b.var(0));
        assertFalse(Double.isNaN((Double) b.var(1).sumNumber()));
        System.out.println(b.var(1));

        System.out.println("=================================");
        System.out.println(b.transpose());
        assertFalse(Double.isNaN((Double) b.transpose().var(0).sumNumber()));
        System.out.println(b.transpose().var(0));
        assertFalse(Double.isNaN((Double) b.transpose().var(1).sumNumber()));
        System.out.println(b.transpose().var(1));
    }

    @Test
    public void testVPull1() {
        int indexes[] = new int[] {0, 2, 4};
        INDArray array = Nd4j.linspace(1, 25, 25).reshape(5, 5);
        INDArray assertion = Nd4j.createUninitialized(new long[] {3, 5}, 'f');
        for (int i = 0; i < 3; i++) {
            assertion.putRow(i, array.getRow(indexes[i]));
        }

        INDArray result = Nd4j.pullRows(array, 1, indexes, 'f');

        assertEquals(3, result.rows());
        assertEquals(5, result.columns());
        assertEquals(assertion, result);
    }

    @Test(expected = IllegalStateException.class)
    public void testPullRowsValidation1() {
        Nd4j.pullRows(Nd4j.create(10, 10), 2, new int[] {0, 1, 2});
    }

    @Test(expected = IllegalStateException.class)
    public void testPullRowsValidation2() {
        Nd4j.pullRows(Nd4j.create(10, 10), 1, new int[] {0, -1, 2});
    }

    @Test(expected = IllegalStateException.class)
    public void testPullRowsValidation3() {
        Nd4j.pullRows(Nd4j.create(10, 10), 1, new int[] {0, 1, 10});
    }

    @Test(expected = IllegalStateException.class)
    public void testPullRowsValidation4() {
        Nd4j.pullRows(Nd4j.create(3, 10), 1, new int[] {0, 1, 2, 3});
    }

    @Test(expected = IllegalStateException.class)
    public void testPullRowsValidation5() {
        Nd4j.pullRows(Nd4j.create(3, 10), 1, new int[] {0, 1, 2}, 'e');
    }



    @Test
    public void testVPull2() {
        val indexes = new int[] {0, 2, 4};
        INDArray array = Nd4j.linspace(1, 25, 25).reshape(5, 5);
        INDArray assertion = Nd4j.createUninitialized(new long[] {3, 5}, 'c');
        for (int i = 0; i < 3; i++) {
            assertion.putRow(i, array.getRow(indexes[i]));
        }

        INDArray result = Nd4j.pullRows(array, 1, indexes, 'c');

        assertEquals(3, result.rows());
        assertEquals(5, result.columns());
        assertEquals(assertion, result);

        System.out.println(assertion.toString());
        System.out.println(result.toString());
    }


    @Test
    public void testCompareAndSet1() {
        INDArray array = Nd4j.zeros(25);

        INDArray assertion = Nd4j.zeros(25);

        array.putScalar(0, 0.1f);
        array.putScalar(10, 0.1f);
        array.putScalar(20, 0.1f);

        Nd4j.getExecutioner().exec(new CompareAndSet(array, 0.1, 0.0, 0.01));

        assertEquals(assertion, array);
    }

    @Test
    public void testReplaceNaNs() {
        INDArray array = Nd4j.zeros(25);
        INDArray assertion = Nd4j.zeros(25);

        array.putScalar(0, Float.NaN);
        array.putScalar(10, Float.NaN);
        array.putScalar(20, Float.NaN);

        assertNotEquals(assertion, array);

        Nd4j.getExecutioner().exec(new ReplaceNans(array, 0.0));

        System.out.println("Array After: " + array);

        assertEquals(assertion, array);
    }

    @Test
    public void testNaNEquality() {
        INDArray array = Nd4j.zeros(25);
        INDArray assertion = Nd4j.zeros(25);

        array.putScalar(0, Float.NaN);
        array.putScalar(10, Float.NaN);
        array.putScalar(20, Float.NaN);

        assertNotEquals(assertion, array);
    }


    @Test
    public void testSingleDeviceAveraging() throws Exception {
        int LENGTH = 512 * 1024 * 2;
        INDArray array1 = Nd4j.valueArrayOf(LENGTH, 1.0);
        INDArray array2 = Nd4j.valueArrayOf(LENGTH, 2.0);
        INDArray array3 = Nd4j.valueArrayOf(LENGTH, 3.0);
        INDArray array4 = Nd4j.valueArrayOf(LENGTH, 4.0);
        INDArray array5 = Nd4j.valueArrayOf(LENGTH, 5.0);
        INDArray array6 = Nd4j.valueArrayOf(LENGTH, 6.0);
        INDArray array7 = Nd4j.valueArrayOf(LENGTH, 7.0);
        INDArray array8 = Nd4j.valueArrayOf(LENGTH, 8.0);
        INDArray array9 = Nd4j.valueArrayOf(LENGTH, 9.0);
        INDArray array10 = Nd4j.valueArrayOf(LENGTH, 10.0);
        INDArray array11 = Nd4j.valueArrayOf(LENGTH, 11.0);
        INDArray array12 = Nd4j.valueArrayOf(LENGTH, 12.0);
        INDArray array13 = Nd4j.valueArrayOf(LENGTH, 13.0);
        INDArray array14 = Nd4j.valueArrayOf(LENGTH, 14.0);
        INDArray array15 = Nd4j.valueArrayOf(LENGTH, 15.0);
        INDArray array16 = Nd4j.valueArrayOf(LENGTH, 16.0);


        long time1 = System.currentTimeMillis();
        INDArray arrayMean = Nd4j.averageAndPropagate(new INDArray[] {array1, array2, array3, array4, array5, array6,
                array7, array8, array9, array10, array11, array12, array13, array14, array15, array16});
        long time2 = System.currentTimeMillis();
        System.out.println("Execution time: " + (time2 - time1));

        assertNotEquals(null, arrayMean);

        assertEquals(8.5f, arrayMean.getFloat(12), 0.1f);
        assertEquals(8.5f, arrayMean.getFloat(150), 0.1f);
        assertEquals(8.5f, arrayMean.getFloat(475), 0.1f);


        assertEquals(8.5f, array1.getFloat(475), 0.1f);
        assertEquals(8.5f, array2.getFloat(475), 0.1f);
        assertEquals(8.5f, array3.getFloat(475), 0.1f);
        assertEquals(8.5f, array5.getFloat(475), 0.1f);
        assertEquals(8.5f, array16.getFloat(475), 0.1f);
    }


    @Test
    public void testDistance1and2() {
        double[] d1 = new double[] {-1, 3, 2};
        double[] d2 = new double[] {0, 1.5, -3.5};
        INDArray arr1 = Nd4j.create(d1);
        INDArray arr2 = Nd4j.create(d2);

        double expD1 = 0.0;
        double expD2 = 0.0;
        for (int i = 0; i < d1.length; i++) {
            double diff = d1[i] - d2[i];
            expD1 += Math.abs(diff);
            expD2 += diff * diff;
        }
        expD2 = Math.sqrt(expD2);

        assertEquals(expD1, arr1.distance1(arr2), 1e-5);
        assertEquals(expD2, arr1.distance2(arr2), 1e-5);
        assertEquals(expD2 * expD2, arr1.squaredDistance(arr2), 1e-5);
    }

    @Test
    public void testEqualsWithEps1() throws Exception {
        INDArray array1 = Nd4j.create(new float[] {0.5f, 1.5f, 2.5f, 3.5f, 4.5f});
        INDArray array2 = Nd4j.create(new float[] {0f, 1f, 2f, 3f, 4f});
        INDArray array3 = Nd4j.create(new float[] {0f, 1.000001f, 2f, 3f, 4f});


        assertFalse(array1.equalsWithEps(array2, Nd4j.EPS_THRESHOLD));
        assertTrue(array2.equalsWithEps(array3, Nd4j.EPS_THRESHOLD));
        assertTrue(array1.equalsWithEps(array2, 0.7f));
        assertEquals(array2, array3);
    }

    @Test
    public void testIMaxIAMax() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ALL);

        INDArray arr = Nd4j.create(new double[] {-0.24, -0.26, -0.07, -0.01});
        IMax iMax = new IMax(arr.dup());
        IAMax iaMax = new IAMax(arr.dup());
        double imax = Nd4j.getExecutioner().execAndReturn(iMax).getFinalResult();
        double iamax = Nd4j.getExecutioner().execAndReturn(iaMax).getFinalResult();
        System.out.println("IMAX: " + imax);
        System.out.println("IAMAX: " + iamax);
        assertEquals(1, iamax, 0.0);
        assertEquals(3, imax, 0.0);
    }


    @Test
    public void testIMinIAMin() {
        INDArray arr = Nd4j.create(new double[] {-0.24, -0.26, -0.07, -0.01});
        INDArray abs = Transforms.abs(arr);
        IAMin iaMin = new IAMin(abs);
        IMin iMin = new IMin(arr.dup());
        double imin = Nd4j.getExecutioner().execAndReturn(iMin).getFinalResult();
        double iamin = Nd4j.getExecutioner().execAndReturn(iaMin).getFinalResult();
        System.out.println("IMin: " + imin);
        System.out.println("IAMin: " + iamin);
        assertEquals(3, iamin, 1e-12);
        assertEquals(1, imin, 1e-12);
    }


    @Test
    public void testBroadcast3d2d() {
        char[] orders = {'c', 'f'};

        for (char orderArr : orders) {
            for (char orderbc : orders) {
                System.out.println(orderArr + "\t" + orderbc);
                INDArray arrOrig = Nd4j.ones(3, 4, 5).dup(orderArr);

                //Broadcast on dimensions 0,1
                INDArray bc01 = Nd4j.create(new double[][] {{1, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 0}}).dup(orderbc);

                INDArray result01 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(arrOrig, bc01, result01, 0, 1));

                for (int i = 0; i < 5; i++) {
                    INDArray subset = result01.tensorAlongDimension(i, 0, 1);//result01.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i));
                    assertEquals(bc01, subset);
                }

                //Broadcast on dimensions 0,2
                INDArray bc02 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {1, 0, 0, 1, 1}, {1, 1, 1, 0, 0}})
                        .dup(orderbc);

                INDArray result02 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(arrOrig, bc02, result02, 0, 2));

                for (int i = 0; i < 4; i++) {
                    INDArray subset = result02.tensorAlongDimension(i, 0, 2); //result02.get(NDArrayIndex.all(), NDArrayIndex.point(i), NDArrayIndex.all());
                    assertEquals(bc02, subset);
                }

                //Broadcast on dimensions 1,2
                INDArray bc12 = Nd4j.create(
                        new double[][] {{1, 1, 1, 1, 1}, {0, 1, 1, 1, 1}, {1, 0, 0, 1, 1}, {1, 1, 1, 0, 0}})
                        .dup(orderbc);

                INDArray result12 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(arrOrig, bc12, result12, 1, 2));

                for (int i = 0; i < 3; i++) {
                    INDArray subset = result12.tensorAlongDimension(i, 1, 2);//result12.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());
                    assertEquals("Failed for subset " + i, bc12, subset);
                }
            }
        }
    }

    @Test
    public void testBroadcast4d2d() {
        char[] orders = {'c', 'f'};

        for (char orderArr : orders) {
            for (char orderbc : orders) {
                System.out.println(orderArr + "\t" + orderbc);
                INDArray arrOrig = Nd4j.ones(3, 4, 5, 6).dup(orderArr);

                //Broadcast on dimensions 0,1
                INDArray bc01 = Nd4j.create(new double[][] {{1, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 0}}).dup(orderbc);

                INDArray result01 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result01, bc01, result01, 0, 1));

                for (int d2 = 0; d2 < 5; d2++) {
                    for (int d3 = 0; d3 < 6; d3++) {
                        INDArray subset = result01.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(d2),
                                NDArrayIndex.point(d3));
                        assertEquals(bc01, subset);
                    }
                }

                //Broadcast on dimensions 0,2
                INDArray bc02 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {1, 0, 0, 1, 1}, {1, 1, 1, 0, 0}})
                        .dup(orderbc);

                INDArray result02 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result02, bc02, result02, 0, 2));

                for (int d1 = 0; d1 < 4; d1++) {
                    for (int d3 = 0; d3 < 6; d3++) {
                        INDArray subset = result02.get(NDArrayIndex.all(), NDArrayIndex.point(d1), NDArrayIndex.all(),
                                NDArrayIndex.point(d3));
                        assertEquals(bc02, subset);
                    }
                }

                //Broadcast on dimensions 0,3
                INDArray bc03 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1}, {1, 1, 1, 0, 0, 0}})
                        .dup(orderbc);

                INDArray result03 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result03, bc03, result03, 0, 3));

                for (int d1 = 0; d1 < 4; d1++) {
                    for (int d2 = 0; d2 < 5; d2++) {
                        INDArray subset = result03.get(NDArrayIndex.all(), NDArrayIndex.point(d1),
                                NDArrayIndex.point(d2), NDArrayIndex.all());
                        assertEquals(bc03, subset);
                    }
                }

                //Broadcast on dimensions 1,2
                INDArray bc12 = Nd4j.create(
                        new double[][] {{1, 1, 1, 1, 1}, {0, 1, 1, 1, 1}, {1, 0, 0, 1, 1}, {1, 1, 1, 0, 0}})
                        .dup(orderbc);

                INDArray result12 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result12, bc12, result12, 1, 2));

                for (int d0 = 0; d0 < 3; d0++) {
                    for (int d3 = 0; d3 < 6; d3++) {
                        INDArray subset = result12.get(NDArrayIndex.point(d0), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.point(d3));
                        assertEquals(bc12, subset);
                    }
                }

                //Broadcast on dimensions 1,3
                INDArray bc13 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1},
                        {1, 1, 1, 0, 0, 1}}).dup(orderbc);

                INDArray result13 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result13, bc13, result13, 1, 3));

                for (int d0 = 0; d0 < 3; d0++) {
                    for (int d2 = 0; d2 < 5; d2++) {
                        INDArray subset = result13.get(NDArrayIndex.point(d0), NDArrayIndex.all(),
                                NDArrayIndex.point(d2), NDArrayIndex.all());
                        assertEquals(bc13, subset);
                    }
                }

                //Broadcast on dimensions 2,3
                INDArray bc23 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1}, {1, 1, 1, 0, 0, 0},
                        {1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0}}).dup(orderbc);

                INDArray result23 = arrOrig.dup(orderArr);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(result23, bc23, result23, 2, 3));

                for (int d0 = 0; d0 < 3; d0++) {
                    for (int d1 = 0; d1 < 4; d1++) {
                        INDArray subset = result23.get(NDArrayIndex.point(d0), NDArrayIndex.point(d1),
                                NDArrayIndex.all(), NDArrayIndex.all());
                        assertEquals(bc23, subset);
                    }
                }

            }
        }
    }

    protected static boolean arrayNotEquals(float[] arrayX, float[] arrayY, float delta) {
        if (arrayX.length != arrayY.length)
            return false;

        // on 2d arrays first & last elements will match regardless of order
        for (int i = 1; i < arrayX.length - 1; i++) {
            if (Math.abs(arrayX[i] - arrayY[i]) < delta) {
                log.info("ArrX[{}]: {}; ArrY[{}]: {}", i, arrayX[i], i, arrayY[i]);
                return false;
            }
        }

        return true;
    }


    @Test
    public void testIsMax2Of3d() {
        double[][][] slices = new double[3][][];
        double[][][] isMax = new double[3][][];

        slices[0] = new double[][] {{1, 10, 2}, {3, 4, 5}};
        slices[1] = new double[][] {{-10, -9, -8}, {-7, -6, -5}};
        slices[2] = new double[][] {{4, 3, 2}, {1, 0, -1}};

        isMax[0] = new double[][] {{0, 1, 0}, {0, 0, 0}};
        isMax[1] = new double[][] {{0, 0, 0}, {0, 0, 1}};
        isMax[2] = new double[][] {{1, 0, 0}, {0, 0, 0}};

        INDArray arr = Nd4j.create(3, 2, 3);
        INDArray expected = Nd4j.create(3, 2, 3);
        for (int i = 0; i < 3; i++) {
            arr.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).assign(Nd4j.create(slices[i]));
            expected.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).assign(Nd4j.create(isMax[i]));
        }

        Nd4j.getExecutioner().exec(new IsMax(arr, 1, 2));

        assertEquals(expected, arr);
    }

    @Test
    public void testIsMax2of4d() {

        Nd4j.getRandom().setSeed(12345);
        val s = new long[] {2, 3, 4, 5};
        INDArray arr = Nd4j.rand(s);

        //Test 0,1
        INDArray exp = Nd4j.create(s);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 5; j++) {
                INDArray subset = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i),
                        NDArrayIndex.point(j));
                INDArray subsetExp = exp.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i),
                        NDArrayIndex.point(j));
                assertArrayEquals(new long[] {2, 3}, subset.shape());

                NdIndexIterator iter = new NdIndexIterator(2, 3);
                val maxIdx = new long[]{0, 0};
                double max = -Double.MAX_VALUE;
                while (iter.hasNext()) {
                    val next = iter.next();
                    double d = subset.getDouble(next);
                    if (d > max) {
                        max = d;
                        maxIdx[0] = next[0];
                        maxIdx[1] = next[1];
                    }
                }

                subsetExp.putScalar(maxIdx, 1.0);
            }
        }

        INDArray actC = Nd4j.getExecutioner().execAndReturn(new IsMax(arr.dup('c'), 0, 1));
        INDArray actF = Nd4j.getExecutioner().execAndReturn(new IsMax(arr.dup('f'), 0, 1));

        assertEquals(exp, actC);
        assertEquals(exp, actF);



        //Test 2,3
        exp = Nd4j.create(s);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                INDArray subset = arr.get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all(),
                        NDArrayIndex.all());
                INDArray subsetExp = exp.get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all(),
                        NDArrayIndex.all());
                assertArrayEquals(new long[] {4, 5}, subset.shape());

                NdIndexIterator iter = new NdIndexIterator(4, 5);
                val maxIdx = new long[]{0, 0};
                double max = -Double.MAX_VALUE;
                while (iter.hasNext()) {
                    val next = iter.next();
                    double d = subset.getDouble(next);
                    if (d > max) {
                        max = d;
                        maxIdx[0] = next[0];
                        maxIdx[1] = next[1];
                    }
                }

                subsetExp.putScalar(maxIdx, 1.0);
            }
        }

        actC = Nd4j.getExecutioner().execAndReturn(new IsMax(arr.dup('c'), 2, 3));
        actF = Nd4j.getExecutioner().execAndReturn(new IsMax(arr.dup('f'), 2, 3));

        assertEquals(exp, actC);
        assertEquals(exp, actF);
    }

    @Test
    public void testIMax2Of3d() {
        double[][][] slices = new double[3][][];

        slices[0] = new double[][] {{1, 10, 2}, {3, 4, 5}};
        slices[1] = new double[][] {{-10, -9, -8}, {-7, -6, -5}};
        slices[2] = new double[][] {{4, 3, 2}, {1, 0, -1}};

        //Based on a c-order traversal of each tensor
        double[] imax = new double[] {1, 5, 0};

        INDArray arr = Nd4j.create(3, 2, 3);
        for (int i = 0; i < 3; i++) {
            arr.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).assign(Nd4j.create(slices[i]));
        }

        INDArray out = Nd4j.getExecutioner().exec(new IMax(arr), 1, 2);

        INDArray exp = Nd4j.create(imax);

        assertEquals(exp, out);
    }


    @Test
    public void testIMax2of4d() {
        Nd4j.getRandom().setSeed(12345);
        val s = new long[] {2, 3, 4, 5};
        INDArray arr = Nd4j.rand(s);

        //Test 0,1
        INDArray exp = Nd4j.create(new long[] {4, 5});
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 5; j++) {
                INDArray subset = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i),
                        NDArrayIndex.point(j));
                assertArrayEquals(new long[] {2, 3}, subset.shape());

                NdIndexIterator iter = new NdIndexIterator('c', 2, 3);
                double max = -Double.MAX_VALUE;
                int maxIdxPos = -1;
                int count = 0;
                while (iter.hasNext()) {
                    val next = iter.next();
                    double d = subset.getDouble(next);
                    if (d > max) {
                        max = d;
                        maxIdxPos = count;
                    }
                    count++;
                }

                exp.putScalar(i, j, maxIdxPos);
            }
        }

        INDArray actC = Nd4j.getExecutioner().exec(new IMax(arr.dup('c')), 0, 1);
        INDArray actF = Nd4j.getExecutioner().exec(new IMax(arr.dup('f')), 0, 1);
        //
        assertEquals(exp, actC);
        assertEquals(exp, actF);



        //Test 2,3
        exp = Nd4j.create(new long[] {2, 3});
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                INDArray subset = arr.get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all(),
                        NDArrayIndex.all());
                assertArrayEquals(new long[] {4, 5}, subset.shape());

                NdIndexIterator iter = new NdIndexIterator('c', 4, 5);
                int maxIdxPos = -1;
                double max = -Double.MAX_VALUE;
                int count = 0;
                while (iter.hasNext()) {
                    val next = iter.next();
                    double d = subset.getDouble(next);
                    if (d > max) {
                        max = d;
                        maxIdxPos = count;
                    }
                    count++;
                }

                exp.putScalar(i, j, maxIdxPos);
            }
        }

        actC = Nd4j.getExecutioner().exec(new IMax(arr.dup('c')), 2, 3);
        actF = Nd4j.getExecutioner().exec(new IMax(arr.dup('f')), 2, 3);

        assertEquals(exp, actC);
        assertEquals(exp, actF);
    }

    @Test
    public void testTadPermuteEquals() {
        INDArray d3c = Nd4j.linspace(1, 5, 5).reshape('c', 1, 5, 1);
        INDArray d3f = d3c.dup('f');

        INDArray tadCi = d3c.tensorAlongDimension(0, 1, 2).permutei(1, 0);
        INDArray tadFi = d3f.tensorAlongDimension(0, 1, 2).permutei(1, 0);

        INDArray tadC = d3c.tensorAlongDimension(0, 1, 2).permute(1, 0);
        INDArray tadF = d3f.tensorAlongDimension(0, 1, 2).permute(1, 0);

        assertArrayEquals(tadCi.shape(), tadC.shape());
        assertArrayEquals(tadCi.stride(), tadC.stride());
        assertArrayEquals(tadCi.data().asDouble(), tadC.data().asDouble(), 1e-8);
        assertEquals(tadC, tadCi.dup());
        assertEquals(tadC, tadCi);

        assertArrayEquals(tadFi.shape(), tadF.shape());
        assertArrayEquals(tadFi.stride(), tadF.stride());
        assertArrayEquals(tadFi.data().asDouble(), tadF.data().asDouble(), 1e-8);

        assertEquals(tadF, tadFi.dup());
        assertEquals(tadF, tadFi);
    }

    @Test
    public void testRemainder1() throws Exception {
        INDArray x = Nd4j.create(10).assign(5.3);
        INDArray y = Nd4j.create(10).assign(2.0);
        INDArray exp = Nd4j.create(10).assign(-0.7);

        INDArray result = x.remainder(2.0);
        assertEquals(exp, result);

        result = x.remainder(y);
        assertEquals(exp, result);
    }

    @Test
    public void testFMod1() throws Exception {
        INDArray x = Nd4j.create(10).assign(5.3);
        INDArray y = Nd4j.create(10).assign(2.0);
        INDArray exp = Nd4j.create(10).assign(1.3);

        INDArray result = x.fmod(2.0);
        assertEquals(exp, result);

        result = x.fmod(y);
        assertEquals(exp, result);
    }

    @Test
    public void testStrangeDups1() throws Exception {
        INDArray array = Nd4j.create(10).assign(0);
        INDArray exp = Nd4j.create(10).assign(1.0f);
        INDArray copy = null;

        for (int x = 0; x < array.length(); x++) {
            array.putScalar(x, 1f);
            copy = array.dup();
        }

        assertEquals(exp, array);
        assertEquals(exp, copy);
    }

    @Test
    public void testStrangeDups2() throws Exception {
        INDArray array = Nd4j.create(10).assign(0);
        INDArray exp1 = Nd4j.create(10).assign(1.0f);
        INDArray exp2 = Nd4j.create(10).assign(1.0f).putScalar(9, 0f);
        INDArray copy = null;

        for (int x = 0; x < array.length(); x++) {
            copy = array.dup();
            array.putScalar(x, 1f);
        }

        assertEquals(exp1, array);
        assertEquals(exp2, copy);
    }

    @Test
    public void testReductionAgreement1() throws Exception {
        INDArray row = Nd4j.linspace(1, 3, 3);
        INDArray mean0 = row.mean(0);
        assertFalse(mean0 == row); //True: same object (should be a copy)

        INDArray col = Nd4j.linspace(1, 3, 3).transpose();
        INDArray mean1 = col.mean(1);
        assertFalse(mean1 == col);
    }


    @Test
    public void testSpecialConcat1() throws Exception {
        for (int i = 0; i < 10; i++) {
            List<INDArray> arrays = new ArrayList<>();
            for (int x = 0; x < 10; x++) {
                arrays.add(Nd4j.create(100).assign(x));
            }

            INDArray matrix = Nd4j.specialConcat(0, arrays.toArray(new INDArray[0]));
            assertEquals(10, matrix.rows());
            assertEquals(100, matrix.columns());

            for (int x = 0; x < 10; x++) {
                assertEquals((double) x, matrix.getRow(x).meanNumber().doubleValue(), 0.1);
                assertEquals(arrays.get(x), matrix.getRow(x));
            }
        }
    }


    @Test
    public void testSpecialConcat2() throws Exception {
        List<INDArray> arrays = new ArrayList<>();
        for (int x = 0; x < 10; x++) {
            arrays.add(Nd4j.create(new double[] {x, x, x, x, x, x}));
        }

        INDArray matrix = Nd4j.specialConcat(0, arrays.toArray(new INDArray[0]));
        assertEquals(10, matrix.rows());
        assertEquals(6, matrix.columns());

        for (int x = 0; x < 10; x++) {
            assertEquals((double) x, matrix.getRow(x).meanNumber().doubleValue(), 0.1);
            assertEquals(arrays.get(x), matrix.getRow(x));
        }
    }

    @Test
    public void testPutScalar1() {
        INDArray array = Nd4j.create(10, 3, 96, 96);

        for (int i = 0; i < 10; i++) {
            log.info("Trying i: {}", i);
            array.tensorAlongDimension(i, 1, 2, 3).putScalar(1, 2, 3, 1);
        }
    }

    @Test
    public void testAveraging1() {
        Nd4j.getAffinityManager().allowCrossDeviceAccess(false);

        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            arrays.add(Nd4j.create(100).assign((double) i));
        }

        INDArray result = Nd4j.averageAndPropagate(arrays);

        assertEquals(4.5, result.meanNumber().doubleValue(), 0.01);

        for (int i = 0; i < 10; i++) {
            assertEquals(result, arrays.get(i));
        }
    }


    @Test
    public void testAveraging2() {

        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            arrays.add(Nd4j.create(100).assign((double) i));
        }

        Nd4j.averageAndPropagate(null, arrays);

        INDArray result = arrays.get(0);

        assertEquals(4.5, result.meanNumber().doubleValue(), 0.01);

        for (int i = 0; i < 10; i++) {
            assertEquals("Failed on iteration " + i, result, arrays.get(i));
        }
    }

    @Test
    public void testAveraging3() {
        Nd4j.getAffinityManager().allowCrossDeviceAccess(false);

        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            arrays.add(Nd4j.create(100).assign((double) i));
        }

        Nd4j.averageAndPropagate(null, arrays);

        INDArray result = arrays.get(0);

        assertEquals(4.5, result.meanNumber().doubleValue(), 0.01);

        for (int i = 0; i < 10; i++) {
            assertEquals("Failed on iteration " + i, result, arrays.get(i));
        }
    }

    @Test
    public void testZ1() throws Exception {
        INDArray matrix = Nd4j.create(10, 10).assign(1.0);

        INDArray exp = Nd4j.create(10).assign(10.0);

        INDArray res = Nd4j.create(10);
        INDArray sums = matrix.sum(res, 0);

        assertTrue(res == sums);

        assertEquals(exp, res);
    }

    @Test
    public void testDupDelayed() {
        if (!(Nd4j.getExecutioner() instanceof GridExecutioner))
            return;

//        Nd4j.getExecutioner().commit();
        val executioner = (GridExecutioner) Nd4j.getExecutioner();

        log.info("Starting: -------------------------------");

        //log.info("Point A: [{}]", executioner.getQueueLength());

        INDArray in = Nd4j.zeros(10);

        List<INDArray> out = new ArrayList<>();
        List<INDArray> comp = new ArrayList<>();

        //log.info("Point B: [{}]", executioner.getQueueLength());
        //log.info("\n\n");

        for (int i = 0; i < in.length(); i++) {
//            log.info("Point C: [{}]", executioner.getQueueLength());

            in.putScalar(i, 1);

//            log.info("Point D: [{}]", executioner.getQueueLength());

            out.add(in.dup());

//            log.info("Point E: [{}]", executioner.getQueueLength());

            //Nd4j.getExecutioner().commit();
            in.putScalar(i, 0);
            //Nd4j.getExecutioner().commit();

//            log.info("Point F: [{}]\n\n", executioner.getQueueLength());
        }

        for (int i = 0; i < in.length(); i++) {
            in.putScalar(i, 1);
            comp.add(Nd4j.create(in.data().dup()));
            //Nd4j.getExecutioner().commit();
            in.putScalar(i, 0);
        }

        for (int i = 0; i < out.size(); i++) {
            assertEquals("Failed at iteration: [" + i + "]", out.get(i), comp.get(i));
        }
    }

    @Test
    public void testScalarReduction1() {
        Accumulation op = new Norm2(Nd4j.create(1).assign(1.0));
        double norm2 = Nd4j.getExecutioner().execAndReturn(op).getFinalResult().doubleValue();
        double norm1 = Nd4j.getExecutioner().execAndReturn(new Norm1(Nd4j.create(1).assign(1.0))).getFinalResult()
                .doubleValue();
        double sum = Nd4j.getExecutioner().execAndReturn(new Sum(Nd4j.create(1).assign(1.0))).getFinalResult()
                .doubleValue();

        assertEquals(1.0, norm2, 0.001);
        assertEquals(1.0, norm1, 0.001);
        assertEquals(1.0, sum, 0.001);
    }

    @Test
    public void sumResultArrayEdgeCase() {
        INDArray delta = Nd4j.create(1, 3);
        delta.assign(Nd4j.rand(delta.shape()));

        INDArray out = delta.sum(0);

        INDArray out2 = Nd4j.zeros(new long[] {1, 3}, 'c');
        INDArray res = delta.sum(out2, 0);

        assertEquals(out, out2);
        assertTrue(res == out2);
    }


    @Test
    public void tesAbsReductions1() throws Exception {
        INDArray array = Nd4j.create(new double[] {-1, -2, -3, -4});

        assertEquals(4, array.amaxNumber().intValue());
    }


    @Test
    public void tesAbsReductions2() throws Exception {
        INDArray array = Nd4j.create(new double[] {-1, -2, -3, -4});

        assertEquals(1, array.aminNumber().intValue());
    }


    @Test
    public void tesAbsReductions3() throws Exception {
        INDArray array = Nd4j.create(new double[] {-2, -2, 2, 2});

        assertEquals(2, array.ameanNumber().intValue());
    }


    @Test
    public void tesAbsReductions4() throws Exception {
        INDArray array = Nd4j.create(new double[] {-2, -2, 2, 2});

        assertEquals(4, array.scan(Conditions.absGreaterThanOrEqual(0.0)).intValue());
    }

    @Test
    public void tesAbsReductions5() throws Exception {
        INDArray array = Nd4j.create(new double[] {-2, 0.0, 2, 2});

        assertEquals(3, array.scan(Conditions.absGreaterThan(0.0)).intValue());
    }

    @Test
    public void testNewBroadcastComparison1() throws Exception {
        INDArray initial = Nd4j.create(3, 5);
        INDArray mask = Nd4j.create(new double[] {5, 4, 3, 2, 1});
        INDArray exp = Nd4j.create(new double[] {1, 1, 1, 0, 0});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i);
        }

        Nd4j.getExecutioner().commit();


        Nd4j.getExecutioner().exec(new BroadcastLessThan(initial, mask, initial, 1));



        for (int i = 0; i < initial.rows(); i++) {
            assertEquals(exp, initial.getRow(i));
        }
    }



    @Test
    public void testNewBroadcastComparison2() throws Exception {
        INDArray initial = Nd4j.create(3, 5);
        INDArray mask = Nd4j.create(new double[] {5, 4, 3, 2, 1});
        INDArray exp = Nd4j.create(new double[] {0, 0, 0, 1, 1});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i);
        }

        Nd4j.getExecutioner().commit();


        Nd4j.getExecutioner().exec(new BroadcastGreaterThan(initial, mask, initial, 1));



        for (int i = 0; i < initial.rows(); i++) {
            assertEquals(exp, initial.getRow(i));
        }
    }


    @Test
    public void testNewBroadcastComparison3() throws Exception {
        INDArray initial = Nd4j.create(3, 5);
        INDArray mask = Nd4j.create(new double[] {5, 4, 3, 2, 1});
        INDArray exp = Nd4j.create(new double[] {0, 0, 1, 1, 1});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i + 1);
        }

        Nd4j.getExecutioner().commit();


        Nd4j.getExecutioner().exec(new BroadcastGreaterThanOrEqual(initial, mask, initial, 1));


        for (int i = 0; i < initial.rows(); i++) {
            assertEquals(exp, initial.getRow(i));
        }
    }

    @Test
    public void testNewBroadcastComparison4() throws Exception {
        INDArray initial = Nd4j.create(3, 5);
        INDArray mask = Nd4j.create(new double[] {5, 4, 3, 2, 1});
        INDArray exp = Nd4j.create(new double[] {0, 0, 1, 0, 0});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i+1);
        }

        Nd4j.getExecutioner().commit();


        Nd4j.getExecutioner().exec(new BroadcastEqualTo(initial, mask, initial, 1 ));


        for (int i = 0; i < initial.rows(); i++) {
            assertEquals(exp, initial.getRow(i));
        }
    }

    @Test
    public void testTadReduce3_0() throws Exception {
        INDArray haystack = Nd4j.create(new double[] {-0.84443557262, -0.06822254508, 0.74266910552, 0.61765557527,
                -0.77555125951, -0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130,
                -1.25485503673, 0.62955373525, -0.31357592344, 1.03362500667, -0.59279078245, 1.1914824247})
                .reshape(3, 5);
        INDArray needle = Nd4j.create(new double[] {-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130,
                -1.25485503673});

        INDArray reduced = Nd4j.getExecutioner().exec(new CosineDistance(haystack, needle), 1);
        log.info("Reduced: {}", reduced);


        INDArray exp = Nd4j.create(new double[] {0.577452, 0.0, 1.80182});
        assertEquals(exp, reduced);

        for (int i = 0; i < haystack.rows(); i++) {
            double res = Nd4j.getExecutioner().execAndReturn(new CosineDistance(haystack.getRow(i).dup(), needle))
                    .getFinalResult().doubleValue();
            assertEquals("Failed at " + i, reduced.getDouble(i), res, 0.001);
        }
        //cosinedistance([-0.84443557262, -0.06822254508, 0.74266910552, 0.61765557527, -0.77555125951], [-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673)
        //cosinedistance([.62955373525, -0.31357592344, 1.03362500667, -0.59279078245, 1.1914824247], [-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673)

    }

    @Test
    public void testTadReduce3_1() throws Exception {
        INDArray initial = Nd4j.create(5, 10);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(new double[] {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9});
        INDArray reduced = Nd4j.getExecutioner().exec(new CosineSimilarity(initial, needle), 1);

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            double res = Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(initial.getRow(i).dup(), needle))
                    .getFinalResult().doubleValue();
            assertEquals("Failed at " + i, reduced.getDouble(i), res, 0.001);
        }
    }

    @Test
    public void testTadReduce3_2() throws Exception {
        INDArray initial = Nd4j.create(5, 10);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0);
        INDArray reduced = Nd4j.getExecutioner().exec(new ManhattanDistance(initial, needle), 1);

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            double res = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(initial.getRow(i).dup(), needle))
                    .getFinalResult().doubleValue();
            assertEquals("Failed at " + i, reduced.getDouble(i), res, 0.001);
        }
    }

    @Test
    public void testTadReduce3_3() throws Exception {
        INDArray initial = Nd4j.create(5, 10);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0);
        INDArray reduced = Nd4j.getExecutioner().exec(new EuclideanDistance(initial, needle), 1);

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            INDArray x = initial.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(x, needle)).getFinalResult()
                    .doubleValue();
            assertEquals("Failed at " + i, reduced.getDouble(i), res, 0.001);

            log.info("Euclidean: {} vs {} is {}", x, needle, res);
        }
    }

    @Test
    public void testTadReduce3_3_NEG() throws Exception {
        INDArray initial = Nd4j.create(5, 10);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0);
        INDArray reduced = Nd4j.getExecutioner().exec(new EuclideanDistance(initial, needle), -1);

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            INDArray x = initial.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(x, needle)).getFinalResult()
                    .doubleValue();
            assertEquals("Failed at " + i, reduced.getDouble(i), res, 0.001);

            log.info("Euclidean: {} vs {} is {}", x, needle, res);
        }
    }

    @Test
    public void testTadReduce3_3_NEG_2() throws Exception {
        INDArray initial = Nd4j.create(5, 10);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0);
        INDArray reduced = Nd4j.create(5);
        Nd4j.getExecutioner().exec(new CosineSimilarity(initial, needle, reduced, initial.lengthLong()), -1);

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            INDArray x = initial.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(x, needle)).getFinalResult()
                    .doubleValue();
            assertEquals("Failed at " + i, reduced.getDouble(i), res, 0.001);

            log.info("Cosine: {} vs {} is {}", x, needle, res);
        }
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testTadReduce3_5() throws Exception {
        INDArray initial = Nd4j.create(5, 10);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(2, 10).assign(1.0);
        INDArray reduced = Nd4j.getExecutioner().exec(new EuclideanDistance(initial, needle), 1);

    }

    @Test
    public void testTadReduce3_4() throws Exception {
        INDArray initial = Nd4j.create(5, 6, 7);
        for (int i = 0; i < 5; i++) {
            initial.tensorAlongDimension(i, 1, 2).assign(i + 1);
        }
        INDArray needle = Nd4j.create(6, 7).assign(1.0);
        INDArray reduced = Nd4j.getExecutioner().exec(new ManhattanDistance(initial, needle), 1, 2);

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < 5; i++) {
            double res = Nd4j.getExecutioner()
                    .execAndReturn(new ManhattanDistance(initial.tensorAlongDimension(i, 1, 2).dup(), needle))
                    .getFinalResult().doubleValue();
            assertEquals("Failed at " + i, reduced.getDouble(i), res, 0.001);
        }
    }


    @Test
    public void testAtan2_1() throws Exception {
        INDArray x = Nd4j.create(10).assign(-1.0);
        INDArray y = Nd4j.create(10).assign(0.0);
        INDArray exp = Nd4j.create(10).assign(Math.PI);

        INDArray z = Transforms.atan2(x, y);

        assertEquals(exp, z);
    }


    @Test
    public void testAtan2_2() throws Exception {
        INDArray x = Nd4j.create(10).assign(1.0);
        INDArray y = Nd4j.create(10).assign(0.0);
        INDArray exp = Nd4j.create(10).assign(0.0);

        INDArray z = Transforms.atan2(x, y);

        assertEquals(exp, z);
    }


    @Test
    public void testJaccardDistance1() throws Exception {
        INDArray x = Nd4j.create(new double[] {0, 1, 0, 0, 1, 0});
        INDArray y = Nd4j.create(new double[] {1, 1, 0, 1, 0, 0});

        double val = Transforms.jaccardDistance(x, y);

        assertEquals(0.75, val, 1e-5);
    }


    @Test
    public void testJaccardDistance2() throws Exception {
        INDArray x = Nd4j.create(new double[] {0, 1, 0, 0, 1, 1});
        INDArray y = Nd4j.create(new double[] {1, 1, 0, 1, 0, 0});

        double val = Transforms.jaccardDistance(x, y);

        assertEquals(0.8, val, 1e-5);
    }

    @Test
    public void testHammingDistance1() throws Exception {
        INDArray x = Nd4j.create(new double[] {0, 0, 0, 1, 0, 0});
        INDArray y = Nd4j.create(new double[] {0, 0, 0, 0, 1, 0});

        double val = Transforms.hammingDistance(x, y);

        assertEquals(2.0 / 6, val, 1e-5);
    }


    @Test
    public void testHammingDistance2() throws Exception {
        INDArray x = Nd4j.create(new double[] {0, 0, 0, 1, 0, 0});
        INDArray y = Nd4j.create(new double[] {0, 1, 0, 0, 1, 0});

        double val = Transforms.hammingDistance(x, y);

        assertEquals(3.0 / 6, val, 1e-5);
    }


    @Test
    public void testHammingDistance3() throws Exception {
        INDArray x = Nd4j.create(10, 6);
        for (int r = 0; r < x.rows(); r++) {
            x.getRow(r).putScalar(r % x.columns(), 1);
        }

        INDArray y = Nd4j.create(new double[] {0, 0, 0, 0, 1, 0});

        INDArray res = Nd4j.getExecutioner().exec(new HammingDistance(x, y), 1);
        assertEquals(10, res.length());

        for (int r = 0; r < x.rows(); r++) {
            if (r == 4) {
                assertEquals(0.0, res.getDouble(r), 1e-5);
            } else {
                assertEquals(2.0 / 6, res.getDouble(r), 1e-5);
            }
        }
    }


    @Test
    public void testAllDistances1() throws Exception {
        INDArray initialX = Nd4j.create(5, 10);
        INDArray initialY = Nd4j.create(7, 10);
        for (int i = 0; i < initialX.rows(); i++) {
            initialX.getRow(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.rows(); i++) {
            initialY.getRow(i).assign(i + 101);
        }

        INDArray result = Transforms.allEuclideanDistances(initialX, initialY, 1);

        Nd4j.getExecutioner().commit();

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.rows(); x++) {

            INDArray rowX = initialX.getRow(x).dup();

            for (int y = 0; y < initialY.rows(); y++) {

                double res = result.getDouble(x, y);
                double exp = Transforms.euclideanDistance(rowX, initialY.getRow(y).dup());

                assertEquals("Failed for [" + x + ", " + y + "]", exp, res, 0.001);
            }
        }
    }


    @Test
    public void testAllDistances2() throws Exception {
        INDArray initialX = Nd4j.create(5, 10);
        INDArray initialY = Nd4j.create(7, 10);
        for (int i = 0; i < initialX.rows(); i++) {
            initialX.getRow(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.rows(); i++) {
            initialY.getRow(i).assign(i + 101);
        }

        INDArray result = Transforms.allManhattanDistances(initialX, initialY, 1);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.rows(); x++) {

            INDArray rowX = initialX.getRow(x).dup();

            for (int y = 0; y < initialY.rows(); y++) {

                double res = result.getDouble(x, y);
                double exp = Transforms.manhattanDistance(rowX, initialY.getRow(y).dup());

                assertEquals("Failed for [" + x + ", " + y + "]", exp, res, 0.001);
            }
        }
    }

    @Test
    public void testAllDistances2_Large() throws Exception {
        INDArray initialX = Nd4j.create(5, 2000);
        INDArray initialY = Nd4j.create(7, 2000);
        for (int i = 0; i < initialX.rows(); i++) {
            initialX.getRow(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.rows(); i++) {
            initialY.getRow(i).assign(i + 101);
        }

        INDArray result = Transforms.allManhattanDistances(initialX, initialY, 1);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.rows(); x++) {

            INDArray rowX = initialX.getRow(x).dup();

            for (int y = 0; y < initialY.rows(); y++) {

                double res = result.getDouble(x, y);
                double exp = Transforms.manhattanDistance(rowX, initialY.getRow(y).dup());

                assertEquals("Failed for [" + x + ", " + y + "]", exp, res, 0.001);
            }
        }
    }


    @Test
    public void testAllDistances3_Large() throws Exception {
        INDArray initialX = Nd4j.create(5, 2000);
        INDArray initialY = Nd4j.create(7, 2000);
        for (int i = 0; i < initialX.rows(); i++) {
            initialX.getRow(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.rows(); i++) {
            initialY.getRow(i).assign(i + 101);
        }

        INDArray result = Transforms.allEuclideanDistances(initialX, initialY, 1);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.rows(); x++) {

            INDArray rowX = initialX.getRow(x).dup();

            for (int y = 0; y < initialY.rows(); y++) {

                double res = result.getDouble(x, y);
                double exp = Transforms.euclideanDistance(rowX, initialY.getRow(y).dup());

                assertEquals("Failed for [" + x + ", " + y + "]", exp, res, 0.001);
            }
        }
    }


    @Test
    public void testAllDistances3_Large_Columns() throws Exception {
        INDArray initialX = Nd4j.create(2000, 5);
        INDArray initialY = Nd4j.create(2000, 7);
        for (int i = 0; i < initialX.columns(); i++) {
            initialX.getColumn(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.columns(); i++) {
            initialY.getColumn(i).assign(i + 101);
        }

        INDArray result = Transforms.allEuclideanDistances(initialX, initialY, 0);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.columns(); x++) {

            INDArray colX = initialX.getColumn(x).dup();

            for (int y = 0; y < initialY.columns(); y++) {

                double res = result.getDouble(x, y);
                double exp = Transforms.euclideanDistance(colX, initialY.getColumn(y).dup());

                assertEquals("Failed for [" + x + ", " + y + "]", exp, res, 0.001);
            }
        }
    }


    @Test
    public void testAllDistances4_Large_Columns() throws Exception {
        INDArray initialX = Nd4j.create(2000, 5);
        INDArray initialY = Nd4j.create(2000, 7);
        for (int i = 0; i < initialX.columns(); i++) {
            initialX.getColumn(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.columns(); i++) {
            initialY.getColumn(i).assign(i + 101);
        }

        INDArray result = Transforms.allManhattanDistances(initialX, initialY, 0);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.columns(); x++) {

            INDArray colX = initialX.getColumn(x).dup();

            for (int y = 0; y < initialY.columns(); y++) {

                double res = result.getDouble(x, y);
                double exp = Transforms.manhattanDistance(colX, initialY.getColumn(y).dup());

                assertEquals("Failed for [" + x + ", " + y + "]", exp, res, 0.001);
            }
        }
    }

    @Test
    public void testAllDistances5_Large_Columns() throws Exception {
        INDArray initialX = Nd4j.create(2000, 5);
        INDArray initialY = Nd4j.create(2000, 7);
        for (int i = 0; i < initialX.columns(); i++) {
            initialX.getColumn(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.columns(); i++) {
            initialY.getColumn(i).assign(i + 101);
        }

        INDArray result = Transforms.allCosineDistances(initialX, initialY, 0);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.columns(); x++) {

            INDArray colX = initialX.getColumn(x).dup();

            for (int y = 0; y < initialY.columns(); y++) {

                double res = result.getDouble(x, y);
                double exp = Transforms.cosineDistance(colX, initialY.getColumn(y).dup());

                assertEquals("Failed for [" + x + ", " + y + "]", exp, res, 0.001);
            }
        }
    }

    @Test
    public void testAllDistances3_Small_Columns() throws Exception {
        INDArray initialX = Nd4j.create(200, 5);
        INDArray initialY = Nd4j.create(200, 7);
        for (int i = 0; i < initialX.columns(); i++) {
            initialX.getColumn(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.columns(); i++) {
            initialY.getColumn(i).assign(i + 101);
        }

        INDArray result = Transforms.allManhattanDistances(initialX, initialY, 0);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.columns(); x++) {
            INDArray colX = initialX.getColumn(x).dup();

            for (int y = 0; y < initialY.columns(); y++) {

                double res = result.getDouble(x, y);
                double exp = Transforms.manhattanDistance(colX, initialY.getColumn(y).dup());

                assertEquals("Failed for [" + x + ", " + y + "]", exp, res, 0.001);
            }
        }
    }



    @Test
    public void testAllDistances3() throws Exception {
        Nd4j.getRandom().setSeed(123);

        INDArray initialX = Nd4j.rand(5, 10);
        INDArray initialY = initialX.mul(-1);

        INDArray result = Transforms.allCosineSimilarities(initialX, initialY, 1);

        assertEquals(5 * 5, result.length());

        for (int x = 0; x < initialX.rows(); x++) {

            INDArray rowX = initialX.getRow(x).dup();

            for (int y = 0; y < initialY.rows(); y++) {

                double res = result.getDouble(x, y);
                double exp = Transforms.cosineSim(rowX, initialY.getRow(y).dup());

                assertEquals("Failed for [" + x + ", " + y + "]", exp, res, 0.001);
            }
        }
    }


    @Test
    public void testStridedTransforms1() throws Exception {
        //output: Rank: 2,Offset: 0
        //Order: c Shape: [5,2],  stride: [2,1]
        //output: [0.5086864, 0.49131358, 0.50720876, 0.4927912, 0.46074104, 0.53925896, 0.49314, 0.50686, 0.5217741, 0.4782259]

        double[] d = {0.5086864, 0.49131358, 0.50720876, 0.4927912, 0.46074104, 0.53925896, 0.49314, 0.50686, 0.5217741,
                0.4782259};

        INDArray in = Nd4j.create(d, new long[] {5, 2}, 'c');

        INDArray col0 = in.getColumn(0);
        INDArray col1 = in.getColumn(1);

        float[] exp0 = new float[d.length / 2];
        float[] exp1 = new float[d.length / 2];
        for (int i = 0; i < col0.length(); i++) {
            exp0[i] = (float) Math.log(col0.getDouble(i));
            exp1[i] = (float) Math.log(col1.getDouble(i));
        }

        INDArray out0 = Transforms.log(col0, true);
        INDArray out1 = Transforms.log(col1, true);

        assertArrayEquals(exp0, out0.data().asFloat(), 1e-4f);
        assertArrayEquals(exp1, out1.data().asFloat(), 1e-4f);
    }

    @Test
    public void testEntropy1() throws Exception {
        INDArray x = Nd4j.rand(1, 100);

        double exp = MathUtils.entropy(x.data().asDouble());
        double res = x.entropyNumber().doubleValue();

        assertEquals(exp, res, 1e-5);
    }

    @Test
    public void testEntropy2() throws Exception {
        INDArray x = Nd4j.rand(10, 100);

        INDArray res = x.entropy(1);

        assertEquals(10, res.lengthLong());

        for (int t = 0; t < x.rows(); t++) {
            double exp = MathUtils.entropy(x.getRow(t).dup().data().asDouble());

            assertEquals(exp, res.getDouble(t), 1e-5);
        }
    }


    @Test
    public void testEntropy3() throws Exception {
        INDArray x = Nd4j.rand(1, 100);

        double exp = getShannonEntropy(x.data().asDouble());
        double res = x.shannonEntropyNumber().doubleValue();

        assertEquals(exp, res, 1e-5);
    }

    @Test
    public void testEntropy4() throws Exception {
        INDArray x = Nd4j.rand(1, 100);

        double exp = getLogEntropy(x.data().asDouble());
        double res = x.logEntropyNumber().doubleValue();

        assertEquals(exp, res, 1e-5);
    }


    protected double getShannonEntropy(double[] array) {
        double ret = 0;
        for (double x : array) {
            ret += FastMath.pow(x, 2) * FastMath.log(FastMath.pow(x, 2));
        }

        return -ret;
    }


    protected double getLogEntropy(double[] array) {
        return Math.log(MathUtils.entropy(array));
    }


    @Test
    public void testReverse1() throws Exception {
        INDArray array = Nd4j.create(new double[] {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        INDArray rev = Nd4j.reverse(array);

        assertEquals(exp, rev);
    }

    @Test
    public void testReverse2() throws Exception {
        INDArray array = Nd4j.create(new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

        INDArray rev = Nd4j.reverse(array);

        assertEquals(exp, rev);
    }

    @Test
    public void testReverse3() throws Exception {
        INDArray array = Nd4j.create(new double[] {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        INDArray rev = Nd4j.getExecutioner().exec(new OldReverse(array, Nd4j.createUninitialized(array.length()))).z();

        assertEquals(exp, rev);
    }

    @Test
    public void testReverse4() throws Exception {
        INDArray array = Nd4j.create(new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

        INDArray rev = Nd4j.getExecutioner().exec(new OldReverse(array, Nd4j.createUninitialized(array.length()))).z();

        assertEquals(exp, rev);
    }

    @Test
    public void testReverse5() throws Exception {
        INDArray array = Nd4j.create(new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

        INDArray rev = Transforms.reverse(array, true);

        assertEquals(exp, rev);
        assertFalse(rev == array);
    }


    @Test
    public void testReverse6() throws Exception {
        INDArray array = Nd4j.create(new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

        INDArray rev = Transforms.reverse(array, false);

        assertEquals(exp, rev);
        assertTrue(rev == array);
    }


    @Test
    public void testNativeSortView1() {
        INDArray matrix = Nd4j.create(10, 10);
        INDArray exp = Nd4j.linspace(0, 9, 10);
        int cnt = 0;
        for (long i = matrix.rows() - 1; i >= 0; i--) {
            // FIXME: int cast
            matrix.getRow((int) i).assign(cnt);
            cnt++;
        }

        Nd4j.sort(matrix.getColumn(0), true);


        log.info("Matrix: {}", matrix);

        assertEquals(exp, matrix.getColumn(0));
    }

    @Test
    public void testNativeSort1() throws Exception {
        INDArray array = Nd4j.create(new double[] {9, 2, 1, 7, 6, 5, 4, 3, 8, 0});
        INDArray exp1 = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        INDArray exp2 = Nd4j.create(new double[] {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

        INDArray res = Nd4j.sort(array, true);

        assertEquals(exp1, res);

        res = Nd4j.sort(res, false);

        assertEquals(exp2, res);
    }

    @Test
    public void testNativeSort2() throws Exception {
        INDArray array = Nd4j.rand(1, 10000);

        INDArray res = Nd4j.sort(array, true);
        INDArray exp = res.dup();

        res = Nd4j.sort(res, false);
        res = Nd4j.sort(res, true);

        assertEquals(exp, res);
    }

    @Test
    public void testNativeSort3() throws Exception {
        INDArray array = Nd4j.linspace(1, 1048576, 1048576);
        INDArray exp = array.dup();
        Nd4j.shuffle(array, 0);

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, true);
        long time2 = System.currentTimeMillis();
        log.info("Time spent: {} ms", time2 - time1);

        assertEquals(exp, res);
    }

    @Test
    public void testNativeSort3_1() throws Exception {
        INDArray array = Nd4j.linspace(1, 2017152, 2017152);
        INDArray exp = array.dup();
        Transforms.reverse(array, false);


        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, true);
        long time2 = System.currentTimeMillis();
        log.info("Time spent: {} ms", time2 - time1);

        assertEquals(exp, res);
    }

    @Test
    public void testNativeSortAlongDimension1() throws Exception {
        INDArray array = Nd4j.create(1000, 1000);
        INDArray exp1 = Nd4j.linspace(1, 1000, 1000);
        INDArray dps = exp1.dup();
        Nd4j.shuffle(dps, 0);

        assertNotEquals(exp1, dps);


        for (int r = 0; r < array.rows(); r++) {
            array.getRow(r).assign(dps);
        }

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, 1, true);
        long time2 = System.currentTimeMillis();

        log.info("Time spent: {} ms", time2 - time1);

        for (int r = 0; r < array.rows(); r++) {
            assertEquals("Failed at " + r, exp1, res.getRow(r).dup());
        }
    }

    @Test
    public void testNativeSortAlongDimension3() throws Exception {
        INDArray array = Nd4j.create(2000, 2000);
        INDArray exp1 = Nd4j.linspace(1, 2000, 2000);
        INDArray dps = exp1.dup();

        Nd4j.getExecutioner().commit();
        Nd4j.shuffle(dps, 0);

        assertNotEquals(exp1, dps);


        for (int r = 0; r < array.rows(); r++) {
            array.getRow(r).assign(dps);
        }

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, 1, true);
        long time2 = System.currentTimeMillis();

        log.info("Time spent: {} ms", time2 - time1);

        for (int r = 0; r < array.rows(); r++) {
            assertEquals("Failed at " + r, exp1, res.getRow(r));
            //assertArrayEquals("Failed at " + r, exp1.data().asDouble(), res.getRow(r).dup().data().asDouble(), 1e-5);
        }
    }

    @Test
    public void testNativeSortAlongDimension2() throws Exception {
        INDArray array = Nd4j.create(100, 10);
        INDArray exp1 = Nd4j.create(new double[] {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

        for (int r = 0; r < array.rows(); r++) {
            array.getRow(r).assign(Nd4j.create(new double[] {3, 8, 2, 7, 5, 6, 4, 9, 1, 0}));
        }

        INDArray res = Nd4j.sort(array, 1, false);

        for (int r = 0; r < array.rows(); r++) {
            assertEquals("Failed at " + r, exp1, res.getRow(r).dup());
        }
    }


    @Test
    public void testPercentile1() throws Exception {
        INDArray array = Nd4j.linspace(1, 10, 10);
        Percentile percentile = new Percentile(50);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(50));
    }

    @Test
    public void testPercentile2() throws Exception {
        INDArray array = Nd4j.linspace(1, 9, 9);
        Percentile percentile = new Percentile(50);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(50));
    }


    @Test
    public void testPercentile3() throws Exception {
        INDArray array = Nd4j.linspace(1, 9, 9);
        Percentile percentile = new Percentile(75);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(75));
    }

    @Test
    public void testPercentile4() throws Exception {
        INDArray array = Nd4j.linspace(1, 10, 10);
        Percentile percentile = new Percentile(75);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(75));
    }

    @Test
    public void testTadPercentile1() throws Exception {
        INDArray array = Nd4j.linspace(1, 10, 10);
        Transforms.reverse(array, false);
        Percentile percentile = new Percentile(75);
        double exp = percentile.evaluate(array.data().asDouble());

        INDArray matrix = Nd4j.create(10, 10);
        for (int i = 0; i < matrix.rows(); i++)
            matrix.getRow(i).assign(array);

        INDArray res = matrix.percentile(75, 1);

        for (int i = 0; i < matrix.rows(); i++)
            assertEquals(exp, res.getDouble(i), 1e-5);
    }

    @Test
    public void testPutiRowVector() throws Exception {
        INDArray matrix = Nd4j.createUninitialized(10, 10);
        INDArray exp = Nd4j.create(10, 10).assign(1.0);
        INDArray row = Nd4j.create(10).assign(1.0);

        matrix.putiRowVector(row);

        assertEquals(exp, matrix);
    }

    @Test
    public void testPutiColumnsVector() throws Exception {
        INDArray matrix = Nd4j.createUninitialized(5, 10);
        INDArray exp = Nd4j.create(5, 10).assign(1.0);
        INDArray row = Nd4j.create(5, 1).assign(1.0);

        matrix.putiColumnVector(row);

        assertEquals(exp, matrix);
    }

    @Test
    public void testRsub1() throws Exception {
        INDArray arr = Nd4j.ones(5).assign(2.0);
        INDArray exp_0 = Nd4j.ones(5).assign(2.0);
        INDArray exp_1 = Nd4j.create(5).assign(-1);

        Nd4j.getExecutioner().commit();

        INDArray res = arr.rsub(1.0);

        assertEquals(exp_0, arr);
        assertEquals(exp_1, res);
    }

    @Test
    public void testBroadcastMin() throws Exception {
        INDArray matrix = Nd4j.create(5, 5);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{2, 3, 3, 4, 5}));
        }

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        Nd4j.getExecutioner().exec(new BroadcastMin(matrix, row, matrix, 1));

        for (int r = 0; r < matrix.rows(); r++) {
            assertEquals(Nd4j.create(new double[] {1, 2, 3, 4, 5}), matrix.getRow(r));
        }
    }

    @Test
    public void testBroadcastMax() throws Exception {
        INDArray matrix = Nd4j.create(5, 5);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{1, 2, 3, 2, 1}));
        }

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        Nd4j.getExecutioner().exec(new BroadcastMax(matrix, row, matrix, 1));

        for (int r = 0; r < matrix.rows(); r++) {
            assertEquals(Nd4j.create(new double[] {1, 2, 3, 4, 5}), matrix.getRow(r));
        }
    }


    @Test
    public void testBroadcastAMax() throws Exception {
        INDArray matrix = Nd4j.create(5, 5);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{1, 2, 3, 2, 1}));
        }

        INDArray row = Nd4j.create(new double[]{1, 2, 3, -4, -5});

        Nd4j.getExecutioner().exec(new BroadcastAMax(matrix, row, matrix, 1));

        for (int r = 0; r < matrix.rows(); r++) {
            assertEquals(Nd4j.create(new double[] {1, 2, 3, -4, -5}), matrix.getRow(r));
        }
    }


    @Test
    public void testBroadcastAMin() throws Exception {
        INDArray matrix = Nd4j.create(5, 5);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{2, 3, 3, 4, 1}));
        }

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4, -5});

        Nd4j.getExecutioner().exec(new BroadcastAMin(matrix, row, matrix, 1));

        for (int r = 0; r < matrix.rows(); r++) {
            assertEquals(Nd4j.create(new double[] {1, 2, 3, 4, 1}), matrix.getRow(r));
        }
    }

    @Test
    public void testLogExpSum1() throws Exception {
        INDArray matrix = Nd4j.create(3, 3);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{1, 2, 3}));
        }

        INDArray res = Nd4j.getExecutioner().exec(new LogSumExp(matrix), 1);

        for (int e = 0; e < res.length(); e++) {
            assertEquals(3.407605, res.getDouble(e), 1e-5);
        }
    }

    @Test
    public void testLogExpSum2() throws Exception {
        INDArray row = Nd4j.create(new double[]{1, 2, 3});

        double res = Nd4j.getExecutioner().exec(new LogSumExp(row)).z().getDouble(0);

        assertEquals(3.407605, res, 1e-5);
    }

    @Test
    public void testPow1() throws Exception {
        val argX = Nd4j.create(3).assign(2.0);
        val argY = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        val exp = Nd4j.create(new double[] {2.0, 4.0, 8.0});
        val res = Transforms.pow(argX, argY);

        assertEquals(exp, res);
    }


    @Test
    public void testRDiv1() throws Exception {
        val argX = Nd4j.create(3).assign(2.0);
        val argY = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        val exp = Nd4j.create(new double[] {0.5, 1.0, 1.5});
        val res = argX.rdiv(argY);

        assertEquals(exp, res);
    }

    @Test
    public void testEqualOrder1() throws Exception {
        val array = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        val arrayC = array.dup('c');
        val arrayF = array.dup('f');

        assertEquals(array, arrayC);
        assertEquals(array, arrayF);
        assertEquals(arrayC, arrayF);
    }


    @Test
    public void testMatchTransform() throws Exception {
        val array = Nd4j.create(new double[] {1, 1, 1, 0, 1, 1},'c');
        val exp = Nd4j.create(new double[] {0, 0, 0, 1, 0, 0},'c');
        Op op = new MatchConditionTransform(array, array, 1e-5, Conditions.epsEquals(0.0));

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, array);
    }

    @Test
    public void test4DSumView() throws Exception {
        INDArray labels = Nd4j.linspace(1, 160, 160).reshape(new long[]{2, 5, 4, 4});
        //INDArray labels = Nd4j.linspace(1, 192, 192).reshape(new long[]{2, 6, 4, 4});

        val size1 = labels.size(1);
        INDArray classLabels = labels.get(NDArrayIndex.all(), NDArrayIndex.interval(4, size1), NDArrayIndex.all(), NDArrayIndex.all());

        /*
        Should be 0s and 1s only in the "classLabels" subset - specifically a 1-hot vector, or all 0s
        double minNumber = classLabels.minNumber().doubleValue();
        double maxNumber = classLabels.maxNumber().doubleValue();
        System.out.println("Min/max: " + minNumber + "\t" + maxNumber);
        System.out.println(sum1);
        */


        assertEquals(classLabels, classLabels.dup());

        //Expect 0 or 1 for each entry (sum of all 0s, or 1-hot vector = 0 or 1)
        INDArray sum1 = classLabels.max(1);
        INDArray sum1_dup = classLabels.dup().max(1);

        assertEquals(sum1_dup, sum1 );
    }

    @Test
    public void testMatMul1() {
        val x = 2;
        val A1 = 3;
        val A2 = 4;
        val B1 = 4;
        val B2 = 3;

        val a = Nd4j.linspace(1, x * A1 * A2, x * A1 * A2).reshape(x, A1, A2);
        val b = Nd4j.linspace(1, x * B1 * B2, x * B1 * B2).reshape(x, B1, B2);

        //

        //log.info("C shape: {}", Arrays.toString(c.shapeInfoDataBuffer().asInt()));
    }

    @Test
    public void testReduction_Z1() throws Exception {
        val arrayX = Nd4j.create(10, 10, 10);

        val res = arrayX.max(1, 2);

        Nd4j.getExecutioner().commit();
    }

    @Test
    public void testReduction_Z2() throws Exception {
        val arrayX = Nd4j.create(10, 10);

        val res = arrayX.max(0);

        Nd4j.getExecutioner().commit();
    }

    @Test
    public void testReduction_Z3() throws Exception {
        val arrayX = Nd4j.create(200, 300);

        val res = arrayX.maxNumber().doubleValue();

        Nd4j.getExecutioner().commit();
    }

    @Test
    public void testSoftmaxZ1() throws Exception {
        val original = Nd4j.linspace(1, 100, 100).reshape(10, 10);
        val reference = original.dup(original.ordering());
        val expected = original.dup(original.ordering());

        Nd4j.getExecutioner().commit();

        Nd4j.getExecutioner().execAndReturn(new OldSoftMax(expected));

        val result = Nd4j.getExecutioner().execAndReturn(new OldSoftMax(original, original.dup(original.ordering())));

        assertEquals(reference, original);
        assertEquals(expected, result);
    }

    @Test
    public void testRDiv() throws Exception {
        val x = Nd4j.create(new double[]{2,2,2});
        val y = Nd4j.create(new double[]{4,6,8});
        val result = Nd4j.createUninitialized(1,3);

        val op = DynamicCustomOp.builder("RDiv")
                .addInputs(x,y)
                .addOutputs(result)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(Nd4j.create(new double[]{2, 3, 4}), result);
    }


    @Test
    public void testIm2Col() {
        int kY = 5;
        int kX = 5;
        int sY = 1;
        int sX = 1;
        int pY = 0;
        int pX = 0;
        int dY = 1;
        int dX = 1;
        int inY = 28;
        int inX = 28;

        boolean isSameMode = true;

        val input = Nd4j.linspace(1, 2 * inY * inX, 2 * inY * inX).reshape(2, 1, inY, inX);
        val output = Nd4j.create(2, 1, 5, 5, 28, 28);

        val im2colOp = Im2col.builder()
                .inputArrays(new INDArray[]{input})
                .outputs(new INDArray[]{output})
                .conv2DConfig(Conv2DConfig.builder()
                        .kH(kY)
                        .kW(kX)
                        .kH(kY)
                        .kW(kX)
                        .sH(sY)
                        .sW(sX)
                        .pH(pY)
                        .pW(pX)
                        .dH(dY)
                        .dW(dX)
                        .isSameMode(isSameMode)
                        .build())

                .build();

        Nd4j.getExecutioner().exec(im2colOp);

        log.info("result: {}", output);
    }


    @Test
    public void testGemmStrides() {
        // 4x5 matrix from arange(20)
        final INDArray X = Nd4j.arange(20).reshape(4,5);
        for (int i=0; i<5; i++){
            // Get i-th column vector
            final INDArray xi = X.get(NDArrayIndex.all(), NDArrayIndex.point(i));
            // Build outer product
            val trans = xi.transpose();
            final INDArray outerProduct = xi.mmul(trans);
            // Build outer product from duplicated column vectors
            final INDArray outerProductDuped = xi.dup().mmul(xi.transpose().dup());
            // Matrices should equal
            //final boolean eq = outerProduct.equalsWithEps(outerProductDuped, 1e-5);
            //assertTrue(eq);
            assertEquals(outerProductDuped, outerProduct);
        }
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testReshapeFailure() {
        val a = Nd4j.linspace(1, 4, 4).reshape(2,2);
        val b = Nd4j.linspace(1, 4, 4).reshape(2,2);
        val score = a.mmul(b);
        val reshaped1 = score.reshape(2,100);
        val reshaped2 = score.reshape(2,1);
    }


    @Test
    public void testScalar_1() {
        val scalar = Nd4j.create(new float[]{2.0f}, new long[]{});

        assertTrue(scalar.isScalar());
        assertEquals(1, scalar.length());
        assertFalse(scalar.isMatrix());
        assertFalse(scalar.isVector());
        assertFalse(scalar.isRowVector());
        assertFalse(scalar.isColumnVector());

        assertEquals(2.0f, scalar.getFloat(0), 1e-5);
    }

    @Test
    public void testScalar_2() {
        val scalar = Nd4j.trueScalar(2.0f);
        val scalar2 = Nd4j.trueScalar(2.0f);
        val scalar3 = Nd4j.trueScalar(3.0f);

        assertTrue(scalar.isScalar());
        assertEquals(1, scalar.length());
        assertFalse(scalar.isMatrix());
        assertFalse(scalar.isVector());
        assertFalse(scalar.isRowVector());
        assertFalse(scalar.isColumnVector());

        assertEquals(2.0f, scalar.getFloat(0), 1e-5);

        assertEquals(scalar, scalar2);
        assertNotEquals(scalar, scalar3);
    }

    @Test
    public void testVector_1() {
        val vector = Nd4j.trueVector(new float[] {1, 2, 3, 4, 5});
        val vector2 = Nd4j.trueVector(new float[] {1, 2, 3, 4, 5});
        val vector3 = Nd4j.trueVector(new float[] {1, 2, 3, 4, 6});

        assertFalse(vector.isScalar());
        assertEquals(5, vector.length());
        assertFalse(vector.isMatrix());
        assertTrue(vector.isVector());
        assertTrue(vector.isRowVector());
        assertFalse(vector.isColumnVector());

        assertEquals(vector, vector2);
        assertNotEquals(vector, vector3);
    }

    @Test
    public void testVectorScalar_2() {
        val vector = Nd4j.trueVector(new float[]{1, 2, 3, 4, 5});
        val scalar = Nd4j.trueScalar(2.0f);
        val exp = Nd4j.trueVector(new float[]{3, 4, 5, 6, 7});

        vector.addi(scalar);

        assertEquals(exp, vector);
    }

    @Test
    public void testReshapeScalar() {
        val scalar = Nd4j.trueScalar(2.0f);
        val newShape = scalar.reshape(1, 1, 1, 1);

        assertEquals(4, newShape.rank());
        assertArrayEquals(new long[]{1, 1, 1, 1}, newShape.shape());
    }


    @Test
    public void testReshapeVector() {
        val vector = Nd4j.trueVector(new float[]{1, 2, 3, 4, 5, 6});
        val newShape = vector.reshape(3, 2);

        assertEquals(2, newShape.rank());
        assertArrayEquals(new long[]{3, 2}, newShape.shape());
    }

    @Test
    public void testTranspose1() {
        val vector = Nd4j.trueVector(new float[]{1, 2, 3, 4, 5, 6});

        assertArrayEquals(new long[]{6}, vector.shape());
        assertArrayEquals(new long[]{1}, vector.stride());

        val transposed = vector.transpose();

        assertArrayEquals(vector.shape(), transposed.shape());
    }

    @Test
    public void testTranspose2() {
        val scalar = Nd4j.trueScalar(2.f);

        assertArrayEquals(new long[]{}, scalar.shape());
        assertArrayEquals(new long[]{}, scalar.stride());

        val transposed = scalar.transpose();

        assertArrayEquals(scalar.shape(), transposed.shape());
    }

    @Test
    public void testMatmul_128by256() throws Exception {
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


    @Test
    public void testScalarSqueeze() {
        val scalar = Nd4j.create(new float[]{2.0f}, new long[]{1, 1});
        val output = Nd4j.trueScalar(0.0f);
        val exp = Nd4j.trueScalar(2.0f);
        val op = DynamicCustomOp.builder("squeeze")
                .addInputs(scalar)
                .addOutputs(output)
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(new long[]{}, shape);

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }

    @Test
    public void testScalarVectorSqueeze() {
        val scalar = Nd4j.create(new float[]{2.0f}, new long[]{1});

        assertArrayEquals(new long[]{1}, scalar.shape());

        val output = Nd4j.trueScalar(0.0f);
        val exp = Nd4j.trueScalar(2.0f);
        val op = DynamicCustomOp.builder("squeeze")
                .addInputs(scalar)
                .addOutputs(output)
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(new long[]{}, shape);

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }

    @Test
    public void testVectorSqueeze() {
        val vector = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6}, new long[]{1, 6});
        val output = Nd4j.trueVector(new float[] {0, 0, 0, 0, 0, 0});
        val exp = Nd4j.trueVector(new float[]{1, 2, 3, 4, 5, 6});

        val op = DynamicCustomOp.builder("squeeze")
                .addInputs(vector)
                .addOutputs(output)
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(new long[]{6}, shape);

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }

    @Test
    public void testVectorGemv() {
        val vectorL = Nd4j.create(new float[]{1, 2, 3}, new long[]{3, 1});
        val vectorN = Nd4j.create(new float[]{1, 2, 3}, new long[]{3});
        val matrix = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, new long[] {3, 3});

        log.info("vectorN: {}", vectorN);
        log.info("vectorL: {}", vectorL);

        val outN = matrix.mmul(vectorN);
        val outL = matrix.mmul(vectorL);

        assertEquals(outL, outN);

        assertEquals(1, outN.rank());
    }


    @Test
    public void testMatrixReshape() {
        val matrix = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, new long[] {3, 3});
        val exp = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, new long[] {9});

        val reshaped = matrix.reshape(-1);

        assertArrayEquals(exp.shape(), reshaped.shape());
        assertEquals(exp, reshaped);
    }


    @Test
    public void testVectorScalarConcat() {
        val vector = Nd4j.trueVector(new float[] {1, 2});
        val scalar = Nd4j.trueScalar(3.0f);

        val output = Nd4j.trueVector(new float[]{0, 0, 0});
        val exp = Nd4j.trueVector(new float[]{1, 2, 3});

        val op = DynamicCustomOp.builder("concat")
                .addInputs(vector, scalar)
                .addOutputs(output)
                .addIntegerArguments(0) // axis
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(exp.shape(), shape);

        Nd4j.getExecutioner().exec(op);

        assertArrayEquals(exp.shape(), output.shape());
        assertEquals(exp, output);
    }


    @Test
    public void testValueArrayOf_1() {
        val vector = Nd4j.valueArrayOf(new long[] {5}, 2f);
        val exp = Nd4j.trueVector(new float[]{2, 2, 2, 2, 2});

        assertArrayEquals(exp.shape(), vector.shape());
        assertEquals(exp, vector);
    }


    @Test
    public void testValueArrayOf_2() {
        val scalar = Nd4j.valueArrayOf(new long[] {}, 2f);
        val exp = Nd4j.trueScalar(2f);

        assertArrayEquals(exp.shape(), scalar.shape());
        assertEquals(exp, scalar);
    }


    @Test
    public void testArrayCreation() {
        val vector = Nd4j.create(new float[]{1, 2, 3}, new long[] {3}, 'c');
        val exp = Nd4j.trueVector(new float[]{1, 2, 3});

        assertArrayEquals(exp.shape(), vector.shape());
        assertEquals(exp, vector);
    }

    @Test
    public void testACosh(){
        //http://www.wolframalpha.com/input/?i=acosh(x)

        INDArray in = Nd4j.linspace(1, 3, 20);
        INDArray out = Nd4j.getExecutioner().execAndReturn(new ACosh(in.dup()));

        INDArray exp = Nd4j.create(in.shape());
        for( int i=0; i<in.length(); i++ ){
            double x = in.getDouble(i);
            double y = Math.log(x + Math.sqrt(x-1) * Math.sqrt(x+1));
            exp.putScalar(i, y);
        }

        assertEquals(exp, out);
    }

    @Test
    public void testCosh(){
        //http://www.wolframalpha.com/input/?i=cosh(x)

        INDArray in = Nd4j.linspace(-2, 2, 20);
        INDArray out = Transforms.cosh(in, true);

        INDArray exp = Nd4j.create(in.shape());
        for( int i=0; i<in.length(); i++ ){
            double x = in.getDouble(i);
            double y = 0.5 * (Math.exp(-x) + Math.exp(x));
            exp.putScalar(i, y);
        }

        assertEquals(exp, out);
    }

    @Test
    public void testAtanh(){
        //http://www.wolframalpha.com/input/?i=atanh(x)

        INDArray in = Nd4j.linspace(-0.9, 0.9, 10);
        INDArray out = Transforms.atanh(in, true);

        INDArray exp = Nd4j.create(in.shape());
        for( int i=0; i<10; i++ ){
            double x = in.getDouble(i);
            //Using "alternative form" from: http://www.wolframalpha.com/input/?i=atanh(x)
            double y = 0.5 * Math.log(x+1.0) - 0.5 * Math.log(1.0-x);
            exp.putScalar(i, y);
        }

        assertEquals(exp, out);
    }

    @Test
    public void testLastIndex(){

        INDArray in = Nd4j.create(new double[][]{
                {1,1,1,0},
                {1,1,0,0}});

        INDArray exp0 = Nd4j.create(new double[]{1,1,0,-1});
        INDArray exp1 = Nd4j.create(new double[]{2,1}).transpose();

        INDArray out0 = BooleanIndexing.lastIndex(in, Conditions.equals(1), 0);
        INDArray out1 = BooleanIndexing.lastIndex(in, Conditions.equals(1), 1);

        assertEquals(exp0, out0);
        assertEquals(exp1, out1);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testBadReduce3Call() {
        val x = Nd4j.create(400,20);
        val y = Nd4j.ones(1, 20);
        x.distance2(y);
    }


    @Test
    public void testReduce3AlexBug() {
        val arr = Nd4j.linspace(1,100,100).reshape('f', 10, 10).dup('c');
        val arr2 = Nd4j.linspace(1,100,100).reshape('c', 10, 10);
        val out = Nd4j.getExecutioner().exec(new EuclideanDistance(arr, arr2), 1);
        val exp = Nd4j.create(new double[] {151.93748, 128.86038, 108.37435, 92.22256, 82.9759, 82.9759, 92.22256, 108.37435, 128.86038, 151.93748});

        assertEquals(exp, out);
    }

    @Test
    public void testAllDistancesEdgeCase1() {
        val x = Nd4j.create(400, 20).assign(2.0);
        val y = Nd4j.ones(1, 20);
        val z = Transforms.allEuclideanDistances(x, y, 1);

        val exp = Nd4j.create(400, 1).assign(4.47214);

        assertEquals(exp, z);
    }

    @Test
    public void testConcat_1() throws Exception{
        for(char order : new char[]{'c', 'f'}) {

            INDArray arr1 = Nd4j.create(new double[]{1, 2}, order);
            INDArray arr2 = Nd4j.create(new double[]{3, 4}, order);

            INDArray out = Nd4j.concat(0, arr1, arr2);
            Nd4j.getExecutioner().commit();
            INDArray exp = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
            assertEquals(String.valueOf(order), exp, out);
        }
    }

    @Test
    public void testRdiv()    {
        final INDArray a = Nd4j.create(new double[]{2.0, 2.0, 2.0, 2.0});
        final INDArray b = Nd4j.create(new double[]{1.0, 2.0, 4.0, 8.0});
        final INDArray c = Nd4j.create(new double[]{2.0, 2.0}).reshape(2, 1);
        final INDArray d = Nd4j.create(new double[]{1.0, 2.0, 4.0, 8.0}).reshape(2, 2);

        final INDArray expected = Nd4j.create(new double[]{2.0, 1.0, 0.5, 0.25});
        final INDArray expected2 = Nd4j.create(new double[]{2.0, 1.0, 0.5, 0.25}).reshape(2, 2);

        assertEquals(expected, a.div(b));
        assertEquals(expected, b.rdiv(a));
        assertEquals(expected, b.rdiv(2));
        assertEquals(expected2, d.rdivColumnVector(c));

        assertEquals(expected, b.rdiv(Nd4j.scalar(2)));
        assertEquals(expected, b.rdivColumnVector(Nd4j.scalar(2)));
    }

    @Test
    public void testRsub()    {
        final INDArray a = Nd4j.create(new double[]{2.0, 2.0, 2.0, 2.0});
        final INDArray b = Nd4j.create(new double[]{1.0, 2.0, 4.0, 8.0});
        final INDArray c = Nd4j.create(new double[]{2.0, 2.0}).reshape(2, 1);
        final INDArray d = Nd4j.create(new double[]{1.0, 2.0, 4.0, 8.0}).reshape('c',2, 2);

        final INDArray expected = Nd4j.create(new double[]{1.0, 0.0, -2.0, -6.0});
        final INDArray expected2 = Nd4j.create(new double[]{1, 0, -2.0, -6.0}).reshape('c',2, 2);

        assertEquals(expected, a.sub(b));
        assertEquals(expected, b.rsub(a));
        assertEquals(expected, b.rsub(2));
        assertEquals(expected2, d.rsubColumnVector(c));

        assertEquals(expected, b.rsub(Nd4j.scalar(2)));
        assertEquals(expected, b.rsubColumnVector(Nd4j.scalar(2)));
    }


    @Test
    public void testHalfStuff() {
        if (!Nd4j.getExecutioner().getClass().getSimpleName().toLowerCase().contains("cuda"))
            return;

        val dtype = Nd4j.dataType();
        Nd4j.setDataType(DataBuffer.Type.HALF);

        val arr = Nd4j.ones(3, 3);
        arr.addi(2.0f);

        val exp = Nd4j.create(3, 3).assign(3.0f);

        assertEquals(exp, arr);

        Nd4j.setDataType(dtype);
    }


    @Test
    public void testInconsistentOutput(){
        INDArray in = Nd4j.rand(1, 802816);
        INDArray W = Nd4j.rand(802816, 1);
        INDArray b = Nd4j.create(1);
        INDArray out = fwd(in, W, b);

        for(int i=0;i<100;i++){
            INDArray out2 = fwd(in, W, b);  //l.activate(inToLayer1, false, LayerWorkspaceMgr.noWorkspaces());
            assertEquals("Failed at iteration [" + String.valueOf(i) + "]", out, out2);
        }
    }

    @Test
    public void test3D_create_1() {
        val jArray = new float[2][3][4];

        fillJvmArray3D(jArray);

        val iArray = Nd4j.create(jArray);
        val fArray = ArrayUtil.flatten(jArray);

        assertArrayEquals(new long[]{2, 3, 4}, iArray.shape());

        assertArrayEquals(fArray, iArray.data().asFloat(), 1e-5f);

        int cnt = 0;
        for (val f : fArray)
            assertTrue("Failed for element [" + cnt++ +"]",f > 0.0f);
    }


    @Test
    public void test4D_create_1() {
        val jArray = new float[2][3][4][5];

        fillJvmArray4D(jArray);

        val iArray = Nd4j.create(jArray);
        val fArray = ArrayUtil.flatten(jArray);

        assertArrayEquals(new long[]{2, 3, 4, 5}, iArray.shape());

        assertArrayEquals(fArray, iArray.data().asFloat(), 1e-5f);

        int cnt = 0;
        for (val f : fArray)
            assertTrue("Failed for element [" + cnt++ +"]",f > 0.0f);
    }

    @Test
    public void testBroadcast_1() {
        val array1 = Nd4j.linspace(1, 10, 10).reshape(5, 1, 2).broadcast(5, 4, 2);
        val array2 = Nd4j.linspace(1, 20, 20).reshape(5, 4, 1).broadcast(5, 4, 2);
        val exp = Nd4j.create(new float[] {2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 8.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 17.0f, 17.0f, 18.0f, 20.0f, 21.0f, 21.0f, 22.0f, 22.0f, 23.0f, 23.0f, 24.0f, 26.0f, 27.0f, 27.0f, 28.0f, 28.0f, 29.0f, 29.0f, 30.0f}).reshape(5,4,2);

        array1.addi(array2);

        assertEquals(exp, array1);
    }


    @Test
    public void testAddiColumnEdge(){
        INDArray arr1 = Nd4j.create(1, 5);
        arr1.addiColumnVector(Nd4j.ones(1));
        assertEquals(Nd4j.ones(1,5), arr1);
    }


    @Test
    public void testMmulViews_1() {
        val arrayX = Nd4j.linspace(1, 27, 27).reshape(3, 3, 3);

        val arrayA = Nd4j.linspace(1, 9, 9).reshape(3, 3);

        val arrayB = arrayX.dup('f');

        val arraya = arrayX.slice(0);
        val arrayb = arrayB.slice(0);

        val exp = arrayA.mmul(arrayA);

        assertEquals(exp, arraya.mmul(arrayA));
        assertEquals(exp, arraya.mmul(arraya));

        assertEquals(exp, arrayb.mmul(arrayb));
    }

    @Test
    public void testTile_1() {
        val array = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        val exp = Nd4j.create(new double[] {1.000000, 2.000000, 3.000000, 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 4.000000, 5.000000, 6.000000, 1.000000, 2.000000, 3.000000, 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 4.000000, 5.000000, 6.000000}, new int[] {4, 6});
        val output = Nd4j.create(4, 6);

        val op = DynamicCustomOp.builder("tile")
                .addInputs(array)
                .addIntegerArguments(2, 2)
                .addOutputs(output)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }

    @Test
    public void testRelativeError_1() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.ones(10, 10);
        val exp = Nd4j.ones(10, 10);

        Nd4j.getExecutioner().exec(new BinaryRelativeError(arrayX, arrayY, arrayX, 0.1));

        assertEquals(exp, arrayX);
    }


    @Test
    public void testMeshGrid(){

        INDArray x1 = Nd4j.create(new double[]{1,2,3,4});
        INDArray y1 = Nd4j.create(new double[]{5,6,7});

        INDArray expX = Nd4j.create(new double[][]{
                {1,2,3,4},
                {1,2,3,4},
                {1,2,3,4}});
        INDArray expY = Nd4j.create(new double[][]{
                {5,5,5,5},
                {6,6,6,6},
                {7,7,7,7}});
        INDArray[] exp = new INDArray[]{expX, expY};

        INDArray[] out1 = Nd4j.meshgrid(x1, y1);
        assertArrayEquals(out1, exp);

        INDArray[] out2 = Nd4j.meshgrid(x1.transpose(), y1.transpose());
        assertArrayEquals(out2, exp);

        INDArray[] out3 = Nd4j.meshgrid(x1, y1.transpose());
        assertArrayEquals(out3, exp);

        INDArray[] out4 = Nd4j.meshgrid(x1.transpose(), y1);
        assertArrayEquals(out4, exp);

        //Test views:
        INDArray x2 = Nd4j.create(1,9).get(NDArrayIndex.all(), NDArrayIndex.interval(1,2,7, true))
                .assign(x1);
        INDArray y2 = Nd4j.create(1,7).get(NDArrayIndex.all(), NDArrayIndex.interval(1,2,5, true))
                .assign(y1);

        INDArray[] out5 = Nd4j.meshgrid(x2, y2);
        assertArrayEquals(out5, exp);
    }

    @Test
    public void testAccumuationWithoutAxis_1() {
        val array = Nd4j.create(3, 3).assign(1.0);

        val result = array.sum();

        assertEquals(1, result.length());
        assertEquals(9.0, result.getDouble(0), 1e-5);
    }

    @Test
    public void testSummaryStatsEquality_1() {
        log.info("Datatype: {}", Nd4j.dataType());

        for(boolean biasCorrected : new boolean[]{false, true}) {

            INDArray indArray1 = Nd4j.rand(1, 4, 10);
            double std = indArray1.stdNumber(biasCorrected).doubleValue();

            val standardDeviation = new org.apache.commons.math3.stat.descriptive.moment.StandardDeviation(biasCorrected);
            double std2 = standardDeviation.evaluate(indArray1.data().asDouble());
            log.info("Bias corrected = {}", biasCorrected);
            log.info("nd4j std: {}", std);
            log.info("apache math3 std: {}", std2);

            assertEquals(std, std2, 1e-5);
        }
    }

    @Test
    public void testMeanEdgeCase_C(){
        INDArray arr = Nd4j.linspace(1, 30,30).reshape(new int[]{3,10,1}).dup('c');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));

        assertEquals(exp, arr2);
    }

    @Test
    public void testMeanEdgeCase_F(){
        INDArray arr = Nd4j.linspace(1, 30,30).reshape(new int[]{3,10,1}).dup('f');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));

        assertEquals(exp, arr2);
    }

    @Test
    public void testMeanEdgeCase2_C(){
        INDArray arr = Nd4j.linspace(1, 60,60).reshape(new int[]{3,10,2}).dup('c');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
        exp.addi(arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)));
        exp.divi(2);


        assertEquals(exp, arr2);
    }

    @Test
    public void testMeanEdgeCase2_F(){
        INDArray arr = Nd4j.linspace(1, 60,60).reshape(new int[]{3,10,2}).dup('f');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
        exp.addi(arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)));
        exp.divi(2);


        assertEquals(exp, arr2);
    }

    @Test
    public void testLegacyDeserialization_1() throws Exception {
        val f = new ClassPathResource("legacy/NDArray.bin").getFile();

        val array = Nd4j.read(new FileInputStream(f));
        val exp = Nd4j.linspace(1, 120, 120).reshape(2, 3, 4, 5);

        assertEquals(120, array.length());
        assertArrayEquals(new long[]{2, 3, 4, 5}, array.shape());
        assertEquals(exp, array);
    }

    @Test
    public void testTearPile_1() {
        val source = Nd4j.rand(new int[]{10, 15});

        val list = Nd4j.tear(source, 1);

        // just want to ensure that axis is right one
        assertEquals(10, list.length);

        val result = Nd4j.pile(list);

        assertEquals(source.shapeInfoDataBuffer(), result.shapeInfoDataBuffer());
        assertEquals(source, result);
    }

    @Test
    public void testVariance_4D_1() {
        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.FLOAT);

        val x = Nd4j.ones(10, 20, 30, 40);
        val result = x.var(false, 0, 2, 3);

        Nd4j.getExecutioner().commit();

        log.info("Result shape: {}", result.shapeInfoDataBuffer().asLong());

        Nd4j.setDataType(dtype);
    }


    @Test
    public void testEye(){

        int[] rows = new int[]{3,3,3,3};
        int[] cols = new int[]{3,2,2,2};
        int[][] batch = new int[][]{null, null, {4}, {3,3}};
        INDArray[] expOut = new INDArray[4];

        expOut[0] = Nd4j.eye(3);
        expOut[1] = Nd4j.create(new double[][]{{1,0,0},{0,1,0}});
        expOut[2] = Nd4j.create(4,3,2);
        for( int i=0; i<4; i++ ){
            expOut[2].get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).assign(expOut[1]);
        }
        expOut[3] = Nd4j.create(3,3,3,2);
        for( int i=0; i<3; i++ ){
            for( int j=0; j<3; j++ ) {
                expOut[3].get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all()).assign(expOut[1]);
            }
        }


        for(int i=0; i<3; i++ ) {
            INDArray out = Nd4j.create(expOut[i].shape());

            DynamicCustomOp.DynamicCustomOpsBuilder op = DynamicCustomOp.builder("eye")
                    .addOutputs(out)
                    .addIntegerArguments(rows[i], cols[i]);
            if(batch[i] != null){
                op.addIntegerArguments(batch[i]);
            }

            Nd4j.getExecutioner().exec(op.build());

            assertEquals(expOut[i], out);
        }
    }

    @Test
    public void testTranspose_Custom(){

        INDArray arr = Nd4j.linspace(1,15, 15).reshape(5,3);
        INDArray out = Nd4j.create(3,5);

        val op = DynamicCustomOp.builder("transpose")
                .addInputs(arr)
                .addOutputs(out)
                .build();

        Nd4j.getExecutioner().exec(op);

        INDArray exp = arr.transpose();
        assertEquals(exp, out);
    }

    @Test
    public void testRowColumnOpsRank1(){

        for( int i=0; i<6; i++ ) {
            INDArray orig = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);
            INDArray in1r = orig.dup();
            INDArray in2r = orig.dup();
            INDArray in1c = orig.dup();
            INDArray in2c = orig.dup();

            INDArray rv1 = Nd4j.create(new double[]{1, 2, 3, 4}, new long[]{1, 4});
            INDArray rv2 = Nd4j.create(new double[]{1, 2, 3, 4}, new long[]{4});
            INDArray cv1 = Nd4j.create(new double[]{1, 2, 3}, new long[]{3, 1});
            INDArray cv2 = Nd4j.create(new double[]{1, 2, 3}, new long[]{3});

            switch (i){
                case 0:
                    in1r.addiRowVector(rv1);
                    in2r.addiRowVector(rv2);
                    in1c.addiColumnVector(cv1);
                    in2c.addiColumnVector(cv2);
                    break;
                case 1:
                    in1r.subiRowVector(rv1);
                    in2r.subiRowVector(rv2);
                    in1c.subiColumnVector(cv1);
                    in2c.subiColumnVector(cv2);
                    break;
                case 2:
                    in1r.muliRowVector(rv1);
                    in2r.muliRowVector(rv2);
                    in1c.muliColumnVector(cv1);
                    in2c.muliColumnVector(cv2);
                    break;
                case 3:
                    in1r.diviRowVector(rv1);
                    in2r.diviRowVector(rv2);
                    in1c.diviColumnVector(cv1);
                    in2c.diviColumnVector(cv2);
                    break;
                case 4:
                    in1r.rsubiRowVector(rv1);
                    in2r.rsubiRowVector(rv2);
                    in1c.rsubiColumnVector(cv1);
                    in2c.rsubiColumnVector(cv2);
                    break;
                case 5:
                    in1r.rdiviRowVector(rv1);
                    in2r.rdiviRowVector(rv2);
                    in1c.rdiviColumnVector(cv1);
                    in2c.rdiviColumnVector(cv2);
                    break;
                default:
                    throw new RuntimeException();
            }


            assertEquals(in1r, in2r);
            assertEquals(in1c, in2c);

        }
    }

    @Test
    public void testEmptyShapeRank0(){
        Nd4j.getRandom().setSeed(12345);
        int[] s = new int[0];
        INDArray create = Nd4j.create(s);
        INDArray zeros = Nd4j.zeros(s);
        INDArray ones = Nd4j.ones(s);
        INDArray uninit = Nd4j.createUninitialized(s).assign(0);
        INDArray rand = Nd4j.rand(s);

        INDArray tsZero = Nd4j.trueScalar(0);
        INDArray tsOne = Nd4j.trueScalar(1);
        Nd4j.getRandom().setSeed(12345);
        INDArray tsRand = Nd4j.trueScalar(Nd4j.rand(new int[]{1,1}).getDouble(0));
        assertEquals(tsZero, create);
        assertEquals(tsZero, zeros);
        assertEquals(tsOne, ones);
        assertEquals(tsZero, uninit);
        assertEquals(tsRand, rand);


        Nd4j.getRandom().setSeed(12345);
        long[] s2 = new long[0];
        create = Nd4j.create(s2);
        zeros = Nd4j.zeros(s2);
        ones = Nd4j.ones(s2);
        uninit = Nd4j.createUninitialized(s2).assign(0);
        rand = Nd4j.rand(s2);

        assertEquals(tsZero, create);
        assertEquals(tsZero, zeros);
        assertEquals(tsOne, ones);
        assertEquals(tsZero, uninit);
        assertEquals(tsRand, rand);
    }

    @Test
    public void testScalarView_1() {
        val array = Nd4j.linspace(1, 5, 5);
        val exp = Nd4j.create(new double[]{1.0, 2.0, 5.0, 4.0, 5.0});
        val scalar = array.getScalar(2);

        assertEquals(3.0, scalar.getDouble(0), 1e-5);
        scalar.addi(2.0);

        assertEquals(exp, array);
    }

    @Test
    public void testScalarView_2() {
        val array = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        val exp = Nd4j.create(new double[]{1.0, 2.0, 5.0, 4.0}).reshape(2, 2);
        val scalar = array.getScalar(1, 0);

        assertEquals(3.0, scalar.getDouble(0), 1e-5);
        scalar.addi(2.0);

        assertEquals(exp, array);
    }

    @Test
    public void testSomething_1() {
        val arrayX = Nd4j.create(128, 128, 'f');
        val arrayY = Nd4j.create(128, 128, 'f');
        val arrayZ = Nd4j.create(128, 128, 'f');

        int iterations = 10000;
        // warmup
        for (int e = 0; e < 1000; e++)
            arrayX.addi(arrayY);

        for (int e = 0; e < iterations; e++) {
            val c = new GemmParams(arrayX, arrayY, arrayZ);
        }

        val tS = System.nanoTime();
        for (int e = 0; e < iterations; e++) {
            //val c = new GemmParams(arrayX, arrayY, arrayZ);
            arrayX.mmuli(arrayY, arrayZ);
        }

        val tE = System.nanoTime();

        log.info("Average time: {}", ((tE - tS) / iterations));
    }

    ///////////////////////////////////////////////////////
    protected static void fillJvmArray3D(float[][][] arr) {
        int cnt = 1;
        for (int i = 0; i < arr.length; i++)
            for (int j = 0; j < arr[0].length; j++)
                for (int k = 0; k < arr[0][0].length; k++)
                    arr[i][j][k] = (float) cnt++;
    }


    protected static void fillJvmArray4D(float[][][][] arr) {
        int cnt = 1;
        for (int i = 0; i < arr.length; i++)
            for (int j = 0; j < arr[0].length; j++)
                for (int k = 0; k < arr[0][0].length; k++)
                    for (int m = 0; m < arr[0][0][0].length; m++)
                        arr[i][j][k][m] = (float) cnt++;
    }


    private static INDArray fwd(INDArray input, INDArray W, INDArray b){
        INDArray ret = Nd4j.createUninitialized(input.size(0), W.size(1));
        input.mmuli(W, ret);
        ret.addiRowVector(b);

        return ret;
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
