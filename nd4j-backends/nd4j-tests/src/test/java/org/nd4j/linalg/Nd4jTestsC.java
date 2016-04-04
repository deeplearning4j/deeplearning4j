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
 *
 */

package org.nd4j.linalg;


import org.apache.commons.io.FilenameUtils;
import org.apache.commons.math3.util.Pair;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.iter.INDArrayIterator;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

/**
 * NDArrayTests
 *
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public  class Nd4jTestsC extends BaseNd4jTest {


    public Nd4jTestsC(Nd4jBackend backend) {
        super(backend);
    }


    @Before
    public void before() {
        super.before();
        Nd4j.factory().setDType(DataBuffer.Type.DOUBLE);
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        Nd4j.getRandom().setSeed(123);

    }

    @After
    public void after() {
        super.after();
        Nd4j.factory().setDType(DataBuffer.Type.DOUBLE);
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

    }


    @Test
    public void testSerialization() throws Exception {
        Nd4j.getRandom().setSeed(12345);
        INDArray arr = Nd4j.rand(1,20);

        String temp = System.getProperty("java.io.tmpdir");

        String outPath = FilenameUtils.concat(temp,"dl4jtestserialization.bin");

        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(outPath)))){
            Nd4j.write(arr,dos);
        }

        INDArray in;
        try(DataInputStream dis = new DataInputStream(new FileInputStream(outPath))){
            in = Nd4j.read(dis);
        }

        INDArray inDup = in.dup();

        System.out.println(in);
        System.out.println(inDup);

        assertEquals(arr,in);       //Passes:   Original array "in" is OK, but array "inDup" is not!?
        assertEquals(in,inDup);     //Fails
    }

    @Test
    public void testTensorAlongDimension2() {
        INDArray array = Nd4j.create( new float[100], new int[]{50,1,2});
        assertArrayEquals(new int[]{1,2},array.slice(0,0).shape());

    }

    @Test
    public void testIsMax() {
        INDArray arr = Nd4j.create(new double[]{1,2,4,3},new int[]{2,2});
        INDArray assertion = Nd4j.create(new double[]{0,0,1,0},new int[]{2,2});
        INDArray test = Nd4j.getExecutioner().exec(new IsMax(arr)).z();
        assertEquals(assertion,test);
    }

    @Test
    public void testArgMax() {
        INDArray toArgMax = Nd4j.linspace(1,24,24).reshape(4, 3, 2);
        INDArray argMaxZero = Nd4j.argMax(toArgMax,0);
        INDArray  argMax = Nd4j.argMax(toArgMax, 1);
        INDArray argMaxTwo = Nd4j.argMax(toArgMax,2);
        INDArray valueArray = Nd4j.valueArrayOf(new int[]{4, 2}, 2.0);
        INDArray valueArrayTwo = Nd4j.valueArrayOf(new int[]{3,2},3.0);
        INDArray valueArrayThree = Nd4j.valueArrayOf(new int[]{4,3},1.0);
        assertEquals(valueArrayTwo, argMaxZero);
        assertEquals(valueArray, argMax);

        assertEquals(valueArrayThree,argMaxTwo);
    }


    @Test
    public void testScalarOps() throws Exception {
        INDArray n = Nd4j.create(Nd4j.ones(27).data(), new int[]{3, 3, 3});
        assertEquals(27d, n.length(), 1e-1);
        n.checkDimensions(n.addi(Nd4j.scalar(1d)));
        n.checkDimensions(n.subi(Nd4j.scalar(1.0d)));
        n.checkDimensions(n.muli(Nd4j.scalar(1.0d)));
        n.checkDimensions(n.divi(Nd4j.scalar(1.0d)));

        n = Nd4j.create(Nd4j.ones(27).data(), new int[]{3, 3, 3});
        assertEquals(getFailureMessage(), 27, n.sumNumber().doubleValue(), 1e-1);
        INDArray a = n.slice(2);
        assertEquals(getFailureMessage(), true, Arrays.equals(new int[]{3, 3}, a.shape()));

    }


    @Test
    public void testTensorAlongDimension() {
        int[] shape = new int[]{4,5,7};
        int length = ArrayUtil.prod(shape);
        INDArray arr = Nd4j.linspace(1, length, length).reshape(shape);


        int[] dim0s = {0,1,2,0,1,2};
        int[] dim1s = {1,0,0,2,2,1};

        double[] sums = {1350.,  1350.,  1582,  1582,  630,  630};

        for( int i = 0; i < dim0s.length; i++) {
            int firstDim = dim0s[i];
            int secondDim = dim1s[i];
            INDArray tad = arr.tensorAlongDimension(0, firstDim, secondDim);
            tad.sumNumber();
//            assertEquals("I " + i + " failed ",sums[i],tad.sumNumber().doubleValue(),1e-1);
        }

        INDArray testMem = Nd4j.create(10,10);
    }



    @Test
    public void testGetDouble() {
        INDArray n2 = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2});
        INDArray swapped = n2.swapAxes(n2.shape().length - 1, 1);
        INDArray slice0 = swapped.slice(0).slice(1);
        INDArray assertion = Nd4j.create(new double[]{2, 4, 6, 8, 10});
        assertEquals(assertion,slice0);
    }

    @Test
    public void testWriteTxt() throws Exception {
        INDArray row = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Nd4j.write(bos, row);
        String s = new String(bos.toByteArray());
        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        INDArray ret = Nd4j.read(bis);
        assertEquals(row, ret);

    }

    @Test
    public void test2dMatrixOrderingSwitch() throws Exception {
        char order = Nd4j.order();
        INDArray c = Nd4j.create(new double[][]{{1, 2}, {3, 4}}, 'c');
        assertEquals('c', c.ordering());
        assertEquals(order,Nd4j.order().charValue());
        INDArray f = Nd4j.create(new double[][]{{1, 2}, {3, 4}}, 'f');
        assertEquals('f', f.ordering());
        assertEquals(order,Nd4j.order().charValue());
    }


    @Test
    public void testMatrix() {
        INDArray arr = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray brr = Nd4j.create(new float[]{5, 6}, new int[]{1, 2});
        INDArray row = arr.getRow(0);
        row.subi(brr);
        assertEquals(Nd4j.create(new double[]{-4, -4}), arr.getRow(0));

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
        INDArray arr = Nd4j.create(new double[][]{
                {1, 2, 3}, {4, 5, 6}
        });

        INDArray assertion = Nd4j.create(new double[][]{
                {14, 32}, {32, 77}
        });

        INDArray test = arr.mmul(arr.transpose());
        assertEquals(getFailureMessage(), assertion, test);

    }







    @Test
    public void testReadWrite() throws Exception {
        INDArray write = Nd4j.linspace(1, 4, 4);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        SerializationUtils.writeObject(write,bos);
        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        INDArray read = SerializationUtils.readObject(bis);
        assertEquals(write, read);

    }




    @Test
    public void testReadWriteDouble() throws Exception {
        INDArray write = Nd4j.linspace(1, 4, 4);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(write, dos);

        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        DataInputStream dis = new DataInputStream(bis);
        INDArray read = Nd4j.read(dis);
        assertEquals(write, read);

    }



    @Test
    public void testSubiRowVector() throws Exception {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4).reshape('c',2, 2);
        INDArray row1 = oneThroughFour.getRow(1);
        oneThroughFour.subiRowVector(row1);
        INDArray result = Nd4j.create(new float[]{-2, -2, 0, 0}, new int[]{2, 2});
        assertEquals(getFailureMessage(), result, oneThroughFour);

    }


    @Test
    public void testAddiRowVectorWithScalar(){
        INDArray colVector = Nd4j.create(5, 1).assign(0.0);
        INDArray scalar = Nd4j.create(1, 1).assign(0.0);
        scalar.putScalar(0, 1);

        assertEquals(scalar.getDouble(0), 1.0, 0.0);

        colVector.addiRowVector(scalar);    //colVector is all zeros after this
        for( int i = 0; i < 5; i++)
            assertEquals(colVector.getDouble(i),1.0,0.0);
    }

    @Test
    public void testTADOnVector() {

        Nd4j.getRandom().setSeed(12345);
        INDArray rowVec = Nd4j.rand(1, 10);
        INDArray thirdElem = rowVec.tensorAlongDimension(2, 0);

        assertEquals(rowVec.getDouble(2),thirdElem.getDouble(0),0.0);

        thirdElem.putScalar(0, 5);
        assertEquals(5, thirdElem.getDouble(0), 0.0);

        assertEquals(5, rowVec.getDouble(2), 0.0);    //Both should be modified if thirdElem is a view

        //Same thing for column vector:
        INDArray colVec = Nd4j.rand(10,1);
        thirdElem = colVec.tensorAlongDimension(2,1);

        assertEquals(colVec.getDouble(2), thirdElem.getDouble(0), 0.0);

        thirdElem.putScalar(0, 5);
        assertEquals(5, thirdElem.getDouble(0), 0.0);
        assertEquals(5,colVec.getDouble(2),0.0);
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


        INDArray expected = Nd4j.repeat(Nd4j.scalar(2), 2).reshape(2,1);

        Accumulation accum = Nd4j.getOpFactory().createAccum("euclidean", values, values2);
        INDArray results = Nd4j.getExecutioner().exec(accum, 1);
        assertEquals(expected, results);

    }

    @Test
    public void testBroadCasting() {
        INDArray first = Nd4j.arange(0, 3).reshape(3, 1);
        INDArray ret = first.broadcast(3, 4);
        INDArray testRet = Nd4j.create(new double[][]{
                {0, 0, 0, 0},
                {1, 1, 1, 1},
                {2, 2, 2, 2}
        });
        assertEquals(testRet, ret);
        INDArray r = Nd4j.arange(0, 4).reshape(1, 4);
        INDArray r2 = r.broadcast(4, 4);
        INDArray testR2 = Nd4j.create(new double[][]{
                {0, 1, 2, 3},
                {0, 1, 2, 3},
                {0, 1, 2, 3},
                {0, 1, 2, 3}
        });
        assertEquals(testR2, r2);

    }


    @Test
    public void testGetColumns() {
        INDArray matrix = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray matrixGet = matrix.getColumns(new int[]{1, 2});
        INDArray matrixAssertion = Nd4j.create(new double[][]{{2, 3}, {5, 6}});
        assertEquals(matrixAssertion, matrixGet);
    }

    @Test
    public void testSort() throws Exception {
        INDArray toSort = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray ascending = Nd4j.sort(toSort.dup(), 1, true);
        //rows already already sorted
        assertEquals(toSort, ascending);

        INDArray columnSorted = Nd4j.create(new float[]{2, 1, 4, 3}, new int[]{2, 2});
        INDArray sorted = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(columnSorted, sorted);
    }

    @Test
    public void testSortRows() {
        int nRows = 10;
        int nCols = 5;
        java.util.Random r = new java.util.Random(12345);

        for( int i = 0; i < nCols; i++) {
            INDArray in = Nd4j.linspace(1,nRows * nCols,nRows * nCols).reshape(nRows,nCols);

            List<Integer> order = new ArrayList<>(nRows);
            //in.row(order(i)) should end up as out.row(i) - ascending
            //in.row(order(i)) should end up as out.row(nRows-j-1) - descending
            for( int j = 0; j<nRows; j++ ) order.add(j);
            Collections.shuffle(order, r);
            for( int j = 0; j<nRows; j++ )
                in.putScalar(new int[]{j,i},order.get(j));

            INDArray outAsc = Nd4j.sortRows(in, i, true);
            INDArray outDesc = Nd4j.sortRows(in, i, false);

            for( int j = 0; j < nRows; j++) {
                assertEquals(outAsc.getDouble(j,i),j,1e-1);
                int origRowIdxAsc = order.indexOf(j);
                assertTrue(outAsc.getRow(j).equals(in.getRow(origRowIdxAsc)));

                assertTrue(outDesc.getDouble(j,i)==(nRows-j-1));
                int origRowIdxDesc = order.indexOf(nRows-j-1);
                assertTrue(outDesc.getRow(j).equals(in.getRow(origRowIdxDesc)));
            }
        }
    }

    @Test
    public void testToFlattenedOrder() {
        INDArray concatC = Nd4j.linspace(1,4,4).reshape('c',2,2);
        INDArray concatF = Nd4j.create(new int[]{2,2},'f');
        concatF.assign(concatC);
        INDArray assertionC = Nd4j.create(new double[]{1,2,3,4,1,2,3,4});
        INDArray testC = Nd4j.toFlattened('c',concatC,concatF);
        assertEquals(assertionC,testC);
        INDArray test = Nd4j.toFlattened('f',concatC,concatF);
        INDArray assertion = Nd4j.create(new double[]{1,3,2,4,1,3,2,4});
        assertEquals(assertion,test);


    }

    @Test
    public void testZero() {
        Nd4j.ones(11).sumNumber();
        Nd4j.ones(12).sumNumber();
        Nd4j.ones(2).sumNumber();
    }


    @Test
    public void testSumNumberRepeatability() {
        INDArray arr = Nd4j.ones(1,450).reshape('c',150,3);

        double first = arr.sumNumber().doubleValue();
        double assertion = 450;
        assertEquals(assertion,first,1e-1);
        for( int i = 0; i < 50; i++) {
            double second = arr.sumNumber().doubleValue();
            assertEquals(assertion,second,1e-1);
            assertEquals(String.valueOf(i),first,second,1e-2);
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
        INDArray f2d = Nd4j.create(new int[]{rows, cols}, 'f').assign(c2d).addi(0.1);

        INDArray c3d = Nd4j.linspace(1, length3d, length3d).reshape('c', rows, cols, dim2);
        INDArray f3d = Nd4j.create(new int[]{rows, cols, dim2}).assign(c3d).addi(0.3);
        c3d.addi(0.2);

        INDArray c4d = Nd4j.linspace(1, length4d, length4d).reshape('c', rows, cols, dim2, dim3);
        INDArray f4d = Nd4j.create(new int[]{rows, cols, dim2, dim3}).assign(c4d).addi(0.3);
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
        assertEquals(toFlattenedViaIterator('c', c2d, f2d, c3d, f3d, c4d, f4d), Nd4j.toFlattened('c', c2d, f2d, c3d, f3d, c4d, f4d));
        assertEquals(toFlattenedViaIterator('f', c2d, f2d, c3d, f3d, c4d, f4d), Nd4j.toFlattened('f', c2d, f2d, c3d, f3d, c4d, f4d));
    }

    @Test
    public void testToFlattenedOnViews(){

        int rows = 8;
        int cols = 8;
        int dim2 = 4;
        int length = rows*cols;
        int length3d = rows*cols*dim2;

        INDArray first = Nd4j.linspace(1,length,length).reshape('c',rows,cols);
        INDArray second = Nd4j.create('f',rows,cols).assign(first);
        INDArray third = Nd4j.linspace(1,length3d,length3d).reshape('c',rows,cols,dim2);
        first.addi(0.1);
        second.addi(0.2);
        third.addi(0.3);

        first = first.get(NDArrayIndex.interval(4,8), NDArrayIndex.interval(0,2,8));
        second = second.get(NDArrayIndex.interval(3,7), NDArrayIndex.all());
        third = third.permute(0,2,1);

        assertEquals(Nd4j.toFlattened('c', first, second, third), toFlattenedViaIterator('c', first, second, third));
        assertEquals(Nd4j.toFlattened('f', first, second, third), toFlattenedViaIterator('f', first, second, third));
    }

    private static INDArray toFlattenedViaIterator(char order, INDArray... toFlatten) {
        int length = 0;
        for (INDArray i : toFlatten) length += i.length();

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
    public void testSortColumns() {
        int nRows = 5;
        int nCols = 10;
        java.util.Random r = new java.util.Random(12345);

        for( int i = 0; i < nRows; i++) {
            INDArray in = Nd4j.rand(new int[]{nRows,nCols});

            List<Integer> order = new ArrayList<>(nRows);
            for( int j = 0; j < nCols; j++) order.add(j);
            Collections.shuffle(order, r);
            for( int j = 0; j < nCols; j++) in.putScalar(new int[]{i,j},order.get(j));

            INDArray outAsc = Nd4j.sortColumns(in, i, true);
            INDArray outDesc = Nd4j.sortColumns(in, i, false);

            for( int j = 0; j < nCols; j++ ){
                assertTrue(outAsc.getDouble(i,j)==j);
                int origColIdxAsc = order.indexOf(j);
                assertTrue(outAsc.getColumn(j).equals(in.getColumn(origColIdxAsc)));

                assertTrue(outDesc.getDouble(i,j)==(nCols-j-1));
                int origColIdxDesc = order.indexOf(nCols-j-1);
                assertTrue(outDesc.getColumn(j).equals(in.getColumn(origColIdxDesc)));
            }
        }
    }


    @Test
    public void testAddVectorWithOffset() throws Exception {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row1 = oneThroughFour.getRow(1);
        row1.addi(1);
        INDArray result = Nd4j.create(new float[]{1, 2, 4, 5}, new int[]{2, 2});
        assertEquals(getFailureMessage(),result, oneThroughFour);


    }



    @Test
    public void testLinearViewGetAndPut() throws Exception {
        INDArray test = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray linear = test.linearView();
        linear.putScalar(2, 6);
        linear.putScalar(3, 7);
        assertEquals(getFailureMessage(), 6, linear.getFloat(2), 1e-1);
        assertEquals(getFailureMessage(),7, linear.getFloat(3), 1e-1);
    }



    @Test
    public void testRowVectorGemm() {
        INDArray linspace = Nd4j.linspace(1, 4, 4);
        INDArray other = Nd4j.linspace(1,16,16).reshape(4, 4);
        INDArray result = linspace.mmul(other);
        INDArray assertion = Nd4j.create(new double[]{90, 100, 110, 120});
        assertEquals(assertion,result);
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
        INDArray arr = Nd4j.linspace(0,7,8).reshape('c',2,2,2);
         /* [0.0,4.0,2.0,6.0,1.0,5.0,3.0,7.0]
         *
         * Rank: 3,Offset: 0
             Order: f shape: [2,2,2], stride: [1,2,4]*/
        INDArray arrF = Nd4j.create(new int[]{2,2,2},'f').assign(arr);

        assertEquals(arr,arrF);
        //0,2,4,6 and 1,3,5,7
        assertEquals(Nd4j.create(new double[]{12,16}),arr.sum(0,1));
        //0,1,4,5 and 2,3,6,7
        assertEquals(Nd4j.create(new double[]{10,18}),arr.sum(0,2));
        //0,2,4,6 and 1,3,5,7
        assertEquals(Nd4j.create(new double[]{12,16}),arrF.sum(0,1));
        //0,1,4,5 and 2,3,6,7
        assertEquals(Nd4j.create(new double[]{10,18}),arrF.sum(0,2));

        //0,1,2,3 and 4,5,6,7
        assertEquals(Nd4j.create(new double[]{6,22}),arr.sum(1,2));
        //0,1,2,3 and 4,5,6,7
        assertEquals(Nd4j.create(new double[]{6,22}),arrF.sum(1,2));


        double[] data = new double[]{10, 26,42};
        INDArray assertion = Nd4j.create(data);
        for(int i = 0; i < data.length; i++) {
            assertEquals(data[i],assertion.getDouble(i),1e-1);
        }

        INDArray twoTwoByThree = Nd4j.linspace(1,12,12).reshape('f',2, 2, 3);
        INDArray multiSum = twoTwoByThree.sum(0, 1);
        assertEquals(assertion,multiSum);
    }


    @Test
    public void testSum2dv2() {
        INDArray in = Nd4j.linspace(1,8,8).reshape('c',2,2,2);

        int[][] dims = new int[][]{{0,1}, {1,0}, {0,2}, {2,0}, {1,2}, {2,1}};
        double[][] exp = new double[][]{{16,20}, {16,20}, {14,22}, {14,22}, {10,26}, {10,26}};

        System.out.println("dims\texpected\t\tactual");
        for(int i = 0; i < dims.length; i++) {
            int[] d = dims[i];
            double[] e = exp[i];

            INDArray out = in.sum(d);

            System.out.println(Arrays.toString(d) + "\t" + Arrays.toString(e) + "\t" + out);
            assertEquals(Nd4j.create(e,out.shape()),out);
        }
    }


    //Passes on 3.9:
    @Test
    public void testSum3Of4_2222() {
        int[] shape = {2, 2, 2, 2};
        int length = ArrayUtil.prod(shape);
        INDArray arrC = Nd4j.linspace(1, length, length).reshape('c', shape);
        INDArray arrF = Nd4j.create(arrC.shape()).assign(arrC);

        int[][] dimsToSum = new int[][]{{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
        double[][] expD = new double[][]{{64, 72}, {60, 76}, {52, 84}, {36, 100}};

        for (int i = 0; i < dimsToSum.length; i++) {
            int[] d = dimsToSum[i];

            INDArray outC = arrC.sum(d);
            INDArray outF = arrF.sum(d);
            INDArray exp = Nd4j.create(expD[i],outC.shape());

            assertEquals(exp, outC);
            assertEquals(exp, outF);

            System.out.println(Arrays.toString(d) + "\t" + outC + "\t" + outF);
        }
    }

    @Test
    public void testBroadcast1d() {
        int[] shape = {4,3,2};
        int[] toBroadcastDims = new int[]{0,1,2};
        int[][] toBroadcastShapes = new int[][]{{1,4}, {1,3}, {1,2}};

        //Expected result values in buffer: c order, need to reshape to {4,3,2}. Values taken from 0.4-rc3.8
        double[][] expFlat = new double[][]{
                {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
                {1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0},
                {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0}};

        double[][] expLinspaced = new double[][]{
                {2.0,3.0,4.0,5.0,6.0,7.0,9.0,10.0,11.0,12.0,13.0,14.0,16.0,17.0,18.0,19.0,20.0,21.0,23.0,24.0,25.0,26.0,27.0,28.0},
                {2.0,3.0,5.0,6.0,8.0,9.0,8.0,9.0,11.0,12.0,14.0,15.0,14.0,15.0,17.0,18.0,20.0,21.0,20.0,21.0,23.0,24.0,26.0,27.0},
                {2.0,4.0,4.0,6.0,6.0,8.0,8.0,10.0,10.0,12.0,12.0,14.0,14.0,16.0,16.0,18.0,18.0,20.0,20.0,22.0,22.0,24.0,24.0,26.0}
        };

        for( int i = 0; i < toBroadcastDims.length; i++) {
            int dim = toBroadcastDims[i];
            int[] vectorShape = toBroadcastShapes[i];
            int length = ArrayUtil.prod(vectorShape);

            INDArray zC = Nd4j.create(shape,'c');
            zC.setData(Nd4j.linspace(1,24,24).data());
            for(int tad = 0; tad < zC.tensorssAlongDimension(dim); tad++) {
                System.out.println("Tad " + tad + " is " + zC.tensorAlongDimension(tad,dim));
            }

            INDArray zF = Nd4j.create(shape,'f');
            zF.assign(zC);
            INDArray toBroadcast = Nd4j.linspace(1,length,length);

            Op opc = new BroadcastAddOp(zC, toBroadcast, zC, dim);
            Op opf = new BroadcastAddOp(zF, toBroadcast, zF, dim);
            INDArray exp = Nd4j.create(expLinspaced[i],shape,'c');
            INDArray expF = Nd4j.create(shape,'f');
            expF.assign(exp);
            for(int tad = 0; tad < zC.tensorssAlongDimension(dim); tad++) {
                System.out.println(zC.tensorAlongDimension(tad,dim).offset() + " and f offset is " + zF.tensorAlongDimension(tad,dim).offset());
            }

            Nd4j.getExecutioner().exec(opc);
            Nd4j.getExecutioner().exec(opf);

            assertEquals(exp,zC);
            assertEquals(exp,zF);
        }
    }

    @Test
    public void testSum3Of4_3322() {
        int[] shape = {3, 3, 2, 2};
        int length = ArrayUtil.prod(shape);
        INDArray arrC = Nd4j.linspace(1, length, length).reshape('c', shape);
        INDArray arrF = Nd4j.create(arrC.shape()).assign(arrC);

        int[][] dimsToSum = new int[][]{{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
        double[][] expD = new double[][]{{324, 342}, {315, 351}, {174, 222, 270}, {78, 222, 366}};

        for (int i = 0; i < dimsToSum.length; i++) {
            int[] d = dimsToSum[i];

            INDArray outC = arrC.sum(d);
            INDArray outF = arrF.sum(d);
            INDArray exp = Nd4j.create(expD[i],outC.shape());

            assertEquals(exp, outC);
            assertEquals(exp, outF);

            //System.out.println(Arrays.toString(d) + "\t" + outC + "\t" + outF);
        }
    }

    @Test
    public void testToFlattened() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        List<INDArray> concat = new ArrayList<>();
        for(int i = 0; i < 3; i++) {
            concat.add(arr.dup());
        }

        INDArray assertion = Nd4j.create(new double[]{1,2,3,4,1,2,3,4,1,2,3,4});
        INDArray flattened = Nd4j.toFlattened(concat);
        assertEquals(assertion,flattened);

    }



    @Test
    public void testDup() {
        for(int x = 0; x < 100; x++) {
            INDArray orig = Nd4j.linspace(1, 4, 4);
            INDArray dup = orig.dup();
            assertEquals(orig, dup);

            INDArray matrix = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
            INDArray dup2 = matrix.dup();
            assertEquals(matrix, dup2);

            INDArray row1 = matrix.getRow(1);
            INDArray dupRow = row1.dup();
            assertEquals(row1, dupRow);


            INDArray columnSorted = Nd4j.create(new float[]{2, 1, 4, 3}, new int[]{2, 2});
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
        INDArray shouldIndex = Nd4j.create(new float[]{1, 0, 1, 0}, new int[]{2, 2});
        assertEquals(shouldIndex, sorted[0]);


    }



    @Test
    public void testSubRowVector() {
        INDArray matrix = Nd4j.linspace(1,6,6).reshape(2, 3);
        INDArray row = Nd4j.linspace(1, 3, 3);
        INDArray test = matrix.subRowVector(row);
        INDArray assertion = Nd4j.create(new double[][]{
                {0, 0, 0}
                , {3, 3, 3}
        });
        assertEquals(assertion,test);

        INDArray threeByThree = Nd4j.linspace(1,9,9).reshape(3, 3);
        INDArray offsetTest = threeByThree.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        assertEquals(2, offsetTest.rows());
        INDArray offsetAssertion = Nd4j.create(new double[][]{
                {3, 3, 3}
                , {6, 6, 6}
        });
        INDArray offsetSub = offsetTest.subRowVector(row);
        assertEquals(offsetAssertion, offsetSub);

    }



    @Test
    public void testDimShuffle() {
        INDArray n = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray twoOneTwo = n.dimShuffle(new Object[]{0, 'x', 1}, new int[]{0, 1}, new boolean[]{false, false});
        assertTrue(Arrays.equals(new int[]{2, 1, 2}, twoOneTwo.shape()));

        INDArray reverse = n.dimShuffle(new Object[]{1, 'x', 0}, new int[]{1, 0}, new boolean[]{false, false});
        assertTrue(Arrays.equals(new int[]{2, 1, 2}, reverse.shape()));

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
        INDArray two = Nd4j.create(new float[]{2, 2, 2, 2});
        INDArray div = two.div(two);
        assertEquals(Nd4j.ones(4), div);

        INDArray half = Nd4j.create(new float[]{0.5f, 0.5f, 0.5f, 0.5f}, new int[]{2, 2});
        INDArray divi = Nd4j.create(new float[]{0.3f, 0.6f, 0.9f, 0.1f}, new int[]{2, 2});
        INDArray assertion = Nd4j.create(new float[]{1.6666666f, 0.8333333f, 0.5555556f, 5}, new int[]{2, 2});
        INDArray result = half.div(divi);
        assertEquals(assertion, result);
    }


    @Test
    public void testSigmoid() {
        INDArray n = Nd4j.create(new float[]{1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new float[]{0.73105858f, 0.88079708f, 0.95257413f, 0.98201379f});
        INDArray sigmoid = Transforms.sigmoid(n, false);
        assertEquals(assertion, sigmoid);
    }

    @Test
    public void testNeg() {
        INDArray n = Nd4j.create(new float[]{1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new float[]{-1, -2, -3, -4});
        INDArray neg = Transforms.neg(n);
        assertEquals(getFailureMessage(), assertion, neg);

    }

    @Test
    public void testNorm2Double() {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        INDArray n = Nd4j.create(new double[]{1, 2, 3, 4});
        double assertion = 5.47722557505;
        double norm3 = n.norm2Number().doubleValue();
        assertEquals(getFailureMessage(),assertion, norm3, 1e-1);

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray row1 = row.getRow(1);
        double norm2 = row1.norm2Number().doubleValue();
        double assertion2 = 5.0f;
        assertEquals(getFailureMessage(),assertion2, norm2, 1e-1);

    }


    @Test
    public void testNorm2() {
        INDArray n = Nd4j.create(new float[]{1, 2, 3, 4});
        float assertion = 5.47722557505f;
        float norm3 = n.norm2Number().floatValue();
        assertEquals(getFailureMessage(),assertion, norm3, 1e-1);


        INDArray row = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray row1 = row.getRow(1);
        float norm2 = row1.norm2Number().floatValue();
        float assertion2 = 5.0f;
        assertEquals(getFailureMessage(),assertion2, norm2, 1e-1);

    }



    @Test
    public void testCosineSim() {
        INDArray vec1 = Nd4j.create(new double[]{1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new double[]{1, 2, 3, 4});
        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(getFailureMessage(),1, sim, 1e-1);

        INDArray vec3 = Nd4j.create(new float[]{0.2f, 0.3f, 0.4f, 0.5f});
        INDArray vec4 = Nd4j.create(new float[]{0.6f, 0.7f, 0.8f, 0.9f});
        sim = Transforms.cosineSim(vec3, vec4);
        assertEquals(0.98, sim, 1e-1);

    }


    @Test
    public void testScal() {
        double assertion = 2;
        INDArray answer = Nd4j.create(new double[]{2, 4, 6, 8});
        INDArray scal = Nd4j.getBlasWrapper().scal(assertion, answer);
        assertEquals(getFailureMessage(),answer, scal);

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray row1 = row.getRow(1);
        double assertion2 = 5.0;
        INDArray answer2 = Nd4j.create(new double[]{15, 20});
        INDArray scal2 = Nd4j.getBlasWrapper().scal(assertion2, row1);
        assertEquals(getFailureMessage(), answer2, scal2);

    }

    @Test
    public void testExp() {
        INDArray n = Nd4j.create(new double[]{1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new double[]{2.71828183f, 7.3890561f, 20.08553692f, 54.59815003f});
        INDArray exped = Transforms.exp(n);
        assertEquals(assertion, exped);
    }


    @Test
    public void testSlices() {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        for (int i = 0; i < arr.slices(); i++) {
            assertEquals(2, arr.slice(i).slice(1).slices());
        }

    }


    @Test
    public void testScalar() {
        INDArray a = Nd4j.scalar(1.0);
        assertEquals(true, a.isScalar());

        INDArray n = Nd4j.create(new float[]{1.0f}, new int[]{1, 1});
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
        assertEquals(true, Shape.shapeEquals(new int[]{3}, testVector.shape()));

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
        INDArray arr = Nd4j.create(data, new int[]{1,4});
        assertEquals(true, arr.isRowVector());
        INDArray arr2 = Nd4j.create(data, new int[]{1, 4});
        assertEquals(true, arr2.isRowVector());

        INDArray columnVector = Nd4j.create(data, new int[]{4, 1});
        assertEquals(true, columnVector.isColumnVector());
    }


    @Test
    public void testColumns() {
        INDArray arr = Nd4j.create(new int[]{3, 2});
        INDArray column2 = arr.getColumn(0);
        //assertEquals(true, Shape.shapeEquals(new int[]{3, 1}, column2.shape()));
        INDArray column = Nd4j.create(new double[]{1, 2, 3}, new int[]{1,3});
        arr.putColumn(0, column);

        INDArray firstColumn = arr.getColumn(0);

        assertEquals(column, firstColumn);


        INDArray column1 = Nd4j.create(new double[]{4, 5, 6}, new int[]{1,3});
        arr.putColumn(1, column1);
        //assertEquals(true, Shape.shapeEquals(new int[]{3, 1}, arr.getColumn(1).shape()));
        INDArray testRow1 = arr.getColumn(1);
        assertEquals(column1, testRow1);


        INDArray evenArr = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray put = Nd4j.create(new double[]{5, 6}, new int[]{1,2});
        evenArr.putColumn(1, put);
        INDArray testColumn = evenArr.getColumn(1);
        assertEquals(put, testColumn);


        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        INDArray column23 = n.getColumn(0);
        INDArray column12 = Nd4j.create(new double[]{1, 3}, new int[]{1, 2});
        assertEquals(column23, column12);


        INDArray column0 = n.getColumn(1);
        INDArray column01 = Nd4j.create(new double[]{2, 4}, new int[]{1,2});
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
        assertEquals(true, Arrays.equals(new int[]{2, 2}, n.shape()));

        INDArray newRow = Nd4j.linspace(5, 6, 2);
        n.putRow(0, newRow);
        d.putRow(0, newRow);


        INDArray testRow = n.getRow(0);
        assertEquals(newRow.length(), testRow.length());
        assertEquals(true, Shape.shapeEquals(new int[]{1,2}, testRow.shape()));


        INDArray nLast = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        INDArray row = nLast.getRow(1);
        INDArray row1 = Nd4j.create(new double[]{3, 4}, new int[]{1,2});
        assertEquals(row, row1);


        INDArray arr = Nd4j.create(new int[]{3, 2});
        INDArray evenRow = Nd4j.create(new double[]{1, 2}, new int[]{1,2});
        arr.putRow(0, evenRow);
        INDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new int[]{1,2}, firstRow.shape()));
        INDArray testRowEven = arr.getRow(0);
        assertEquals(evenRow, testRowEven);


        INDArray row12 = Nd4j.create(new double[]{5, 6}, new int[]{1,2});
        arr.putRow(1, row12);
        assertEquals(true, Shape.shapeEquals(new int[]{1,2}, arr.getRow(0).shape()));
        INDArray testRow1 = arr.getRow(1);
        assertEquals(row12, testRow1);


        INDArray multiSliceTest = Nd4j.create(Nd4j.linspace(1, 16, 16).data(), new int[]{4, 2, 2});
        INDArray test = Nd4j.create(new double[]{5,6}, new int[]{1,2});
        INDArray test2 = Nd4j.create(new double[]{7,8}, new int[]{1,2});

        INDArray multiSliceRow1 = multiSliceTest.slice(1).getRow(0);
        INDArray multiSliceRow2 = multiSliceTest.slice(1).getRow(1);

        assertEquals(test, multiSliceRow1);
        assertEquals(test2,multiSliceRow2);



        INDArray threeByThree = Nd4j.create(3,3);
        INDArray threeByThreeRow1AndTwo = threeByThree.get(NDArrayIndex.interval(1,3),NDArrayIndex.all());
        threeByThreeRow1AndTwo.putRow(1,Nd4j.ones(3));
        assertEquals(Nd4j.ones(3),threeByThreeRow1AndTwo.getRow(1));

    }


    @Test
    public void testMulRowVector() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2, 2);
        arr.muliRowVector(Nd4j.linspace(1, 2, 2));
        INDArray assertion = Nd4j.create(new double[][]{
                {1, 4}, {3, 8}
        });

        assertEquals(assertion,arr);
    }



    @Test
    public void testSum() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        INDArray test = Nd4j.create(new float[]{3, 7, 11, 15}, new int[]{2, 2});
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
    public void testTADMMul(){
        Nd4j.getRandom().setSeed(12345);
        int[] shape = new int[]{4,5,7};
        INDArray arr = Nd4j.rand(shape);

        INDArray tad = arr.tensorAlongDimension(0, 1, 2);
        assertArrayEquals(tad.shape(), new int[]{7, 5});


        INDArray copy = Nd4j.zeros(7,5).assign(0.0);
        for( int i = 0; i < 7; i++) {
            for( int j = 0; j < 5; j++) {
                copy.putScalar(new int[]{i,j},tad.getDouble(i,j));
            }
        }


        assertTrue(tad.equals(copy));

        INDArray first = Nd4j.rand(new int[]{2, 7});
        INDArray mmul = first.mmul(tad);
        INDArray mmulCopy = first.mmul(copy);

        assertTrue(mmul.equals(mmulCopy));

        INDArray mmul2 = tad.mmul(first);
        INDArray mmul2copy = copy.mmul(first);
        assertTrue(mmul2.equals(mmul2copy));
    }

    @Test
    public void testTADMMulLeadingOne(){
        Nd4j.getRandom().setSeed(12345);
        int[] shape = new int[]{1,5,7};
        INDArray arr = Nd4j.rand(shape);

        INDArray tad = arr.tensorAlongDimension(0, 1, 2);
        boolean order = Shape.cOrFortranOrder(tad.shape(), tad.stride(), tad.elementStride());
        assertArrayEquals(tad.shape(),new int[]{7,5});


        INDArray copy = Nd4j.zeros(7,5);
        for( int i = 0; i < 7; i++ ){
            for( int j = 0; j < 5; j++ ){
                copy.putScalar(new int[]{i,j},tad.getDouble(i,j));
            }
        }

        assertTrue(tad.equals(copy));

        INDArray first = Nd4j.rand(new int[]{2, 7});
        INDArray mmul = first.mmul(tad);
        INDArray mmulCopy = first.mmul(copy);

        assertTrue(mmul.equals(mmulCopy));

        INDArray mmul2 = tad.mmul(first);
        INDArray mmul2copy = copy.mmul(first);
        assertTrue(mmul2.equals(mmul2copy));
    }


    @Test
    public void testSum2() {
        INDArray test = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray sum = test.sum(1);
        INDArray assertion = Nd4j.create(new float[]{3, 7});
        assertEquals(assertion, sum);
        INDArray sum0 = Nd4j.create(new double[]{4, 6});
        assertEquals(sum0, test.sum(0));
    }


    @Test
    public void testGetIntervalEdgeCase(){
        Nd4j.getRandom().setSeed(12345);

        int[] shape = {3,2,4};
        INDArray arr3d = Nd4j.rand(shape);

        INDArray get0 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 1));
        INDArray getPoint0 = arr3d.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(0));
        INDArray tad0 = arr3d.tensorAlongDimension(0,1,0);

        assertTrue(get0.equals(getPoint0)); //OK
        assertTrue(get0.equals(tad0));      //OK

        INDArray get1 = arr3d.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(1,2));
        INDArray getPoint1 = arr3d.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(1));
        INDArray tad1 = arr3d.tensorAlongDimension(1,1,0);

        assertTrue(getPoint1.equals(tad1)); //OK
        assertTrue(get1.equals(getPoint1)); //Fails
        assertTrue(get1.equals(tad1));

        INDArray get2 = arr3d.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(2,3));
        INDArray getPoint2 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(2));
        INDArray tad2 = arr3d.tensorAlongDimension(2,1,0);

        assertTrue(getPoint2.equals(tad2)); //OK
        assertTrue(get2.equals(getPoint2)); //Fails
        assertTrue(get2.equals(tad2));

        INDArray get3 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(3, 4));
        INDArray getPoint3 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(3));
        INDArray tad3 = arr3d.tensorAlongDimension(3, 1, 0);

        assertTrue(getPoint3.equals(tad3)); //OK
        assertTrue(get3.equals(getPoint3)); //Fails
        assertTrue(get3.equals(tad3));
    }


    @Test
    public void testGetIntervalEdgeCase2(){
        Nd4j.getRandom().setSeed(12345);

        int[] shape = {3,2,4};
        INDArray arr3d = Nd4j.rand(shape);

        for(int x = 0; x < 4; x++) {
            INDArray getInterval = arr3d.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(x,x + 1));   //3d
            INDArray getPoint = arr3d.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(x));             //2d
            INDArray tad = arr3d.tensorAlongDimension(x,1,0);                                                       //2d

            assertTrue(getPoint.equals(tad));   //OK, comparing 2d with 2d
            assertArrayEquals(getInterval.shape(),new int[]{3,2,1});
            for( int i = 0; i < 3; i++ ){
                for( int j = 0; j < 2; j++ ){
                    assertEquals(getInterval.getDouble(i,j,0) , getPoint.getDouble(i,j),1e-1);
                }
            }
        }
    }


    @Test
    public void testMmul() {
        DataBuffer data = Nd4j.linspace(1, 10, 10).data();
        INDArray n = Nd4j.create(data, new int[]{1, 10});
        INDArray transposed = n.transpose();
        assertEquals(true, n.isRowVector());
        assertEquals(true, transposed.isColumnVector());

        INDArray d = Nd4j.create(n.rows(), n.columns());
        d.setData(n.data());


        INDArray d3 = Nd4j.create(new double[]{1, 2}).reshape(2, 1);
        INDArray d4 = Nd4j.create(new double[]{3, 4});
        INDArray resultNDArray = d3.mmul(d4);
        INDArray result = Nd4j.create(new double[][]{{3, 4}, {6, 8}});
        assertEquals(result, resultNDArray);


        INDArray innerProduct = n.mmul(transposed);

        INDArray scalar = Nd4j.scalar(385);
        assertEquals(getFailureMessage(),scalar, innerProduct);

        INDArray outerProduct = transposed.mmul(n);
        assertEquals(getFailureMessage(),true, Shape.shapeEquals(new int[]{10, 10}, outerProduct.shape()));




        INDArray three = Nd4j.create(new double[]{3, 4}, new int[]{1,2});
        INDArray test = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2});
        INDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(getFailureMessage(),three, sliceRow);

        INDArray twoSix = Nd4j.create(new double[]{2, 6}, new int[]{2, 1});
        INDArray threeTwoSix = three.mmul(twoSix);

        INDArray sliceRowTwoSix = sliceRow.mmul(twoSix);

        assertEquals(threeTwoSix, sliceRowTwoSix);


        INDArray vectorVector = Nd4j.create(new double[]{
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225
        }, new int[]{16, 16});


        INDArray n1 = Nd4j.create(Nd4j.linspace(0, 15, 16).data(), new int[]{1,16});
        INDArray k1 = n1.transpose();

        INDArray testVectorVector = k1.mmul(n1);
        assertEquals(getFailureMessage(),vectorVector, testVectorVector);


    }





    @Test
    public void testRowsColumns() {
        DataBuffer data = Nd4j.linspace(1, 6, 6).data();
        INDArray rows = Nd4j.create(data, new int[]{2, 3});
        assertEquals(2, rows.rows());
        assertEquals(3, rows.columns());

        INDArray columnVector = Nd4j.create(data, new int[]{6, 1});
        assertEquals(6, columnVector.rows());
        assertEquals(1, columnVector.columns());
        INDArray rowVector = Nd4j.create(data, new int[]{1, 6});
        assertEquals(1, rowVector.rows());
        assertEquals(6, rowVector.columns());
    }


    @Test
    public void testTranspose() {
        INDArray n = Nd4j.create(Nd4j.ones(100).data(), new int[]{5, 5, 4});
        INDArray transpose = n.transpose();
        assertEquals(n.length(), transpose.length());
        assertEquals(true, Arrays.equals(new int[]{4, 5, 5}, transpose.shape()));

        INDArray rowVector = Nd4j.linspace(1, 10, 10);
        assertTrue(rowVector.isRowVector());
        INDArray columnVector = rowVector.transpose();
        assertTrue(columnVector.isColumnVector());


        INDArray linspaced = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray transposed = Nd4j.create(new float[]{1, 3, 2, 4}, new int[]{2, 2});
        INDArray linSpacedT = linspaced.transpose();
        assertEquals(transposed, linSpacedT);



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
        INDArray n = Nd4j.linspace(1,27,27).reshape(3, 3, 3);
        INDArray newSlice = Nd4j.zeros(3, 3);
        n.putSlice(0, newSlice);
        assertEquals(newSlice, n.slice(0));

        INDArray firstDimensionAs1 = newSlice.reshape(1, 3, 3);
        n.putSlice(0, firstDimensionAs1);


    }





    @Test
    public void testRowVectorMultipleIndices() {
        INDArray linear = Nd4j.create(1, 4);
        linear.putScalar(new int[]{0, 1}, 1);
        assertEquals(linear.getDouble(0, 1), 1, 1e-1);
    }




    @Test
    public void testEps() {
        INDArray ones = Nd4j.ones(5);
        double sum = Nd4j.getExecutioner().exec(new Eps(ones, ones, ones, ones.length())).z().sumNumber().doubleValue();
        assertEquals(0, sum, 1e-1);
    }


    @Test
    public void testLogDouble() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        INDArray log = Transforms.log(linspace);
        INDArray assertion = Nd4j.create(new double[]{0, 0.6931471805599453, 1.0986122886681098, 1.3862943611198906, 1.6094379124341005, 1.791759469228055});
        assertEquals(assertion, log);
    }

    @Test
    public void testIterator() {
        INDArray x = Nd4j.linspace(1,4,4).reshape(2, 2);
        INDArray repeated = x.repeat(new int[]{2});
        assertEquals(8, repeated.length());
        Iterator<Double> arrayIter = new INDArrayIterator(x);
        double[] vals = Nd4j.linspace(1,4,4).data().asDouble();
        for(int i = 0; i < vals.length; i++)
            assertEquals(vals[i],arrayIter.next().doubleValue(),1e-1);
    }

    @Test
    public void testTile() {
        INDArray x = Nd4j.linspace(1,4,4).reshape(2, 2);
        INDArray repeated = x.repeat(new int[]{2});
        assertEquals(8,repeated.length());
        INDArray repeatAlongDimension = x.repeat(1,new int[]{2});
        INDArray assertionRepeat = Nd4j.create(new double[][]{
                {1, 1, 2, 2},
                {3, 3, 4, 4}
        });
        assertArrayEquals(new int[]{2,4},assertionRepeat.shape());
        assertEquals(assertionRepeat,repeatAlongDimension);
        System.out.println(repeatAlongDimension);
        INDArray ret = Nd4j.create(new double[]{0, 1, 2});
        INDArray tile = Nd4j.tile(ret, 2, 2);
        INDArray assertion = Nd4j.create(new double[][]{
                {0, 1, 2, 0, 1, 2}
                , {0, 1, 2, 0, 1, 2}
        });
        assertEquals(assertion,tile);
    }

    @Test
    public void testNegativeOneReshape() {
        INDArray arr = Nd4j.create(new double[]{0, 1, 2});
        INDArray newShape = arr.reshape(-1, 3);
        assertEquals(newShape,arr);
    }


    @Test
    public void testSmallSum() {
        INDArray base = Nd4j.create(new double[]{5.843333333333335, 3.0540000000000007});
        base.addi(1e-12);
        INDArray assertion = Nd4j.create(new double[]{5.84333433, 3.054001});
        assertEquals(assertion, base);

    }


    @Test
    public void test2DArraySlice(){
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
        for( int i = 0; i < 7; i++) {
            INDArray slice = array2D.slice(i,1);
            assertTrue(Arrays.equals(slice.shape(), new int[]{5,1}));
        }

        for( int i = 0; i < 5; i++ ){
            INDArray slice = array2D.slice(i, 0);
            assertTrue(Arrays.equals(slice.shape(), new int[]{1,7}));
        }
    }

    @Test
    public void testTensorDot() {
        INDArray oneThroughSixty = Nd4j.arange(60).reshape(3, 4, 5);
        INDArray oneThroughTwentyFour = Nd4j.arange(24).reshape(4, 3, 2);
        INDArray result = Nd4j.tensorMmul(oneThroughSixty, oneThroughTwentyFour, new int[][]{{1, 0}, {0, 1}});
        assertArrayEquals(new int[]{5, 2}, result.shape());
        INDArray assertion = Nd4j.create(new double[][]{
                {   4400 ,  4730},
                {  4532 ,  4874},
                {  4664  , 5018},
                {  4796 ,  5162},
                {  4928 , 5306}
        });
        assertEquals(assertion, result);

        INDArray w = Nd4j.valueArrayOf(new int[]{2, 1, 2, 2}, 0.5);
        INDArray col = Nd4j.create(new double[]{
                1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3,
                3, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4,
                4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4
        }, new int[]{1, 1, 2, 2, 4, 4});

        INDArray test = Nd4j.tensorMmul(col, w, new int[][]{{1, 2, 3}, {1, 2, 3}});
        INDArray assertion2 = Nd4j.create(new double[]{3., 3., 3., 3., 3., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7., 3., 3.
                , 3., 3., 3., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7.}, new int[]{1, 4, 4, 2}, new int[]{16, 8, 2, 1}, 0, 'f');
        assertion2.setOrder('f');
        assertEquals(assertion2,test);
    }



    @Test
    public void testGetRow(){
        INDArray arr = Nd4j.ones(10, 4);
        for( int i=0; i<10; i++ ){
            INDArray row = arr.getRow(i);
            assertArrayEquals(row.shape(), new int[]{1, 4});
        }
    }


    @Test
    public void testGetPermuteReshapeSub(){
        Nd4j.getRandom().setSeed(12345);

        INDArray first = Nd4j.rand(new int[]{10, 4});

        //Reshape, as per RnnOutputLayer etc on labels
        INDArray orig3d = Nd4j.rand(new int[]{2, 4, 15});
        INDArray subset3d = orig3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(5, 10));
        INDArray permuted = subset3d.permute(0, 2, 1);
        int[] newShape = { subset3d.size(0) * subset3d.size(2),subset3d.size(1)};
        INDArray second = permuted.reshape(newShape);

        assertArrayEquals(first.shape(), second.shape());
        assertEquals(first.length(), second.length());
        assertArrayEquals(first.stride(), second.stride());

        first.sub(second);  //Exception
    }


    @Test
    public void testPutAtIntervalIndexWithStride(){
        INDArray n1 = Nd4j.create(3, 3).assign(0.0);
        INDArrayIndex[] indices = {NDArrayIndex.interval(0,2,3),NDArrayIndex.all()};
        n1.put(indices, 1);
        INDArray expected = Nd4j.create(new double[][]{{1d,1d,1d},{0d,0d,0d},{1d,1d,1d}});
        assertEquals(expected, n1);
    }

    @Test
    public void testMMulMatrixTimesColVector(){
        //[1 1 1 1 1; 10 10 10 10 10; 100 100 100 100 100] x [1; 1; 1; 1; 1] = [5; 50; 500]
        INDArray matrix = Nd4j.ones(3, 5);
        matrix.getRow(1).muli(10);
        matrix.getRow(2).muli(100);

        INDArray colVector = Nd4j.ones(5, 1);
        INDArray out = matrix.mmul(colVector);

        INDArray expected = Nd4j.create(new double[]{5, 50, 500}, new int[]{3, 1});
        assertEquals(expected, out);
    }


    @Test
    public void testMMulMixedOrder(){
        INDArray first = Nd4j.ones(5, 2);
        INDArray second = Nd4j.ones(2, 3);
        INDArray out = first.mmul(second);
        assertArrayEquals(out.shape(),new int[]{5,3});
        assertTrue(out.equals(Nd4j.ones(5,3).muli(2)));
        //Above: OK

        INDArray firstC = Nd4j.create(new int[]{5, 2}, 'c');
        INDArray secondF = Nd4j.create(new int[]{2, 3}, 'f');
        for(int i=0; i<firstC.length(); i++ ) firstC.putScalar(i, 1.0);
        for(int i=0; i<secondF.length(); i++ ) secondF.putScalar(i, 1.0);
        assertTrue(first.equals(firstC));
        assertTrue(second.equals(secondF));

        INDArray outCF = firstC.mmul(secondF);
        assertArrayEquals(outCF.shape(), new int[]{5, 3});
        assertEquals(outCF, Nd4j.ones(5, 3).muli(2));
    }


    @Test
    public void testFTimesCAddiRow() {

        INDArray arrF = Nd4j.create(2,3,'f').assign(1.0);
        INDArray arrC = Nd4j.create(2,3,'c').assign(1.0);
        INDArray arr2 = Nd4j.create(new int[]{3, 4},'c').assign(1.0);

        INDArray mmulC = arrC.mmul(arr2);   //[2,4] with elements 3.0
        INDArray mmulF = arrF.mmul(arr2);   //[2,4] with elements 3.0
        assertArrayEquals(mmulC.shape(),new int[]{2,4});
        assertArrayEquals(mmulF.shape(), new int[]{2, 4});
        assertTrue(arrC.equals(arrF));

        INDArray row = Nd4j.zeros(1,4).assign(0.0).addi(0.5);
        mmulC.addiRowVector(row);   //OK
        mmulF.addiRowVector(row);   //Exception

        assertTrue(mmulC.equals(mmulF));

        for( int i = 0; i < mmulC.length(); i++ )
            assertEquals(mmulC.getDouble(i),3.5, 1e-1);    //OK
        for( int i = 0; i < mmulF.length(); i++)
            assertEquals(mmulF.getDouble(i),3.5,1e-1);    //Exception
    }




    @Test
    public void testMmulGet(){
        Nd4j.getRandom().setSeed(12345L);
        INDArray elevenByTwo = Nd4j.rand(new int[]{11, 2});
        INDArray twoByEight = Nd4j.rand(new int[]{2, 8});

        INDArray view = twoByEight.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        INDArray viewCopy = view.dup();
        assertTrue(view.equals(viewCopy));

        INDArray mmul1 = elevenByTwo.mmul(view);
        INDArray mmul2 = elevenByTwo.mmul(viewCopy);

        assertTrue(mmul1.equals(mmul2));
    }


    @Test
    public void testMMulRowColVectorMixedOrder(){
        INDArray colVec = Nd4j.ones(5,1);
        INDArray rowVec = Nd4j.ones(1,3);
        INDArray out = colVec.mmul(rowVec);
        assertArrayEquals(out.shape(),new int[]{5,3});
        assertTrue(out.equals(Nd4j.ones(5, 3)));
        //Above: OK

        INDArray colVectorC = Nd4j.create(new int[]{5, 1}, 'c');
        INDArray rowVectorF = Nd4j.create(new int[]{1, 3}, 'f');
        for(int i = 0; i < colVectorC.length(); i++)
            colVectorC.putScalar(i, 1.0);
        for (int i = 0; i < rowVectorF.length(); i++)
            rowVectorF.putScalar(i, 1.0);
        assertTrue(colVec.equals(colVectorC));
        assertTrue(rowVec.equals(rowVectorF));

        INDArray outCF = colVectorC.mmul(rowVectorF);
        assertArrayEquals(outCF.shape(),new int[]{5,3});
        assertEquals(outCF, Nd4j.ones(5, 3));
    }

    @Test
    public void testMMulFTimesC() {
        int nRows = 3;
        int nCols = 3;
        java.util.Random r = new java.util.Random(12345);

        INDArray arrC = Nd4j.create(new int[]{nRows, nCols}, 'c');
        INDArray arrF = Nd4j.create(new int[]{nRows, nCols}, 'f');
        INDArray arrC2 = Nd4j.create(new int[]{nRows, nCols}, 'c');
        for( int i = 0; i< nRows; i++ ){
            for( int j = 0; j< nCols; j++ ){
                double rv = r.nextDouble();
                arrC.putScalar(new int[]{i,j}, rv);
                arrF.putScalar(new int[]{i,j}, rv);
                arrC2.putScalar(new int[]{i,j}, r.nextDouble());
            }
        }
        assertTrue(arrF.equals(arrC));

        INDArray fTimesC = arrF.mmul(arrC2);
        INDArray cTimesC = arrC.mmul(arrC2);

        assertEquals(fTimesC, cTimesC);
    }

    @Test
    public void testMMulColVectorRowVectorMixedOrder(){
        INDArray colVec = Nd4j.ones(5, 1);
        INDArray rowVec = Nd4j.ones(1, 5);
        INDArray out = rowVec.mmul(colVec);
        assertArrayEquals(out.shape(), new int[]{1, 1});
        assertTrue(out.equals(Nd4j.ones(1, 1).muli(5)));

        INDArray colVectorC = Nd4j.create(new int[]{5, 1}, 'c');
        INDArray rowVectorF = Nd4j.create(new int[]{1, 5}, 'f');
        for(int i=0; i<colVectorC.length(); i++ ) colVectorC.putScalar(i, 1.0);
        for(int i=0; i<rowVectorF.length(); i++ ) rowVectorF.putScalar(i, 1.0);
        assertTrue(colVec.equals(colVectorC));
        assertTrue(rowVec.equals(rowVectorF));

        INDArray outCF = rowVectorF.mmul(colVectorC);
        assertArrayEquals(outCF.shape(), new int[]{1, 1});
        assertTrue(outCF.equals(Nd4j.ones(1, 1).muli(5)));
    }

    @Test
    public void testPermute() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 20, 20).data(), new int[]{5, 4});
        INDArray transpose = n.transpose();
        INDArray permute = n.permute(1, 0);
        assertEquals(permute, transpose);
        assertEquals(transpose.length(), permute.length(), 1e-1);


        INDArray toPermute = Nd4j.create(Nd4j.linspace(0, 7, 8).data(), new int[]{2, 2, 2});
        INDArray permuted = toPermute.permute(2, 1, 0);
        INDArray assertion = Nd4j.create(new float[]{0, 4, 2, 6, 1, 5, 3, 7}, new int[]{2, 2, 2});
        assertEquals(permuted, assertion);

    }



    @Test
    public void testSwapAxes() {
        INDArray n = Nd4j.create(Nd4j.linspace(0, 7, 8).data(), new int[]{2, 2, 2});
        INDArray assertion = n.permute(2, 1, 0);
        INDArray permuteTranspose = assertion.slice(1).slice(1);
        INDArray validate = Nd4j.create(new float[]{0, 4, 2, 6, 1, 5, 3, 7}, new int[]{2, 2, 2});
        assertEquals(validate, assertion);

        INDArray thirty = Nd4j.linspace(1, 30, 30).reshape(3, 5, 2);
        INDArray swapped = thirty.swapAxes(2, 1);
        INDArray slice = swapped.slice(0).slice(0);
        INDArray assertion2 = Nd4j.create(new double[]{1, 3, 5, 7, 9});
        assertEquals(assertion2, slice);


    }


    @Test
    public void testMuliRowVector(){
        INDArray arrC = Nd4j.linspace(1,6,6).reshape('c',3,2);
        INDArray arrF = Nd4j.create(new int[]{3,2},'f').assign(arrC);

        INDArray temp = Nd4j.create(new int[]{2,11},'c');
        INDArray vec = temp.get(NDArrayIndex.all(), NDArrayIndex.interval(9,10)).transpose();
        vec.assign(Nd4j.linspace(1,2,2));

        //Passes if we do one of these...
//        vec = vec.dup('c');
//        vec = vec.dup('f');

        INDArray outC = arrC.muliRowVector(vec);
        INDArray outF = arrF.muliRowVector(vec);

        double[][] expD = new double[][]{{1,4},{3,8},{5,12}};
        INDArray exp = Nd4j.create(expD);

        assertEquals(exp, outC);
        assertEquals(exp, outF);
    }

    @Test
    public void testSliceConstructor() throws Exception {
        List<INDArray> testList = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            testList.add(Nd4j.scalar(i + 1));

        INDArray test = Nd4j.create(testList, new int[]{1, testList.size()}).reshape(1, 5);
        INDArray expected = Nd4j.create(new float[]{1, 2, 3, 4, 5}, new int[]{1, 5});
        assertEquals(expected, test);
    }



    @Test
    public void testStdev0(){
        double[][] ind = {{5.1, 3.5, 1.4}, {4.9, 3.0, 1.4}, {4.7, 3.2, 1.3}};
        INDArray in = Nd4j.create(ind);
        INDArray stdev = in.std(0);
        INDArray exp = Nd4j.create(new double[]{0.20000005,0.24527183,0.047140464});

        assertEquals(exp,stdev);
    }

    @Test
    public void testStdev1(){
        double[][] ind = {{5.1, 3.5, 1.4}, {4.9, 3.0, 1.4}, {4.7, 3.2, 1.3}};
        INDArray in = Nd4j.create(ind);
        INDArray stdev = in.std(1);
        INDArray exp = Nd4j.create(new double[]{1.8552212,1.7519685,1.7035841});
        assertEquals(exp,stdev);
    }


    @Test
    public void testSignXZ(){
        double[] d =   {1.0, -1.1,  1.2,  1.3, -1.4, -1.5, 1.6, -1.7, -1.8, -1.9, -1.01, -1.011};
        double[] e =   {1.0, -1.0,  1.0,  1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

        INDArray arrF = Nd4j.create(d,new int[]{4,3},'f');
        INDArray arrC = Nd4j.create(new int[]{4,3},'c').assign(arrF);

        INDArray exp = Nd4j.create(e, new int[]{4,3}, 'f');

        //First: do op with just x (inplace)
        INDArray arrFCopy = arrF.dup('f');
        INDArray arrCCopy = arrC.dup('c');
        Nd4j.getExecutioner().exec(new Sign(arrFCopy));
        Nd4j.getExecutioner().exec(new Sign(arrCCopy));
        assertEquals(exp, arrFCopy);
        assertEquals(exp, arrCCopy);

        //Second: do op with both x and z:
        INDArray zOutFC = Nd4j.create(new int[]{4,3},'c');
        INDArray zOutFF = Nd4j.create(new int[]{4,3},'f');
        INDArray zOutCC = Nd4j.create(new int[]{4,3},'c');
        INDArray zOutCF = Nd4j.create(new int[]{4,3},'f');
        Nd4j.getExecutioner().exec(new Sign(arrF, zOutFC));
        Nd4j.getExecutioner().exec(new Sign(arrF, zOutFF));
        Nd4j.getExecutioner().exec(new Sign(arrC, zOutCC));
        Nd4j.getExecutioner().exec(new Sign(arrC, zOutCF));

        assertEquals(exp, zOutFC);  //fails
        assertEquals(exp, zOutFF);  //pass
        assertEquals(exp, zOutCC);  //pass
        assertEquals(exp, zOutCF);  //fails
    }

    @Test
    public void testTanhXZ(){
        INDArray arrC = Nd4j.linspace(-6,6,12).reshape('c',4,3);
        INDArray arrF = Nd4j.create(new int[]{4,3},'f').assign(arrC);
        double[] d = arrC.data().asDouble();
        double[] e = new double[d.length];
        for(int i=0; i<e.length; i++ ) e[i] = Math.tanh(d[i]);

        INDArray exp = Nd4j.create(e, new int[]{4,3}, 'c');

        //First: do op with just x (inplace)
        INDArray arrFCopy = arrF.dup('f');
        INDArray arrCCopy = arrF.dup('c');
        Nd4j.getExecutioner().exec(new Tanh(arrFCopy));
        Nd4j.getExecutioner().exec(new Tanh(arrCCopy));
        assertEquals(exp, arrFCopy);
        assertEquals(exp, arrCCopy);

        //Second: do op with both x and z:
        INDArray zOutFC = Nd4j.create(new int[]{4,3},'c');
        INDArray zOutFF = Nd4j.create(new int[]{4,3},'f');
        INDArray zOutCC = Nd4j.create(new int[]{4,3},'c');
        INDArray zOutCF = Nd4j.create(new int[]{4,3},'f');
        Nd4j.getExecutioner().exec(new Tanh(arrF, zOutFC));
        Nd4j.getExecutioner().exec(new Tanh(arrF, zOutFF));
        Nd4j.getExecutioner().exec(new Tanh(arrC, zOutCC));
        Nd4j.getExecutioner().exec(new Tanh(arrC, zOutCF));

        assertEquals(exp, zOutFC);  //fails
        assertEquals(exp, zOutFF);  //pass
        assertEquals(exp, zOutCC);  //pass
        assertEquals(exp, zOutCF);  //fails
    }


    @Test
    public void testBroadcastDiv() {
        INDArray num = Nd4j.create(new double[] {
                1.00,1.00,1.00,1.00,2.00,2.00,2.00,2.00,1.00,1.00,1.00,1.00,2.00,2.00,2.00,2.00,
                -1.00,-1.00,-1.00,-1.00,-2.00,-2.00,-2.00,-2.00,-1.00,-1.00,-1.00,-1.00,-2.00,-2.00,-2.00,-2.00
        }).reshape(2,16);

        INDArray denom = Nd4j.create(new double[] {
                1.00, 1.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 1.00, 1.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00
        });

        INDArray expected = Nd4j.create(new double[] {
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        }, new int[]{2, 16});

        INDArray actual = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(num, denom, num.dup(),-1));
        assertEquals(expected, actual);
    }


    @Test
    public void testBroadcastMult() {
        INDArray num = Nd4j.create(new double[] {
                1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,
                -1.00,-2.00,-3.00,-4.00,-5.00,-6.00,-7.00,-8.00
        }).reshape(2,8);

        INDArray denom = Nd4j.create(new double[] {
                1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00
        });

        INDArray expected = Nd4j.create(new double[] {
                1,4,9,16,25,36,49,64,
                -1,-4,-9,-16,-25,-36,-49,-64
        }, new int[]{2, 8});

        INDArray actual = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(num, denom, num.dup(),-1));
        assertEquals(expected, actual);
    }

    @Test
    public void testBroadcastSub() {
        INDArray num = Nd4j.create(new double[] {
                1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,
                -1.00,-2.00,-3.00,-4.00,-5.00,-6.00,-7.00,-8.00
        }).reshape(2,8);

        INDArray denom = Nd4j.create(new double[] {
                1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00
        });

        INDArray expected = Nd4j.create(new double[] {
                0,0,0,0,0,0,0,0,
                -2,-4,-6,-8,-10,-12,-14,-16
        }, new int[]{2, 8});

        INDArray actual = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(num, denom, num.dup(),-1));
        assertEquals(expected, actual);
    }

    @Test
    public void testBroadcastAdd() {
        INDArray num = Nd4j.create(new double[] {
                1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,
                -1.00,-2.00,-3.00,-4.00,-5.00,-6.00,-7.00,-8.00
        }).reshape(2,8);

        INDArray denom = Nd4j.create(new double[] {
                1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00
        });

        INDArray expected = Nd4j.create(new double[] {
                2,4,6,8,10,12,14,16,
                0,0,0,0,0,0,0,0,
        }, new int[]{2, 8});
        INDArray dup = num.dup();
        INDArray actual = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(num, denom, dup,-1));
        assertEquals(expected, actual);
    }


    @Test
    public void testDimension() {
        INDArray test = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        //row
        INDArray slice0 = test.slice(0, 1);
        INDArray slice02 = test.slice(1, 1);

        INDArray assertSlice0 = Nd4j.create(new float[]{1, 3});
        INDArray assertSlice02 = Nd4j.create(new float[]{2, 4});
        assertEquals(assertSlice0, slice0);
        assertEquals(assertSlice02, slice02);

        //column
        INDArray assertSlice1 = Nd4j.create(new float[]{1, 2});
        INDArray assertSlice12 = Nd4j.create(new float[]{3, 4});


        INDArray slice1 = test.slice(0, 0);
        INDArray slice12 = test.slice(1, 0);


        assertEquals(assertSlice1, slice1);
        assertEquals(assertSlice12, slice12);


        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        INDArray secondSliceFirstDimension = arr.slice(1, 1);
        assertEquals(secondSliceFirstDimension, secondSliceFirstDimension);


    }






    @Test
    public void testReshape() {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        INDArray reshaped = arr.reshape(2, 3, 4);
        assertEquals(arr.length(), reshaped.length());
        assertEquals(true, Arrays.equals(new int[]{4, 3, 2}, arr.shape()));
        assertEquals(true, Arrays.equals(new int[]{2, 3, 4}, reshaped.shape()));

    }






    @Test
    public void testDot() {
        INDArray vec1 = Nd4j.create(new float[]{1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new float[]{1, 2, 3, 4});
        assertEquals(30, Nd4j.getBlasWrapper().dot(vec1, vec2), 1e-1);

        INDArray matrix = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row = matrix.getRow(1);
        assertEquals(25, Nd4j.getBlasWrapper().dot(row, row), 1e-1);

    }


    @Test
    public void testIdentity() {
        INDArray eye = Nd4j.eye(5);
        assertTrue(Arrays.equals(new int[]{5, 5}, eye.shape()));
        eye = Nd4j.eye(5);
        assertTrue(Arrays.equals(new int[]{5, 5}, eye.shape()));


    }

    @Test
    public void testTemp(){
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(new int[]{2, 2, 2});
        System.out.println("In:\n" + in);
        INDArray permuted = in.permute(0, 2, 1);    //Permute, so we get correct order after reshaping
        INDArray out = permuted.reshape(4, 2);
        System.out.println("Out:\n" + out);

        int countZero = 0;
        for( int i = 0; i < 8; i++)
            if(out.getDouble(i) == 0.0)
                countZero++;
        assertEquals(countZero, 0);
    }


    @Test
    public void testMeans() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray mean1 = a.mean(1);
        assertEquals(getFailureMessage(), Nd4j.create(new double[]{1.5, 3.5}), mean1);
        assertEquals(getFailureMessage(), Nd4j.create(new double[]{2, 3}), a.mean(0));
        assertEquals(getFailureMessage(),2.5, Nd4j.linspace(1, 4, 4).meanNumber().doubleValue(), 1e-1);
        assertEquals(getFailureMessage(), 2.5, a.meanNumber().doubleValue(), 1e-1);

    }


    @Test
    public void testSums() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        assertEquals(getFailureMessage(),Nd4j.create(new float[]{3, 7}), a.sum(1));
        assertEquals(getFailureMessage(), Nd4j.create(new float[]{4, 6}), a.sum(0));
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
        assertTrue(Arrays.equals(new int[]{5, 2, 2}, concat.shape()));

        INDArray columnConcat = Nd4j.linspace(1,6, 6).reshape(2, 3);
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
        INDArray input = Nd4j.create(new double[]{ -0.75, 0.58, 0.42, 1.03, -0.61, 0.19, -0.37, -0.40, -1.42, -0.04}).transpose();
        System.out.println("Input transpose " + Shape.shapeToString(input.shapeInfo()));
        INDArray output = Nd4j.create(10,1);
        System.out.println("Element wise stride of output " + output.elementWiseStride());
        Nd4j.getExecutioner().exec(new SoftMax(input, output));
    }

    @Test
    public void testAssignOffset() {
        INDArray arr = Nd4j.ones(5, 5);
        INDArray row = arr.slice(1);
        row.assign(1);
        assertEquals(Nd4j.ones(5),row);
    }

    @Test
    public void testAddScalar() {
        INDArray div = Nd4j.valueArrayOf(new int[]{1, 4}, 4);
        INDArray rdiv = div.add(1);
        INDArray answer = Nd4j.valueArrayOf(new int[]{1, 4}, 5);
        assertEquals(answer, rdiv);
    }

    @Test
    public void testRdivScalar() {
        INDArray div = Nd4j.valueArrayOf(2, 4);
        INDArray rdiv = div.rdiv(1);
        INDArray answer = Nd4j.valueArrayOf(new int[]{1, 4}, 0.25);
        assertEquals(rdiv, answer);
    }

    @Test
    public void testRDivi() {
        INDArray n2 = Nd4j.valueArrayOf(new int[]{1, 2}, 4);
        INDArray n2Assertion = Nd4j.valueArrayOf(new int[]{1, 2}, 0.5);
        INDArray nRsubi = n2.rdivi(2);
        assertEquals(n2Assertion, nRsubi);
    }






    @Test
    public void testElementWiseAdd() {
        INDArray linspace = Nd4j.linspace(1,4,4).reshape(2, 2);
        INDArray linspace2 = linspace.dup();
        INDArray assertion = Nd4j.create(new double[][]{{2, 4}, {6, 8}});
        linspace.addi(linspace2);
        assertEquals(assertion, linspace);
    }

    @Test
    public void testSquareMatrix() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        INDArray eightFirstTest = n.vectorAlongDimension(0, 2);
        INDArray eightFirstAssertion = Nd4j.create(new float[]{1, 2}, new int[]{1, 2});
        assertEquals(eightFirstAssertion, eightFirstTest);

        INDArray eightFirstTestSecond = n.vectorAlongDimension(1, 2);
        INDArray eightFirstTestSecondAssertion = Nd4j.create(new float[]{3, 4});
        assertEquals(eightFirstTestSecondAssertion, eightFirstTestSecond);

    }

    @Test
    public void testNumVectorsAlongDimension() {
        INDArray arr = Nd4j.linspace(1, 24, 24).reshape(4, 3, 2);
        assertEquals(12, arr.vectorsAlongDimension(2));
    }


    @Test
    public void testNewAxis() {
        INDArray arr = Nd4j.linspace(1, 12, 12).reshape(3, 2, 2);
        INDArray get = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.newAxis(), NDArrayIndex.newAxis());
        int[] shapeAssertion = {3, 2, 1, 1, 2};
        assertArrayEquals(shapeAssertion, get.shape());
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
        INDArray broadCasted3 = fourD.broadcast(1, 1, 36, 36);
        assertTrue(Arrays.equals(new int[]{1, 2, 36, 36}, broadCasted3.shape()));
    }


    @Test
    public void testPutRowGetRowOrdering() {
        INDArray row1 = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray put = Nd4j.create(new double[]{5, 6});
        row1.putRow(1, put);


        INDArray row1Fortran = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray putFortran = Nd4j.create(new double[]{5, 6});
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
    public void testNdArrayCreation(){
        double delta = 1e-1;
        INDArray n1 = Nd4j.create(new double[]{0d, 1d, 2d, 3d}, new int[]{2, 2}, 'c');
        INDArray lv = n1.linearView();
        assertEquals(0d, lv.getDouble(0), delta);
        assertEquals(1d,lv.getDouble(1),delta);
        assertEquals(2d,lv.getDouble(2),delta);
        assertEquals(3d, lv.getDouble(3), delta);
    }

    @Test
    public void testToFlattenedWithOrder(){

        int[] firstShape = {10,3};
        int firstLen = ArrayUtil.prod(firstShape);
        int[] secondShape = {2,7};
        int secondLen = ArrayUtil.prod(secondShape);
        int[] thirdShape = {3,3};
        int thirdLen = ArrayUtil.prod(thirdShape);
        INDArray firstC = Nd4j.linspace(1,firstLen,firstLen).reshape('c',firstShape);
        INDArray firstF = Nd4j.create(firstShape,'f').assign(firstC);
        INDArray secondC = Nd4j.linspace(1,secondLen,secondLen).reshape('c',secondShape);
        INDArray secondF = Nd4j.create(secondShape,'f').assign(secondC);
        INDArray thirdC = Nd4j.linspace(1,thirdLen,thirdLen).reshape('c',thirdShape);
        INDArray thirdF = Nd4j.create(thirdShape,'f').assign(thirdC);


        assertEquals(firstC,firstF);
        assertEquals(secondC,secondF);
        assertEquals(thirdC,thirdF);

        INDArray cc = Nd4j.toFlattened('c',firstC,secondC,thirdC);
        INDArray cf = Nd4j.toFlattened('c',firstF,secondF,thirdF);
        assertEquals(cc,cf);

        INDArray cmixed = Nd4j.toFlattened('c',firstC,secondF,thirdF);
        assertEquals(cc,cmixed);

        INDArray fc = Nd4j.toFlattened('f',firstC,secondC,thirdC);
        assertNotEquals(cc,fc);

        INDArray ff = Nd4j.toFlattened('f',firstF,secondF,thirdF);
        assertEquals(fc,ff);

        INDArray fmixed = Nd4j.toFlattened('f',firstC,secondF,thirdF);
        assertEquals(fc,fmixed);
    }


    @Test
    public void testLeakyRelu(){
        INDArray arr = Nd4j.linspace(-1,1,10);
        double[] expected = new double[10];
        for( int i = 0; i < 10; i++ ){
            double in = arr.getDouble(i);
            expected[i] = (in <= 0.0 ? 0.01 * in : in);
        }

        INDArray out = Nd4j.getExecutioner().execAndReturn(new LeakyReLU(arr,0.01));

        INDArray exp = Nd4j.create(expected);
        assertEquals(exp,out);
    }

    @Test
    public void testSoftmaxRow() {
        for( int i = 0; i < 20; i++ ){
            INDArray arr1 = Nd4j.zeros(100);
            Nd4j.getExecutioner().execAndReturn(new SoftMax(arr1));
            System.out.println(Arrays.toString(arr1.data().asFloat()));
        }
    }

    @Test
    public void testLeakyRelu2(){
        INDArray arr = Nd4j.linspace(-1,1,10);
        double[] expected = new double[10];
        for( int i = 0; i < 10; i++) {
            double in = arr.getDouble(i);
            expected[i] = (in <= 0.0 ? 0.01 * in : in);
        }

        INDArray out = Nd4j.getExecutioner().execAndReturn(new LeakyReLU(arr,0.01));

        System.out.println("Expected: " + Arrays.toString(expected));
        System.out.println("Actual:   " + Arrays.toString(out.data().asDouble()));

        INDArray exp = Nd4j.create(expected);
        assertEquals(exp,out);
    }

    @Test
    public void testDupAndDupWithOrder() {
        List<Pair<INDArray,String>> testInputs = NDArrayCreationUtil.getAllTestMatricesWithShape(ordering(),4, 5, 123);
        for(Pair<INDArray,String> pair : testInputs) {

            String msg = pair.getSecond();
            INDArray in = pair.getFirst();
            INDArray dup = in.dup();
            INDArray dupc = in.dup('c');
            INDArray dupf = in.dup('f');

            assertEquals(dup.ordering(),ordering());
            assertEquals(dupc.ordering(),'c');
            assertEquals(dupf.ordering(),'f');
            assertEquals(msg,in,dupc);
            assertEquals(msg,in,dupf);
        }
    }

    @Test
    public void testToOffsetZeroCopy(){
        List<Pair<INDArray,String>> testInputs = NDArrayCreationUtil.getAllTestMatricesWithShape(ordering(),4, 5, 123);

        for(int i = 0; i < testInputs.size(); i++) {
            Pair<INDArray,String> pair = testInputs.get(i);
            String msg = pair.getSecond();
            msg += "Failed on " + i;
            INDArray in = pair.getFirst();
            INDArray dup = Shape.toOffsetZeroCopy(in,ordering());
            INDArray dupc = Shape.toOffsetZeroCopy(in, 'c');
            INDArray dupf = Shape.toOffsetZeroCopy(in, 'f');
            INDArray dupany = Shape.toOffsetZeroCopyAnyOrder(in);

            assertEquals(msg,in,dup);
            assertEquals(msg,in,dupc);
            assertEquals(msg,in,dupf);
            assertEquals(msg,dupc.ordering(),'c');
            assertEquals(msg,dupf.ordering(),'f');
            assertEquals(msg,in,dupany);

            assertEquals(dup.offset(),0);
            assertEquals(dupc.offset(),0);
            assertEquals(dupf.offset(),0);
            assertEquals(dupany.offset(),0);
            assertEquals(dup.length(),dup.data().length());
            assertEquals(dupc.length(),dupc.data().length());
            assertEquals(dupf.length(),dupf.data().length());
            assertEquals(dupany.length(),dupany.data().length());
        }
    }

    @Test
    public void testTensorStats() {
        List<Pair<INDArray,String>> testInputs = NDArrayCreationUtil.getAllTestMatricesWithShape(9, 13, 123);

        for(Pair<INDArray,String> pair : testInputs) {
            INDArray arr = pair.getFirst();
            String msg = pair.getSecond();

            int nTAD0 = arr.tensorssAlongDimension(0);
            int nTAD1 = arr.tensorssAlongDimension(1);

            OpExecutionerUtil.Tensor1DStats t0 = OpExecutionerUtil.get1DTensorStats(arr, 0);
            OpExecutionerUtil.Tensor1DStats t1 = OpExecutionerUtil.get1DTensorStats(arr, 1);

            assertEquals(nTAD0,t0.getNumTensors());
            assertEquals(nTAD1, t1.getNumTensors());

            INDArray tFirst0 = arr.tensorAlongDimension(0,0);
            INDArray tSecond0 = arr.tensorAlongDimension(1,0);

            INDArray tFirst1 = arr.tensorAlongDimension(0,1);
            INDArray tSecond1 = arr.tensorAlongDimension(1,1);

            assertEquals(tFirst0.offset(),t0.getFirstTensorOffset());
            assertEquals(tFirst1.offset(),t1.getFirstTensorOffset());
            int separation0 = tSecond0.offset()-tFirst0.offset();
            int separation1 = tSecond1.offset()-tFirst1.offset();
            assertEquals(separation0,t0.getTensorStartSeparation());
            assertEquals(separation1,t1.getTensorStartSeparation());

            for( int i = 0; i < nTAD0; i++) {
                INDArray tad0 = arr.tensorAlongDimension(i,0);
                assertEquals(tad0.length(), t0.getTensorLength());
                assertEquals(tad0.elementWiseStride(),t0.getElementWiseStride());

                int offset = tad0.offset();
                int calcOffset = t0.getFirstTensorOffset() + i*t0.getTensorStartSeparation();
                assertEquals(offset,calcOffset);
            }

            for( int i = 0; i < nTAD1; i++) {
                INDArray tad1 = arr.tensorAlongDimension(i,1);
                assertEquals(tad1.length(), t1.getTensorLength());
                assertEquals(tad1.elementWiseStride(),t1.getElementWiseStride());

                int offset = tad1.offset();
                int calcOffset = t1.getFirstTensorOffset() + i*t1.getTensorStartSeparation();
                assertEquals(offset,calcOffset);
            }
        }
    }




    @Test
    @Ignore
    public void largeInstantiation() {
        Nd4j.ones((1024 * 1024 * 511) + (1024*1024 - 1)); // Still works; this can even be called as often as I want, allowing me even to spill over on disk
        Nd4j.ones((1024 * 1024 * 511) + (1024*1024)); // Crashes
    }

    @Test
    public void testAssignNumber() {
        int nRows = 10;
        int nCols = 20;
        INDArray in = Nd4j.linspace(1,nRows * nCols,nRows * nCols).reshape('c',new int[]{nRows,nCols});

        INDArray subset1 = in.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(0, nCols / 2));
        subset1.assign(1.0);

        INDArray subset2 = in.get(NDArrayIndex.interval(5,8), NDArrayIndex.interval(nCols / 2,nCols));
        subset2.assign(2.0);
        INDArray assertion = Nd4j.create(
                new double[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                        21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
                        40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0,
                        60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                        80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
                        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                        121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0,
                        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 141.0, 142.0,
                        143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 161.0, 162.0, 163.0, 164.0,
                        165.0, 166.0, 167.0, 168.0, 169.0,
                        170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, 180.0, 181.0, 182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0,
                        190.0, 191.0, 192.0, 193.0,
                        194.0, 195.0, 196.0, 197.0, 198.0, 199.0, 200.0}, in.shape(), 0, 'c');
        assertEquals(assertion, in);
    }


    @Test
    public void testSumDifferentOrdersSquareMatrix() {
        INDArray arrc = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray arrf = Nd4j.create(new int[]{2,2},'f').assign(arrc);

        INDArray cSum = arrc.sum(0);
        INDArray fSum = arrf.sum(0);
        assertEquals(arrc,arrf);
        assertEquals(cSum,fSum);  //Expect: 4,6. Getting [4, 4] for f order
    }

    @Test
    public void testAssign(){
        int[] shape1 = {3,2,2,2,2,2};
        int[] shape2 = {12,8};
        int length = ArrayUtil.prod(shape1);

        assertEquals(ArrayUtil.prod(shape1),ArrayUtil.prod(shape2));

        INDArray arr = Nd4j.linspace(1,length,length).reshape('c',shape1);
        INDArray arr2c = Nd4j.create(shape2,'c');
        INDArray arr2f = Nd4j.create(shape2,'f');

        arr2c.assign(arr);
        arr2f.assign(arr);

        INDArray exp = Nd4j.linspace(1,length,length).reshape('c',shape2);

        assertEquals(exp,arr2c);
        assertEquals(exp,arr2f);
    }

    @Test
    public void testSumDifferentOrders() {
        INDArray arrc = Nd4j.linspace(1,6,6).reshape('c',3,2);
        INDArray arrf = Nd4j.create(new double[6],new int[]{3,2},'f').assign(arrc);

        assertEquals(arrc,arrf);
        //c works
        INDArray cSum = arrc.sum(0);
        //f doesn't
        INDArray fSum = arrf.sum(0);
        assertEquals(cSum,fSum);  //Expect: 0.51, 1.79; getting [0.51,1.71] for f order
    }
    @Override
    public char ordering() {
        return 'c';
    }
}
