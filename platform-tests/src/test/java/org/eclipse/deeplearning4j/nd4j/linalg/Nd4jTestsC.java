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
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.util.FastMath;
import org.junit.jupiter.api.*;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.common.util.MathUtils;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.blas.params.GemmParams;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.iter.INDArrayIterator;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAMax;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAMin;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMax;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMin;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastEqualTo;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastGreaterThan;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastGreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastLessThan;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgAmax;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgAmin;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMin;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Im2col;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.reduce.bool.All;
import org.nd4j.linalg.api.ops.impl.reduce.custom.LogSumExp;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2;
import org.nd4j.linalg.api.ops.impl.reduce.same.Sum;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.HammingDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.scalar.LeakyReLU;
import org.nd4j.linalg.api.ops.impl.scalar.ReplaceNans;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.api.ops.impl.transforms.custom.BatchToSpaceND;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Reverse;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.BinaryRelativeError;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.Set;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.Axpy;
import org.nd4j.linalg.api.ops.impl.transforms.same.Sign;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ACosh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh;
import org.nd4j.linalg.api.ops.util.PrintVariable;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * NDArrayTests
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
public class Nd4jTestsC extends BaseNd4jTestWithBackends {

    @TempDir Path testDir;

    @Override
    public long getTimeoutMilliseconds() {
        return 90000;
    }

    @BeforeEach
    public void before() throws Exception {
        Nd4j.getRandom().setSeed(123);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @AfterEach
    public void after() throws Exception {
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyStringScalar(Nd4jBackend backend) {
        INDArray arr = Nd4j.empty(DataType.UTF8);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutWhereWithMask(Nd4jBackend backend) {
        double[][] arr = new double[][]{{1., 2.}, {1., 4.}, {1., 6}};
        double[][] expected = new double[][] {
                {2,2},
                {2,4},
                {2,6}
        };
        INDArray assertion = Nd4j.create(expected);
        INDArray dataMatrix = Nd4j.createFromArray(arr);
        INDArray compareTo = Nd4j.valueArrayOf(dataMatrix.shape(), 1.);
        INDArray replacement = Nd4j.valueArrayOf(dataMatrix.shape(), 2);
        INDArray mask = dataMatrix.match(compareTo, Conditions.equals(1));
        INDArray out = dataMatrix.putWhereWithMask(mask, replacement);
        assertEquals(assertion,out);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConditions(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);
        double[][] arr = new double[][]{{1., 2.}, {1., 4.}, {1., 6}};
        INDArray dataMatrix = Nd4j.createFromArray(arr);
        INDArray compareTo = Nd4j.valueArrayOf(dataMatrix.shape(), 1.);
        INDArray mask1 = dataMatrix.dup().match(compareTo, Conditions.epsNotEquals(1));
        INDArray mask2 = dataMatrix.dup().match(compareTo, Conditions.epsEquals(1));
        assertNotEquals(mask1,mask2);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArangeNegative(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(-2,2).castTo(DataType.DOUBLE);
        INDArray assertion = Nd4j.create(new double[]{-2, -1,  0,  1});
        assertEquals(assertion,arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTri(Nd4jBackend backend) {
        INDArray assertion = Nd4j.create(new double[][]{
                {1,1,1,0,0},
                {1,1,1,1,0},
                {1,1,1,1,1}
        });

        INDArray tri = Nd4j.tri(3,5,2);
        assertEquals(assertion,tri);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTriu(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        INDArray input = Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape(4,3);
        int k = -1;
        INDArray test = Nd4j.triu(input,k);
        INDArray create = Nd4j.create(new double[][]{
                {1,2,3},
                {4,5,6},
                {0,8,9},
                {0,0,12}
        });

        assertEquals(create,test);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDiag(Nd4jBackend backend) {
        INDArray diag = Nd4j.diag(Nd4j.linspace(1,4,4, DataType.DOUBLE).reshape(4,1));
        assertArrayEquals(new long[] {4,4},diag.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetRowEdgeCase(Nd4jBackend backend) {
        INDArray orig = Nd4j.linspace(1,300,300, DataType.DOUBLE).reshape('c', 100, 3);
        INDArray col = orig.getColumn(0).reshape(100, 1);

        for( int i = 0; i < 100; i++) {
            INDArray row = col.getRow(i);
            INDArray rowDup = row.dup();
            double d = orig.getDouble(i, 0);
            double d2 = col.getDouble(i);
            double dRowDup = rowDup.getDouble(0);
            double dRow = row.getDouble(0);

            String s = String.valueOf(i);
            assertEquals(d, d2, 0.0,s);
            assertEquals(d, dRowDup, 0.0,s);   //Fails
            assertEquals(d, dRow, 0.0,s);      //Fails
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNd4jEnvironment(Nd4jBackend backend) {
        System.out.println(Nd4j.getExecutioner().getEnvironmentInformation());
        int manualNumCores = Integer.parseInt(Nd4j.getExecutioner().getEnvironmentInformation()
                .get(Nd4jEnvironment.CPU_CORES_KEY).toString());
        assertEquals(Runtime.getRuntime().availableProcessors(), manualNumCores);
        assertEquals(Runtime.getRuntime().availableProcessors(), Nd4jEnvironment.getEnvironment().getNumCores());
        System.out.println(Nd4jEnvironment.getEnvironment());
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
    public void testTensorAlongDimension2(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new float[100], new long[] {50, 1, 2});
        assertArrayEquals(new long[] {1, 2}, array.slice(0, 0).shape());

    }

    @Disabled // with broadcastables mechanic it'll be ok
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShapeEqualsOnElementWise(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            Nd4j.ones(10000, 1).sub(Nd4j.ones(1, 2));

        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsMaxVectorCase(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[] {1, 2, 4, 3}, new long[] {2, 2});
        INDArray assertion = Nd4j.create(new boolean[] {false, false, true, false}, new long[] {2, 2}, DataType.BOOL);
        INDArray test = Nd4j.getExecutioner().exec(new IsMax(arr))[0];
        assertEquals(assertion, test);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArgMax(Nd4jBackend backend) {
        INDArray toArgMax = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape(4, 3, 2);
        INDArray argMaxZero = Nd4j.argMax(toArgMax, 0);
        INDArray argMax = Nd4j.argMax(toArgMax, 1);
        INDArray argMaxTwo = Nd4j.argMax(toArgMax, 2);
        INDArray valueArray = Nd4j.valueArrayOf(new long[] {4, 2}, 2, DataType.LONG);
        INDArray valueArrayTwo = Nd4j.valueArrayOf(new long[] {3, 2}, 3, DataType.LONG);
        INDArray valueArrayThree = Nd4j.valueArrayOf(new long[] {4, 3}, 1, DataType.LONG);
        assertEquals(valueArrayTwo, argMaxZero);
        assertEquals(valueArray, argMax);

        assertEquals(valueArrayThree, argMaxTwo);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArgMax_119(Nd4jBackend backend) {
        val array = Nd4j.create(new double[]{1, 2, 119, 2});
        val max = array.argMax();

        assertTrue(max.isScalar());
        assertEquals(2L, max.getInt(0));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAutoBroadcastShape(Nd4jBackend backend) {
        val assertion = new long[]{2,2,2,5};
        val shapeTest = Shape.broadcastOutputShape(new long[]{2,1,2,1},new long[]{2,1,5});
        assertArrayEquals(assertion,shapeTest);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")

    public void testAutoBroadcastAdd(Nd4jBackend backend) {
        INDArray left = Nd4j.linspace(1,4,4, DataType.DOUBLE).reshape(2,1,2,1);
        INDArray right = Nd4j.linspace(1,10,10, DataType.DOUBLE).reshape(2,1,5);
        INDArray assertion = Nd4j.create(new double[]{2,3,4,5,6,3,4,5,6,7,7,8,9,10,11,8,9,10,11,12,4,5,6,7,8,5,6,7,8,9,9,10,11,12,13,10,11,12,13,14}).reshape(2,2,2,5);
        INDArray test = left.add(right);
        assertEquals(assertion,test);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAudoBroadcastAddMatrix(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1,4,4, DataType.DOUBLE).reshape(2,2);
        INDArray row = Nd4j.ones(1, 2);
        INDArray assertion = arr.add(1.0);
        INDArray test = arr.add(row);
        assertEquals(assertion,test);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarOps(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.ones(27).data(), new long[] {3, 3, 3});
        assertEquals(27d, n.length(), 1e-1);
        n.addi(Nd4j.scalar(1d));
        n.subi(Nd4j.scalar(1.0d));
        n.muli(Nd4j.scalar(1.0d));
        n.divi(Nd4j.scalar(1.0d));

        n = Nd4j.create(Nd4j.ones(27).data(), new long[] {3, 3, 3});
        assertEquals(27, n.sumNumber().doubleValue(), 1e-1,getFailureMessage(backend));
        INDArray a = n.slice(2);
        assertEquals( true, Arrays.equals(new long[] {3, 3}, a.shape()),getFailureMessage(backend));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTensorAlongDimension(Nd4jBackend backend) {
        val shape = new long[] {4, 5, 7};
        int length = ArrayUtil.prod(shape);
        INDArray arr = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape(shape);


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
    public void testGetDouble(Nd4jBackend backend) {
        INDArray n2 = Nd4j.create(Nd4j.linspace(1, 30, 30, DataType.DOUBLE).data(), new long[] {3, 5, 2});
        INDArray swapped = n2.swapAxes(n2.shape().length - 1, 1);
        INDArray slice0 = swapped.slice(0).slice(1);
        INDArray assertion = Nd4j.create(new double[] {2, 4, 6, 8, 10});
        assertEquals(assertion, slice0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWriteTxt() throws Exception {
        INDArray row = Nd4j.create(new double[][] {{1, 2}, {3, 4}});
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
        INDArray c = Nd4j.create(new double[][] {{1, 2}, {3, 4}}, 'c');
        assertEquals('c', c.ordering());
        assertEquals(order, Nd4j.order().charValue());
        INDArray f = Nd4j.create(new double[][] {{1, 2}, {3, 4}}, 'f');
        assertEquals('f', f.ordering());
        assertEquals(order, Nd4j.order().charValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatrix(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new float[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray brr = Nd4j.create(new float[] {5, 6}, new long[] {2});
        INDArray row = arr.getRow(0);
        row.subi(brr);
        assertEquals(Nd4j.create(new float[] {-4, -4}), arr.getRow(0));

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
    @Disabled
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
    public void testSubiRowVector(Nd4jBackend backend) {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape('c', 2, 2);
        INDArray row1 = oneThroughFour.getRow(1).dup();
        oneThroughFour.subiRowVector(row1);
        INDArray result = Nd4j.create(new double[] {-2, -2, 0, 0}, new long[] {2, 2});
        assertEquals(result, oneThroughFour,getFailureMessage(backend));

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddiRowVectorWithScalar(Nd4jBackend backend) {
        INDArray colVector = Nd4j.create(5, 1).assign(0.0);
        INDArray scalar = Nd4j.create(1, 1).assign(0.0);
        scalar.putScalar(0, 1);

        assertEquals(scalar.getDouble(0), 1.0, 0.0);

        colVector.addiRowVector(scalar); //colVector is all zeros after this
        for (int i = 0; i < 5; i++)
            assertEquals(colVector.getDouble(i), 1.0, 0.0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTADOnVector(Nd4jBackend backend) {

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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLength(Nd4jBackend backend) {
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


        INDArray expected = Nd4j.repeat(Nd4j.scalar(DataType.DOUBLE, 2).reshape(1, 1), 2).reshape(2);

        val accum = new EuclideanDistance(values, values2);
        accum.setDimensions(1);

        INDArray results = Nd4j.getExecutioner().exec(accum);
        assertEquals(expected, results);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadCasting(Nd4jBackend backend) {
        INDArray first = Nd4j.arange(0, 3).reshape(3, 1).castTo(DataType.DOUBLE);
        INDArray ret = first.broadcast(3, 4);
        INDArray testRet = Nd4j.create(new double[][] {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
        assertEquals(testRet, ret);
        INDArray r = Nd4j.arange(0, 4).reshape(1, 4).castTo(DataType.DOUBLE);
        INDArray r2 = r.broadcast(4, 4);
        INDArray testR2 = Nd4j.create(new double[][] {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}});
        assertEquals(testR2, r2);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetColumns(Nd4jBackend backend) {
        INDArray matrix = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
        INDArray matrixGet = matrix.getColumns(1, 2);
        INDArray matrixAssertion = Nd4j.create(new double[][] {{2, 3}, {5, 6}});
        assertEquals(matrixAssertion, matrixGet);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSort(Nd4jBackend backend) {
        INDArray toSort = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray ascending = Nd4j.sort(toSort.dup(), 1, true);
        //rows are already sorted
        assertEquals(toSort, ascending);

        INDArray columnSorted = Nd4j.create(new double[] {2, 1, 4, 3}, new long[] {2, 2});
        INDArray sorted = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(columnSorted, sorted);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSortRows(Nd4jBackend backend) {
        int nRows = 10;
        int nCols = 5;
        Random r = new Random(12345);

        for (int i = 0; i < nCols; i++) {
            INDArray in = Nd4j.linspace(1, nRows * nCols, nRows * nCols, DataType.DOUBLE).reshape(nRows, nCols);

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

//            System.out.println("outDesc: " + Arrays.toString(outAsc.data().asFloat()));
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFlattenedOrder(Nd4jBackend backend) {
        INDArray concatC = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape('c', 2, 2);
        INDArray concatF = Nd4j.create(new long[] {2, 2}, 'f');
        concatF.assign(concatC);
        INDArray assertionC = Nd4j.create(new double[] {1, 2, 3, 4, 1, 2, 3, 4});
        INDArray testC = Nd4j.toFlattened('c', concatC, concatF);
        assertEquals(assertionC, testC);
        INDArray test = Nd4j.toFlattened('f', concatC, concatF);
        INDArray assertion = Nd4j.create(new double[] {1, 3, 2, 4, 1, 3, 2, 4});
        assertEquals(assertion, test);


    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testZero(Nd4jBackend backend) {
        Nd4j.ones(11).sumNumber();
        Nd4j.ones(12).sumNumber();
        Nd4j.ones(2).sumNumber();
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumNumberRepeatability(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(1, 450).reshape('c', 150, 3);

        double first = arr.sumNumber().doubleValue();
        double assertion = 450;
        assertEquals(assertion, first, 1e-1);
        for (int i = 0; i < 50; i++) {
            double second = arr.sumNumber().doubleValue();
            assertEquals(assertion, second, 1e-1);
            assertEquals( first, second, 1e-2,String.valueOf(i));
        }
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
        INDArray f2d = Nd4j.create(new long[] {rows, cols}, 'f').assign(c2d).addi(0.1);

        INDArray c3d = Nd4j.linspace(1, length3d, length3d, DataType.DOUBLE).reshape('c', rows, cols, dim2);
        INDArray f3d = Nd4j.create(new long[] {rows, cols, dim2}).assign(c3d).addi(0.3);
        c3d.addi(0.2);

        INDArray c4d = Nd4j.linspace(1, length4d, length4d, DataType.DOUBLE).reshape('c', rows, cols, dim2, dim3);
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFlattenedOnViews(Nd4jBackend backend) {
        int rows = 8;
        int cols = 8;
        int dim2 = 4;
        int length = rows * cols;
        int length3d = rows * cols * dim2;

        INDArray first = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', rows, cols);
        INDArray second = Nd4j.create(new long[] {rows, cols}, 'f').assign(first);
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



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsMax2(Nd4jBackend backend) {
        //Tests: full buffer...
        //1d
        INDArray arr1 = Nd4j.create(new double[] {1, 2, 3, 1});
        val res1 = Nd4j.getExecutioner().exec(new IsMax(arr1))[0];
        INDArray exp1 = Nd4j.create(new boolean[] {false, false, true, false});

        assertEquals(exp1, res1);

        arr1 = Nd4j.create(new double[] {1, 2, 3, 1});
        INDArray result = Nd4j.createUninitialized(DataType.BOOL, 4);
        Nd4j.getExecutioner().execAndReturn(new IsMax(arr1, result));

        assertEquals(Nd4j.create(new double[] {1, 2, 3, 1}), arr1);
        assertEquals(exp1, result);

        //2d
        INDArray arr2d = Nd4j.create(new double[][] {{0, 1, 2}, {2, 9, 1}});
        INDArray exp2d = Nd4j.create(new boolean[][] {{false, false, false}, {false, true, false}});

        INDArray f = arr2d.dup('f');
        INDArray out2dc = Nd4j.getExecutioner().exec(new IsMax(arr2d.dup('c')))[0];
        INDArray out2df = Nd4j.getExecutioner().exec(new IsMax(arr2d.dup('f')))[0];
        assertEquals(exp2d, out2dc);
        assertEquals(exp2d, out2df);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFlattened3(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsMaxEqualValues(Nd4jBackend backend) {
        //Assumption here: should only have a 1 for *first* maximum value, if multiple values are exactly equal

        //[1 1 1] -> [1 0 0]
        //Loop to double check against any threading weirdness...
        for (int i = 0; i < 10; i++) {
            val res = Transforms.isMax(Nd4j.ones(3), DataType.BOOL);
            assertEquals(Nd4j.create(new boolean[] {true, false, false}), res);
        }

        //[0 0 0 2 2 0] -> [0 0 0 1 0 0]
        assertEquals(Nd4j.create(new boolean[] {false, false, false, true, false, false}), Transforms.isMax(Nd4j.create(new double[] {0, 0, 0, 2, 2, 0}), DataType.BOOL));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIMaxVector_1(Nd4jBackend backend) {
        val array = Nd4j.ones(3);
        val idx = array.argMax(0).getInt(0);
        assertEquals(0, idx);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIMaxVector_2(Nd4jBackend backend) {
        val array = Nd4j.ones(3);
        val idx = array.argMax(Integer.MAX_VALUE).getInt(0);
        assertEquals(0, idx);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIMaxVector_3(Nd4jBackend backend) {
        val array = Nd4j.ones(3);
        val idx = array.argMax().getInt(0);
        assertEquals(0, idx);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsMaxEqualValues_2(Nd4jBackend backend) {
        //[0 2]    [0 1]
        //[2 1] -> [0 0]bg
        INDArray orig = Nd4j.create(new double[][] {{0, 3}, {2, 1}});
        INDArray exp = Nd4j.create(new double[][] {{0, 1}, {0, 0}});
        INDArray outc = Transforms.isMax(orig.dup('c'));
        assertEquals(exp, outc);

//        log.info("Orig: {}", orig.dup('f').data().asFloat());

        INDArray outf = Transforms.isMax(orig.dup('f'), orig.dup('f').ulike());
//        log.info("OutF: {}", outf.data().asFloat());
        assertEquals(exp, outf);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsMaxEqualValues_3(Nd4jBackend backend) {
        //[0 2]    [0 1]
        //[2 1] -> [0 0]
        INDArray orig = Nd4j.create(new double[][] {{0, 2}, {3, 1}});
        INDArray exp = Nd4j.create(new double[][] {{0, 0}, {1, 0}});
        INDArray outc = Transforms.isMax(orig.dup('c'));
        assertEquals(exp, outc);

        INDArray outf = Transforms.isMax(orig.dup('f'), orig.dup('f').ulike());
        assertEquals(exp, outf);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSqrt_1(Nd4jBackend backend) {
        val x = Nd4j.createFromArray(9.0, 9.0, 9.0, 9.0);
        val x2 = Nd4j.createFromArray(9.0, 9.0, 9.0, 9.0);
        val e = Nd4j.createFromArray(3.0, 3.0, 3.0, 3.0);

        val z1 = Transforms.sqrt(x, true);
        val z2 = Transforms.sqrt(x2, false);


        assertEquals(e, z2);
        assertEquals(e, x2);
        assertEquals(e, z1);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssign_CF(Nd4jBackend backend) {
        val orig = Nd4j.create(new double[][] {{0, 2}, {2, 1}});
        val oc = orig.dup('c');
        val of = orig.dup('f');

        assertEquals(orig, oc);
        assertEquals(orig, of);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsMaxAlongDimension(Nd4jBackend backend) {
        //1d: row vector
        INDArray orig = Nd4j.create(new double[] {1, 2, 3, 1}).reshape(1,4 );

        INDArray alongDim0 = Nd4j.getExecutioner().exec(new IsMax(orig.dup(), Nd4j.createUninitialized(DataType.BOOL, orig.shape()), 0))[0];
        INDArray alongDim1 = Nd4j.getExecutioner().exec(new IsMax(orig.dup(), Nd4j.createUninitialized(DataType.BOOL, orig.shape()), 1))[0];

        INDArray expAlong0 = Nd4j.create(new boolean[]{true, true, true, true}).reshape(1,4);
        INDArray expAlong1 = Nd4j.create(new boolean[] {false, false, true, false}).reshape(1,4);

        assertEquals(expAlong0, alongDim0);
        assertEquals(expAlong1, alongDim1);


        //1d: col vector
//        System.out.println("----------------------------------");
        INDArray col = Nd4j.create(new double[] {1, 2, 3, 1}, new long[] {4, 1});
        INDArray alongDim0col = Nd4j.getExecutioner().exec(new IsMax(col.dup(), Nd4j.createUninitialized(DataType.BOOL, col.shape()), 0))[0];
        INDArray alongDim1col = Nd4j.getExecutioner().exec(new IsMax(col.dup(), Nd4j.createUninitialized(DataType.BOOL, col.shape()),1))[0];

        INDArray expAlong0col = Nd4j.create(new boolean[] {false, false, true, false}).reshape(4,1);
        INDArray expAlong1col = Nd4j.create(new boolean[] {true, true, true, true}).reshape(4,1);



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
//        System.out.println("---------------------");
        INDArray orig2d = Nd4j.create(new double[][] {{1, 0, 2}, {2, 3, 1}});
        INDArray alongDim0c_2d = Nd4j.getExecutioner().exec(new IsMax(orig2d.dup('c'), Nd4j.createUninitialized(DataType.BOOL, orig2d.shape()), 0))[0];
        INDArray alongDim0f_2d = Nd4j.getExecutioner().exec(new IsMax(orig2d.dup('f'), Nd4j.createUninitialized(DataType.BOOL, orig2d.shape(), 'f'), 0))[0];
        INDArray alongDim1c_2d = Nd4j.getExecutioner().exec(new IsMax(orig2d.dup('c'), Nd4j.createUninitialized(DataType.BOOL, orig2d.shape()), 1))[0];
        INDArray alongDim1f_2d = Nd4j.getExecutioner().exec(new IsMax(orig2d.dup('f'), Nd4j.createUninitialized(DataType.BOOL, orig2d.shape(), 'f'), 1))[0];

        INDArray expAlong0_2d = Nd4j.create(new boolean[][] {{false, false, true}, {true, true, false}});
        INDArray expAlong1_2d = Nd4j.create(new boolean[][] {{false, false, true}, {false, true, false}});

        assertEquals(expAlong0_2d, alongDim0c_2d);
        assertEquals(expAlong0_2d, alongDim0f_2d);
        assertEquals(expAlong1_2d, alongDim1c_2d);
        assertEquals(expAlong1_2d, alongDim1f_2d);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIMaxSingleDim1(Nd4jBackend backend) {
        INDArray orig2d = Nd4j.create(new double[][] {{1, 0, 2}, {2, 3, 1}});

        INDArray result = Nd4j.argMax(orig2d.dup('c'), 0);

//        System.out.println("IMAx result: " + result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsMaxSingleDim1(Nd4jBackend backend) {
        INDArray orig2d = Nd4j.create(new double[][] {{1, 0, 2}, {2, 3, 1}});
        INDArray alongDim0c_2d = Nd4j.getExecutioner().exec(new IsMax(orig2d.dup('c'), Nd4j.createUninitialized(DataType.BOOL, orig2d.shape()), 0))[0];
        INDArray expAlong0_2d = Nd4j.create(new boolean[][] {{false, false, true}, {true, true, false}});

//        System.out.println("Original shapeInfo: " + orig2d.dup('c').shapeInfoDataBuffer());

//        System.out.println("Expected: " + Arrays.toString(expAlong0_2d.data().asFloat()));
//        System.out.println("Actual: " + Arrays.toString(alongDim0c_2d.data().asFloat()));
        assertEquals(expAlong0_2d, alongDim0c_2d);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastRepeated(Nd4jBackend backend) {
        INDArray z = Nd4j.create(1, 4, 4, 3);
        INDArray bias = Nd4j.create(1, 3);
        BroadcastOp op = new BroadcastAddOp(z, bias, z, 3);
        Nd4j.getExecutioner().exec(op);
//        System.out.println("First: OK");
        //OK at this point: executes successfully


        z = Nd4j.create(1, 4, 4, 3);
        bias = Nd4j.create(1, 3);
        op = new BroadcastAddOp(z, bias, z, 3);
        Nd4j.getExecutioner().exec(op); //Crashing here, when we are doing exactly the same thing as before...
//        System.out.println("Second: OK");
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVStackDifferentOrders(Nd4jBackend backend) {
        INDArray expected = Nd4j.linspace(1, 9, 9, DataType.DOUBLE).reshape(3, 3);

        for (char order : new char[] {'c', 'f'}) {
//            System.out.println(order);

            INDArray arr1 = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape( 2, 3).dup('c');
            INDArray arr2 = Nd4j.linspace(7, 9, 3, DataType.DOUBLE).reshape(1, 3).dup('c');

            Nd4j.factory().setOrder(order);

//            log.info("arr1: {}", arr1.data());
//            log.info("arr2: {}", arr2.data());

            INDArray merged = Nd4j.vstack(arr1, arr2);
//            System.out.println(merged.data());
//            System.out.println(expected);

            assertEquals( expected, merged,"Failed for [" + order + "] order");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVStackEdgeCase(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 4, 4, DataType.DOUBLE);
        INDArray vstacked = Nd4j.vstack(arr);
        assertEquals(arr.reshape(1,4), vstacked);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEps3(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        INDArray first = Nd4j.linspace(1, 10, 10, DataType.DOUBLE);
        INDArray second = Nd4j.linspace(20, 30, 10, DataType.DOUBLE);

        INDArray firstResult = Nd4j.create(DataType.BOOL, 10);
        INDArray secondResult = Nd4j.create(DataType.BOOL, 10);

        INDArray expAllZeros = Nd4j.getExecutioner().exec(new Eps(first, second, firstResult));
        INDArray expAllOnes = Nd4j.getExecutioner().exec(new Eps(first, first, secondResult));


        val allones = Nd4j.getExecutioner().exec(new All(expAllOnes)).getDouble(0);

        assertTrue(expAllZeros.none());
        assertTrue(expAllOnes.all());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testSumAlongDim1sEdgeCases(Nd4jBackend backend) {
        val shapes = new long[][] {
                //Standard case:
                {2, 2, 3, 4},
                //Leading 1s:
                {1, 2, 3, 4}, {1, 1, 2, 3},
                //Trailing 1s:
                {4, 3, 2, 1}, {4, 3, 1, 1},
                //1s for non-leading/non-trailing dimensions
                {4, 1, 3, 2}, {4, 3, 1, 2}, {4, 1, 1, 2}};

        long[][] sumDims = {{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3},
                {0, 1, 2, 3}};

        for (val shape : shapes) {
            for (long[] dims : sumDims) {
                int length = ArrayUtil.prod(shape);
                INDArray inC = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape);
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsMaxAlongDimensionSimple(Nd4jBackend backend) {
        //Simple test: when doing IsMax along a dimension, we expect all values to be either 0 or 1
        //Do IsMax along dims 0&1 for rank 2, along 0,1&2 for rank 3, etc

        for (int rank = 2; rank <= 6; rank++) {

            int[] shape = new int[rank];
            for (int i = 0; i < rank; i++)
                shape[i] = 2;
            int length = ArrayUtil.prod(shape);


            for (int alongDimension = 0; alongDimension < rank; alongDimension++) {
//                System.out.println("Testing rank " + rank + " along dimension " + alongDimension + ", (shape="
//                        + Arrays.toString(shape) + ")");
                INDArray arrC = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape);
                INDArray arrF = arrC.dup('f');
                val resC = Nd4j.getExecutioner().exec(new IsMax(arrC, alongDimension))[0];
                val resF = Nd4j.getExecutioner().exec(new IsMax(arrF, alongDimension))[0];


                double[] cBuffer = resC.data().asDouble();
                double[] fBuffer = resF.data().asDouble();
                for (int i = 0; i < length; i++) {
                    assertTrue(cBuffer[i] == 0.0 || cBuffer[i] == 1.0,"c buffer value at [" + i + "]=" + cBuffer[i] + ", expected 0 or 1; dimension = "
                            + alongDimension + ", rank = " + rank + ", shape=" + Arrays.toString(shape));
                }
                for (int i = 0; i < length; i++) {
                    assertTrue(fBuffer[i] == 0.0 || fBuffer[i] == 1.0,"f buffer value at [" + i + "]=" + fBuffer[i] + ", expected 0 or 1; dimension = "
                            + alongDimension + ", rank = " + rank + ", shape=" + Arrays.toString(shape));
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSortColumns(Nd4jBackend backend) {
        int nRows = 5;
        int nCols = 10;
        Random r = new Random(12345);

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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddVectorWithOffset(Nd4jBackend backend) {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray row1 = oneThroughFour.getRow(1);
        row1.addi(1);
        INDArray result = Nd4j.create(new double[] {1, 2, 4, 5}, new long[] {2, 2});
        assertEquals(result, oneThroughFour,getFailureMessage(backend));


    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLinearViewGetAndPut(Nd4jBackend backend) {
        INDArray test = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray linear = test.reshape(-1);
        linear.putScalar(2, 6);
        linear.putScalar(3, 7);
        assertEquals(6, linear.getFloat(2), 1e-1,getFailureMessage(backend));
        assertEquals(7, linear.getFloat(3), 1e-1,getFailureMessage(backend));
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowVectorGemm(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(1, 4);
        INDArray other = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(4, 4);
        INDArray result = linspace.mmul(other);
        INDArray assertion = Nd4j.create(new double[] {90, 100, 110, 120}).reshape(4, 1);
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
    public void testMultiSum(Nd4jBackend backend) {
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
        INDArray arr = Nd4j.linspace(0, 7, 8, DataType.DOUBLE).reshape('c', 2, 2, 2);
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

        INDArray twoTwoByThree = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape('f', 2, 2, 3);
        INDArray multiSum = twoTwoByThree.sum(0, 1);
        assertEquals(assertion, multiSum);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum2dv2(Nd4jBackend backend) {
        INDArray in = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape('c', 2, 2, 2);

        val dims = new long[][] {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
        double[][] exp = new double[][] {{16, 20}, {16, 20}, {14, 22}, {14, 22}, {10, 26}, {10, 26}};

        for (int i = 0; i < dims.length; i++) {
            val d = dims[i];
            double[] e = exp[i];

            INDArray out = in.sum(d);

            assertEquals(Nd4j.create(e, out.shape()), out);
        }
    }


    //Passes on 3.9:
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum3Of4_2222(Nd4jBackend backend) {
        int[] shape = {2, 2, 2, 2};
        int length = ArrayUtil.prod(shape);
        INDArray arrC = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape);
        INDArray arrF = Nd4j.create(arrC.shape()).assign(arrC);

        long[][] dimsToSum = new long[][] {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
        double[][] expD = new double[][] {{64, 72}, {60, 76}, {52, 84}, {36, 100}};

        for (int i = 0; i < dimsToSum.length; i++) {
            long[] d = dimsToSum[i];

            INDArray outC = arrC.sum(d);
            INDArray outF = arrF.sum(d);
            INDArray exp = Nd4j.create(expD[i], outC.shape()).castTo(DataType.DOUBLE);

            assertEquals(exp, outC);
            assertEquals(exp, outF);

//            System.out.println(Arrays.toString(d) + "\t" + outC + "\t" + outF);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcast1d(Nd4jBackend backend) {
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
            zC.setData(Nd4j.linspace(1, 24, 24, DataType.DOUBLE).data());
            for (int tad = 0; tad < zC.tensorsAlongDimension(dim); tad++) {
                INDArray javaTad = zC.tensorAlongDimension(tad, dim);

            }

            INDArray zF = Nd4j.create(shape, 'f');
            zF.assign(zC);
            INDArray toBroadcast = Nd4j.linspace(1, length, length, DataType.DOUBLE);

            Op opc = new BroadcastAddOp(zC, toBroadcast, zC, dim);
            Op opf = new BroadcastAddOp(zF, toBroadcast, zF, dim);
            INDArray exp = Nd4j.create(expLinspaced[i], shape, 'c');
            INDArray expF = Nd4j.create(shape, 'f');
            expF.assign(exp);

            Nd4j.getExecutioner().exec(opc);
            Nd4j.getExecutioner().exec(opf);

            assertEquals(exp, zC);
            assertEquals(exp, zF);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum3Of4_3322(Nd4jBackend backend) {
        int[] shape = {3, 3, 2, 2};
        int length = ArrayUtil.prod(shape);
        INDArray arrC = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape);
        INDArray arrF = Nd4j.create(arrC.shape()).assign(arrC);

        long[][] dimsToSum = new long[][] {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
        double[][] expD = new double[][] {{324, 342}, {315, 351}, {174, 222, 270}, {78, 222, 366}};

        for (int i = 0; i < dimsToSum.length; i++) {
            long[] d = dimsToSum[i];

            INDArray outC = arrC.sum(d);
            INDArray outF = arrF.sum(d);
            INDArray exp = Nd4j.create(expD[i], outC.shape());

            assertEquals(exp, outC);
            assertEquals(exp, outF);

            //System.out.println(Arrays.toString(d) + "\t" + outC + "\t" + outF);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFlattened(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        List<INDArray> concat = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            concat.add(arr.dup());
        }

        INDArray assertion = Nd4j.create(new double[] {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, new int[]{12});
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSortWithIndicesDescending(Nd4jBackend backend) {
        INDArray toSort = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        //indices,data
        INDArray[] sorted = Nd4j.sortWithIndices(toSort.dup(), 1, false);
        INDArray sorted2 = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(sorted[1], sorted2);
        INDArray shouldIndex = Nd4j.create(new double[] {1, 0, 1, 0}, new long[] {2, 2});
        assertEquals(shouldIndex, sorted[0]);


    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetFromRowVector(Nd4jBackend backend) {
        INDArray matrix = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray rowGet = matrix.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 2));
        assertArrayEquals(new long[] {2}, rowGet.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testSubRowVector(Nd4jBackend backend) {
        INDArray matrix = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
        INDArray row = Nd4j.linspace(1, 3, 3, DataType.DOUBLE);
        INDArray test = matrix.subRowVector(row);
        INDArray assertion = Nd4j.create(new double[][] {{0, 0, 0}, {3, 3, 3}});
        assertEquals(assertion, test);

        INDArray threeByThree = Nd4j.linspace(1, 9, 9, DataType.DOUBLE).reshape(3, 3);
        INDArray offsetTest = threeByThree.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        assertEquals(2, offsetTest.rows());
        INDArray offsetAssertion = Nd4j.create(new double[][] {{3, 3, 3}, {6, 6, 6}});
        INDArray offsetSub = offsetTest.subRowVector(row);
        assertEquals(offsetAssertion, offsetSub);

    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDimShuffle(Nd4jBackend backend) {
        INDArray n = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray twoOneTwo = n.dimShuffle(new Object[] {0, 'x', 1}, new int[] {0, 1}, new boolean[] {false, false});
        assertTrue(Arrays.equals(new long[] {2, 1, 2}, twoOneTwo.shape()));

        INDArray reverse = n.dimShuffle(new Object[] {1, 'x', 0}, new int[] {1, 0}, new boolean[] {false, false});
        assertTrue(Arrays.equals(new long[] {2, 1, 2}, reverse.shape()));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetVsGetScalar(Nd4jBackend backend) {
        INDArray a = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        float element = a.getFloat(0, 1);
        double element2 = a.getDouble(0, 1);
        assertEquals(element, element2, 1e-1);
        INDArray a2 = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        float element23 = a2.getFloat(0, 1);
        double element22 = a2.getDouble(0, 1);
        assertEquals(element23, element22, 1e-1);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDivide(Nd4jBackend backend) {
        INDArray two = Nd4j.create(new double[] {2, 2, 2, 2}).castTo(DataType.DOUBLE);
        INDArray div = two.div(two);
        assertEquals(Nd4j.ones(4), div);

        INDArray half = Nd4j.create(new double[] {0.5f, 0.5f, 0.5f, 0.5f}, new long[] {2, 2});
        INDArray divi = Nd4j.create(new double[] {0.3f, 0.6f, 0.9f, 0.1f}, new long[] {2, 2});
        INDArray assertion = Nd4j.create(new double[] {1.6666666f, 0.8333333f, 0.5555556f, 5}, new long[] {2, 2});
        INDArray result = half.div(divi);
        assertEquals(assertion, result);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSigmoid(Nd4jBackend backend) {
        INDArray n = Nd4j.create(new float[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        INDArray assertion = Nd4j.create(new float[] {0.73105858f, 0.88079708f, 0.95257413f, 0.98201379f}).castTo(DataType.DOUBLE);
        INDArray sigmoid = Transforms.sigmoid(n, false);
        assertEquals(assertion, sigmoid);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNeg(Nd4jBackend backend) {
        INDArray n = Nd4j.create(new float[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        INDArray assertion = Nd4j.create(new float[] {-1, -2, -3, -4}).castTo(DataType.DOUBLE);
        INDArray neg = Transforms.neg(n);
        assertEquals(assertion, neg,getFailureMessage(backend));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNorm2Double(Nd4jBackend backend) {
        DataType initialType = Nd4j.dataType();

        INDArray n = Nd4j.create(new double[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        double assertion = 5.47722557505;
        double norm3 = n.norm2Number().doubleValue();
        assertEquals(assertion, norm3, 1e-1,getFailureMessage(backend));

        INDArray row = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {2, 2}).castTo(DataType.DOUBLE);
        INDArray row1 = row.getRow(1);
        double norm2 = row1.norm2Number().doubleValue();
        double assertion2 = 5.0f;
        assertEquals(assertion2, norm2, 1e-1,getFailureMessage(backend));

        Nd4j.setDataType(initialType);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNorm2(Nd4jBackend backend) {
        INDArray n = Nd4j.create(new float[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        float assertion = 5.47722557505f;
        float norm3 = n.norm2Number().floatValue();
        assertEquals(assertion, norm3, 1e-1,getFailureMessage(backend));


        INDArray row = Nd4j.create(new float[] {1, 2, 3, 4}, new long[] {2, 2}).castTo(DataType.DOUBLE);
        INDArray row1 = row.getRow(1);
        float norm2 = row1.norm2Number().floatValue();
        float assertion2 = 5.0f;
        assertEquals(assertion2, norm2, 1e-1,getFailureMessage(backend));

    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCosineSim(Nd4jBackend backend) {
        INDArray vec1 = Nd4j.create(new double[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        INDArray vec2 = Nd4j.create(new double[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(1, sim, 1e-1,getFailureMessage(backend));

        INDArray vec3 = Nd4j.create(new float[] {0.2f, 0.3f, 0.4f, 0.5f});
        INDArray vec4 = Nd4j.create(new float[] {0.6f, 0.7f, 0.8f, 0.9f});
        sim = Transforms.cosineSim(vec3, vec4);
        assertEquals(0.98, sim, 1e-1);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScal(Nd4jBackend backend) {
        double assertion = 2;
        INDArray answer = Nd4j.create(new double[] {2, 4, 6, 8}).castTo(DataType.DOUBLE);
        INDArray scal = Nd4j.getBlasWrapper().scal(assertion, answer);
        assertEquals(answer, scal,getFailureMessage(backend));

        INDArray row = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray row1 = row.getRow(1);
        double assertion2 = 5.0;
        INDArray answer2 = Nd4j.create(new double[] {15, 20});
        INDArray scal2 = Nd4j.getBlasWrapper().scal(assertion2, row1);
        assertEquals(answer2, scal2,getFailureMessage(backend));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExp(Nd4jBackend backend) {
        INDArray n = Nd4j.create(new double[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        INDArray assertion = Nd4j.create(new double[] {2.71828183f, 7.3890561f, 20.08553692f, 54.59815003f}).castTo(DataType.DOUBLE);
        INDArray exped = Transforms.exp(n);
        assertEquals(assertion, exped);

        assertArrayEquals(new double[] {2.71828183f, 7.3890561f, 20.08553692f, 54.59815003f}, exped.toDoubleVector(), 1e-5);
        assertArrayEquals(new double[] {2.71828183f, 7.3890561f, 20.08553692f, 54.59815003f}, assertion.toDoubleVector(), 1e-5);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSlices(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24, DataType.DOUBLE).data(), new long[] {4, 3, 2});
        for (int i = 0; i < arr.slices(); i++) {
            assertEquals(2, arr.slice(i).slice(1).slices());
        }

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalar(Nd4jBackend backend) {
        INDArray a = Nd4j.scalar(1.0f).castTo(DataType.DOUBLE);
        assertEquals(true, a.isScalar());

        INDArray n = Nd4j.create(new float[] {1.0f}, new long[0]).castTo(DataType.DOUBLE);
        assertEquals(n, a);
        assertTrue(n.isScalar());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWrap(Nd4jBackend backend) {
        int[] shape = {2, 4};
        INDArray d = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape(shape[0], shape[1]);
        INDArray n = d;
        assertEquals(d.rows(), n.rows());
        assertEquals(d.columns(), n.columns());

        INDArray vector = Nd4j.linspace(1, 3, 3, DataType.DOUBLE);
        INDArray testVector = vector;
        for (int i = 0; i < vector.length(); i++)
            assertEquals(vector.getDouble(i), testVector.getDouble(i), 1e-1);
        assertEquals(3, testVector.length());
        assertEquals(true, testVector.isVector());
        assertEquals(true, Shape.shapeEquals(new long[] {3}, testVector.shape()));

        INDArray row12 = Nd4j.linspace(1, 2, 2, DataType.DOUBLE).reshape(2, 1);
        INDArray row22 = Nd4j.linspace(3, 4, 2, DataType.DOUBLE).reshape(1, 2);

        assertEquals(row12.rows(), 2);
        assertEquals(row12.columns(), 1);
        assertEquals(row22.rows(), 1);
        assertEquals(row22.columns(), 2);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorInit(Nd4jBackend backend) {
        DataBuffer data = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data();
        INDArray arr = Nd4j.create(data, new long[] {1, 4});
        assertEquals(true, arr.isRowVector());
        INDArray arr2 = Nd4j.create(data, new long[] {1, 4});
        assertEquals(true, arr2.isRowVector());

        INDArray columnVector = Nd4j.create(data, new long[] {4, 1});
        assertEquals(true, columnVector.isColumnVector());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testColumns(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new long[] {3, 2}).castTo(DataType.DOUBLE);
        INDArray column2 = arr.getColumn(0);
        //assertEquals(true, Shape.shapeEquals(new long[]{3, 1}, column2.shape()));
        INDArray column = Nd4j.create(new double[] {1, 2, 3}, new long[] {3});
        arr.putColumn(0, column);

        INDArray firstColumn = arr.getColumn(0);

        assertEquals(column, firstColumn);


        INDArray column1 = Nd4j.create(new double[] {4, 5, 6}, new long[] {3});
        arr.putColumn(1, column1);
        //assertEquals(true, Shape.shapeEquals(new long[]{3, 1}, arr.getColumn(1).shape()));
        INDArray testRow1 = arr.getColumn(1);
        assertEquals(column1, testRow1);


        INDArray evenArr = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray put = Nd4j.create(new double[] {5, 6}, new long[] {2});
        evenArr.putColumn(1, put);
        INDArray testColumn = evenArr.getColumn(1);
        assertEquals(put, testColumn);


        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data(), new long[] {2, 2});
        INDArray column23 = n.getColumn(0);
        INDArray column12 = Nd4j.create(new double[] {1, 3}, new long[] {2});
        assertEquals(column23, column12);


        INDArray column0 = n.getColumn(1);
        INDArray column01 = Nd4j.create(new double[] {2, 4}, new long[] {2});
        assertEquals(column0, column01);


    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutRow(Nd4jBackend backend) {
        INDArray d = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
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

        INDArray newRow = Nd4j.linspace(5, 6, 2, DataType.DOUBLE);
        n.putRow(0, newRow);
        d.putRow(0, newRow);


        INDArray testRow = n.getRow(0);
        assertEquals(newRow.length(), testRow.length());
        assertEquals(true, Shape.shapeEquals(new long[] {1, 2}, testRow.shape()));


        INDArray nLast = Nd4j.create(Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data(), new long[] {2, 2});
        INDArray row = nLast.getRow(1);
        INDArray row1 = Nd4j.create(new double[] {3, 4});
        assertEquals(row, row1);


        INDArray arr = Nd4j.create(new long[] {3, 2});
        INDArray evenRow = Nd4j.create(new double[] {1, 2});
        arr.putRow(0, evenRow);
        INDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new long[] {2}, firstRow.shape()));
        INDArray testRowEven = arr.getRow(0);
        assertEquals(evenRow, testRowEven);


        INDArray row12 = Nd4j.create(new double[] {5, 6}, new long[] {2});
        arr.putRow(1, row12);
        assertEquals(true, Shape.shapeEquals(new long[] {2}, arr.getRow(0).shape()));
        INDArray testRow1 = arr.getRow(1);
        assertEquals(row12, testRow1);


        INDArray multiSliceTest = Nd4j.create(Nd4j.linspace(1, 16, 16, DataType.DOUBLE).data(), new long[] {4, 2, 2});
        INDArray test = Nd4j.create(new double[] {5, 6}, new long[] {2});
        INDArray test2 = Nd4j.create(new double[] {7, 8}, new long[] {2});

        INDArray multiSliceRow1 = multiSliceTest.slice(1).getRow(0);
        INDArray multiSliceRow2 = multiSliceTest.slice(1).getRow(1);

        assertEquals(test, multiSliceRow1);
        assertEquals(test2, multiSliceRow2);



        INDArray threeByThree = Nd4j.create(3, 3);
        INDArray threeByThreeRow1AndTwo = threeByThree.get(NDArrayIndex.interval(1, 2), NDArrayIndex.all());
        threeByThreeRow1AndTwo.putRow(1, Nd4j.ones(3));
        assertEquals(Nd4j.ones(3), threeByThreeRow1AndTwo.getRow(0));

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMulRowVector(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        arr.muliRowVector(Nd4j.linspace(1, 2, 2, DataType.DOUBLE));
        INDArray assertion = Nd4j.create(new double[][] {{1, 4}, {3, 8}});

        assertEquals(assertion, arr);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8, DataType.DOUBLE).data(), new long[] {2, 2, 2});
        INDArray test = Nd4j.create(new double[] {3, 7, 11, 15}, new long[] {2, 2});
        INDArray sum = n.sum(-1);
        assertEquals(test, sum);

    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInplaceTranspose(Nd4jBackend backend) {
        INDArray test = Nd4j.rand(3, 4).castTo(DataType.DOUBLE);
        INDArray orig = test.dup();
        INDArray transposei = test.transposei();

        for (int i = 0; i < orig.rows(); i++) {
            for (int j = 0; j < orig.columns(); j++) {
                assertEquals(orig.getDouble(i, j), transposei.getDouble(j, i), 1e-1);
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
    public void testSum2(Nd4jBackend backend) {
        INDArray test = Nd4j.create(new float[] {1, 2, 3, 4}, new long[] {2, 2}).castTo(DataType.DOUBLE);
        INDArray sum = test.sum(1);
        INDArray assertion = Nd4j.create(new float[] {3, 7}).castTo(DataType.DOUBLE);
        assertEquals(assertion, sum);
        INDArray sum0 = Nd4j.create(new float[] {4, 6}).castTo(DataType.DOUBLE);
        assertEquals(sum0, test.sum(0));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetIntervalEdgeCase(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int[] shape = {3, 2, 4};
        INDArray arr3d = Nd4j.rand(shape).castTo(DataType.DOUBLE);

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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetIntervalEdgeCase2(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int[] shape = {3, 2, 4};
        INDArray arr3d = Nd4j.rand(shape).castTo(DataType.DOUBLE);

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
    public void testRowsColumns(Nd4jBackend backend) {
        DataBuffer data = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).data();
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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTranspose(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.ones(100).data(), new long[] {5, 5, 4}).castTo(DataType.DOUBLE);
        INDArray transpose = n.transpose();
        assertEquals(n.length(), transpose.length());
        assertEquals(true, Arrays.equals(new long[] {4, 5, 5}, transpose.shape()));

        INDArray rowVector = Nd4j.linspace(1, 10, 10, DataType.DOUBLE).reshape(1, -1);
        assertTrue(rowVector.isRowVector());
        INDArray columnVector = rowVector.transpose();
        assertTrue(columnVector.isColumnVector());


        INDArray linspaced = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray transposed = Nd4j.create(new double[] {1, 3, 2, 4}, new long[] {2, 2});
        INDArray linSpacedT = linspaced.transpose();
        assertEquals(transposed, linSpacedT);



    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLogX1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(10).assign(7).castTo(DataType.DOUBLE);

        INDArray logX5 = Transforms.log(x, 5, true);

        INDArray exp = Transforms.log(x, true).div(Transforms.log(Nd4j.create(10).assign(5)));

        assertEquals(exp, logX5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddMatrix(Nd4jBackend backend) {
        INDArray five = Nd4j.ones(5).castTo(DataType.DOUBLE);
        five.addi(five);
        INDArray twos = Nd4j.valueArrayOf(5, 2);
        assertEquals(twos, five);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutSlice(Nd4jBackend backend) {
        INDArray n = Nd4j.linspace(1, 27, 27, DataType.DOUBLE).reshape(3, 3, 3);
        INDArray newSlice = Nd4j.zeros(3, 3);
        n.putSlice(0, newSlice);
        assertEquals(newSlice, n.slice(0));
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowVectorMultipleIndices(Nd4jBackend backend) {
        INDArray linear = Nd4j.create(1, 4).castTo(DataType.DOUBLE);
        linear.putScalar(new long[] {0, 1}, 1);
        assertEquals(linear.getDouble(0, 1), 1, 1e-1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSize(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class,() -> {
            INDArray arr = Nd4j.create(4, 5).castTo(DataType.DOUBLE);

            for (int i = 0; i < 6; i++) {
                //This should fail for i >= 2, but doesn't
//            System.out.println(arr.size(i));
                arr.size(i);
            }
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNullPointerDataBuffer(Nd4jBackend backend) {
        ByteBuffer allocate = ByteBuffer.allocateDirect(10 * 4).order(ByteOrder.nativeOrder());
        allocate.asFloatBuffer().put(new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        DataBuffer buff = Nd4j.createBuffer(allocate, DataType.FLOAT, 10);
        float sum = Nd4j.create(buff).sumNumber().floatValue();
//        System.out.println(sum);
        assertEquals(55f, sum, 0.001f);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEps(Nd4jBackend backend) {
        INDArray ones = Nd4j.ones(5);
        val res = Nd4j.create(DataType.BOOL, 5);
        Nd4j.getExecutioner().exec(new Eps(ones, ones, res));

//        log.info("Result: {}", res);
        assertTrue(res.all());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEps2(Nd4jBackend backend) {

        INDArray first = Nd4j.valueArrayOf(10, 1e-6).castTo(DataType.DOUBLE); //0.01
        INDArray second = Nd4j.zeros(10).castTo(DataType.DOUBLE); //0.0

        INDArray expAllZeros1 = Nd4j.getExecutioner().exec(new Eps(first, second, Nd4j.create(DataType.BOOL, new long[] {1, 10}, 'f')));
        INDArray expAllZeros2 = Nd4j.getExecutioner().exec(new Eps(second, first, Nd4j.create(DataType.BOOL, new long[] {1, 10}, 'f')));

        assertTrue(expAllZeros1.all());
        assertTrue(expAllZeros2.all());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLogDouble(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 6, 6, DataType.DOUBLE);
        INDArray log = Transforms.log(linspace);
        INDArray assertion = Nd4j.create(new double[] {0, 0.6931471805599453, 1.0986122886681098, 1.3862943611198906,
                1.6094379124341005, 1.791759469228055});
        assertEquals(assertion, log);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDupDimension(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        assertEquals(arr.tensorAlongDimension(0, 1), arr.tensorAlongDimension(0, 1));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIterator(Nd4jBackend backend) {
        INDArray x = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray repeated = x.repeat(1, 2);
        assertEquals(8, repeated.length());
        Iterator<Double> arrayIter = new INDArrayIterator(x);
        double[] vals = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data().asDouble();
        for (int i = 0; i < vals.length; i++)
            assertEquals(vals[i], arrayIter.next().doubleValue(), 1e-1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTile(Nd4jBackend backend) {
        INDArray x = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray repeated = x.repeat(0, 2);
        assertEquals(8, repeated.length());
        INDArray repeatAlongDimension = x.repeat(1, new long[] {2});
        INDArray assertionRepeat = Nd4j.create(new double[][] {{1, 1, 2, 2}, {3, 3, 4, 4}});
        assertArrayEquals(new long[] {2, 4}, assertionRepeat.shape());
        assertEquals(assertionRepeat, repeatAlongDimension);
//        System.out.println(repeatAlongDimension);
        INDArray ret = Nd4j.create(new double[] {0, 1, 2}).reshape(1, 3);
        INDArray tile = Nd4j.tile(ret, 2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{0, 1, 2, 0, 1, 2}, {0, 1, 2, 0, 1, 2}});
        assertEquals(assertion, tile);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNegativeOneReshape(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[] {0, 1, 2});
        INDArray newShape = arr.reshape(-1);
        assertEquals(newShape, arr);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSmallSum(Nd4jBackend backend) {
        INDArray base = Nd4j.create(new double[] {5.843333333333335, 3.0540000000000007});
        base.addi(1e-12);
        INDArray assertion = Nd4j.create(new double[] {5.84333433, 3.054001});
        assertEquals(assertion, base);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test2DArraySlice(Nd4jBackend backend) {
        INDArray array2D = Nd4j.ones(5, 7).castTo(DataType.DOUBLE);
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
            assertArrayEquals(slice.shape(), new long[] {5});
        }

        for (int i = 0; i < 5; i++) {
            INDArray slice = array2D.slice(i, 0);
            assertArrayEquals(slice.shape(), new long[]{7});
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testTensorDot(Nd4jBackend backend) {
        INDArray oneThroughSixty = Nd4j.arange(60).reshape(3, 4, 5).castTo(DataType.DOUBLE);
        INDArray oneThroughTwentyFour = Nd4j.arange(24).reshape(4, 3, 2).castTo(DataType.DOUBLE);
        INDArray result = Nd4j.tensorMmul(oneThroughSixty, oneThroughTwentyFour, new int[][] {{1, 0}, {0, 1}});
        assertArrayEquals(new long[] {5, 2}, result.shape());
        INDArray assertion = Nd4j.create(new double[][] {{4400, 4730}, {4532, 4874}, {4664, 5018}, {4796, 5162}, {4928, 5306}});
        assertEquals(assertion, result);

        INDArray w = Nd4j.valueArrayOf(new long[] {2, 1, 2, 2}, 0.5);
        INDArray col = Nd4j.create(new double[] {1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3,
                1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4,
                2, 2, 2, 2, 4, 4, 4, 4}, new long[] {1, 1, 2, 2, 4, 4});

        INDArray test = Nd4j.tensorMmul(col, w, new int[][] {{1, 2, 3}, {1, 2, 3}});
        INDArray assertion2 = Nd4j.create(
                new double[] {3., 3., 3., 3., 3., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7., 3., 3., 3., 3.,
                        3., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7.},
                new long[] {1, 4, 4, 2}, new long[] {16, 8, 2, 1}, 'f', DataType.DOUBLE);

        assertEquals(assertion2, test);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetRow(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(10, 4).castTo(DataType.DOUBLE);
        for (int i = 0; i < 10; i++) {
            INDArray row = arr.getRow(i);
            assertArrayEquals(row.shape(), new long[] {4});
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetPermuteReshapeSub(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray first = Nd4j.rand(new long[] {10, 4}).castTo(DataType.DOUBLE);

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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutAtIntervalIndexWithStride(Nd4jBackend backend) {
        INDArray n1 = Nd4j.create(3, 3).assign(0.0).castTo(DataType.DOUBLE);
        INDArrayIndex[] indices = {NDArrayIndex.interval(0, 2, 3), NDArrayIndex.all()};
        n1.put(indices, 1);
        INDArray expected = Nd4j.create(new double[][] {{1d, 1d, 1d}, {0d, 0d, 0d}, {1d, 1d, 1d}});
        assertEquals(expected, n1);
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
    public void testFTimesCAddiRow(Nd4jBackend backend) {

        INDArray arrF = Nd4j.create(2, 3, 'f').assign(1.0).castTo(DataType.DOUBLE);
        INDArray arrC = Nd4j.create(2, 3, 'c').assign(1.0).castTo(DataType.DOUBLE);
        INDArray arr2 = Nd4j.create(new long[] {3, 4}, 'c').assign(1.0).castTo(DataType.DOUBLE);

        INDArray mmulC = arrC.mmul(arr2); //[2,4] with elements 3.0
        INDArray mmulF = arrF.mmul(arr2); //[2,4] with elements 3.0
        assertArrayEquals(mmulC.shape(), new long[] {2, 4});
        assertArrayEquals(mmulF.shape(), new long[] {2, 4});
        assertTrue(arrC.equals(arrF));

        INDArray row = Nd4j.zeros(1, 4).assign(0.0).addi(0.5).castTo(DataType.DOUBLE);
        mmulC.addiRowVector(row); //OK
        mmulF.addiRowVector(row); //Exception

        assertTrue(mmulC.equals(mmulF));

        for (int i = 0; i < mmulC.length(); i++)
            assertEquals(mmulC.getDouble(i), 3.5, 1e-1); //OK
        for (int i = 0; i < mmulF.length(); i++)
            assertEquals(mmulF.getDouble(i), 3.5, 1e-1); //Exception
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
    public void testPermute(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 20, 20, DataType.DOUBLE).data(), new long[] {5, 4}).castTo(DataType.DOUBLE);
        INDArray transpose = n.transpose();
        INDArray permute = n.permute(1, 0);
        assertEquals(permute, transpose);
        assertEquals(transpose.length(), permute.length(), 1e-1);


        INDArray toPermute = Nd4j.create(Nd4j.linspace(0, 7, 8, DataType.DOUBLE).data(), new long[] {2, 2, 2});
        INDArray permuted = toPermute.permute(2, 1, 0);
        INDArray assertion = Nd4j.create(new double[] {0, 4, 2, 6, 1, 5, 3, 7}, new long[] {2, 2, 2});
        assertEquals(permuted, assertion);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermutei(Nd4jBackend backend) {
        //Check in-place permute vs. copy array permute

        //2d:
        INDArray orig = Nd4j.linspace(1, 3 * 4, 3 * 4, DataType.DOUBLE).reshape('c', 3, 4).castTo(DataType.DOUBLE);
        INDArray exp01 = orig.permute(0, 1);
        INDArray exp10 = orig.permute(1, 0);
        List<Pair<INDArray, String>> list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(3, 4, 12345, DataType.DOUBLE);
        List<Pair<INDArray, String>> list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(3, 4, 12345, DataType.DOUBLE);
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
        orig = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape('c', 1, 4);
        exp01 = orig.permute(0, 1);
        exp10 = orig.permute(1, 0);
        list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(1, 4, 12345, DataType.DOUBLE);
        list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(1, 4, 12345, DataType.DOUBLE);
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
        INDArray orig3d = Nd4j.linspace(1, 3 * 4 * 5, 3 * 4 * 5, DataType.DOUBLE).reshape('c', 3, 4, 5);
        INDArray exp012 = orig3d.permute(0, 1, 2);
        INDArray exp021 = orig3d.permute(0, 2, 1);
        INDArray exp120 = orig3d.permute(1, 2, 0);
        INDArray exp102 = orig3d.permute(1, 0, 2);
        INDArray exp201 = orig3d.permute(2, 0, 1);
        INDArray exp210 = orig3d.permute(2, 1, 0);

        List<Pair<INDArray, String>> list012 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);
        List<Pair<INDArray, String>> list021 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);
        List<Pair<INDArray, String>> list120 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);
        List<Pair<INDArray, String>> list102 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);
        List<Pair<INDArray, String>> list201 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);
        List<Pair<INDArray, String>> list210 = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, new long[]{3, 4, 5}, DataType.DOUBLE);

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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermuteiShape(Nd4jBackend backend) {

        INDArray row = Nd4j.create(1, 10).castTo(DataType.DOUBLE);

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



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSwapAxes(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(0, 7, 8, DataType.DOUBLE).data(), new long[] {2, 2, 2});
        INDArray assertion = n.permute(2, 1, 0);
        INDArray permuteTranspose = assertion.slice(1).slice(1);
        INDArray validate = Nd4j.create(new double[] {0, 4, 2, 6, 1, 5, 3, 7}, new long[] {2, 2, 2});
        assertEquals(validate, assertion);

        INDArray thirty = Nd4j.linspace(1, 30, 30, DataType.DOUBLE).reshape(3, 5, 2);
        INDArray swapped = thirty.swapAxes(2, 1);
        INDArray slice = swapped.slice(0).slice(0);
        INDArray assertion2 = Nd4j.create(new double[] {1, 3, 5, 7, 9});
        assertEquals(assertion2, slice);


    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMuliRowVector(Nd4jBackend backend) {
        INDArray arrC = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape('c', 3, 2);
        INDArray arrF = Nd4j.create(new long[] {3, 2}, 'f').assign(arrC);

        INDArray temp = Nd4j.create(new long[] {2, 11}, 'c');
        INDArray vec = temp.get(NDArrayIndex.all(), NDArrayIndex.interval(9, 10)).transpose();
        vec.assign(Nd4j.linspace(1, 2, 2, DataType.DOUBLE));

        //Passes if we do one of these...
        //        vec = vec.dup('c');
        //        vec = vec.dup('f');

//        System.out.println("Vec: " + vec);

        INDArray outC = arrC.muliRowVector(vec);
        INDArray outF = arrF.muliRowVector(vec);

        double[][] expD = new double[][] {{1, 4}, {3, 8}, {5, 12}};
        INDArray exp = Nd4j.create(expD);

        assertEquals(exp, outC);
        assertEquals(exp, outF);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSliceConstructor(Nd4jBackend backend) {
        List<INDArray> testList = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            testList.add(Nd4j.scalar(i + 1.0f));

        INDArray test = Nd4j.create(testList, new long[] {1, testList.size()}).reshape(1, 5);
        INDArray expected = Nd4j.create(new float[] {1, 2, 3, 4, 5}, new long[] {1, 5});
        assertEquals(expected, test);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStdev0(Nd4jBackend backend) {
        double[][] ind = {{5.1, 3.5, 1.4}, {4.9, 3.0, 1.4}, {4.7, 3.2, 1.3}};
        INDArray in = Nd4j.create(ind);
        INDArray stdev = in.std(0);
        INDArray exp = Nd4j.create(new double[] {0.19999999999999973, 0.2516611478423583, 0.057735026918962505});

        assertEquals(exp, stdev);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStdev1(Nd4jBackend backend) {
        double[][] ind = {{5.1, 3.5, 1.4}, {4.9, 3.0, 1.4}, {4.7, 3.2, 1.3}};
        INDArray in = Nd4j.create(ind).castTo(DataType.DOUBLE);
        INDArray stdev = in.std(1);
//        log.info("StdDev: {}", stdev.toDoubleVector());
        INDArray exp = Nd4j.create(new double[] {1.8556220879622372, 1.7521415467935233, 1.7039170558842744});
        assertEquals(exp, stdev);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSignXZ(Nd4jBackend backend) {
        double[] d = {1.0, -1.1, 1.2, 1.3, -1.4, -1.5, 1.6, -1.7, -1.8, -1.9, -1.01, -1.011};
        double[] e = {1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

        INDArray arrF = Nd4j.create(d, new long[] {4, 3}, 'f').castTo(DataType.DOUBLE);
        INDArray arrC = Nd4j.create(new long[] {4, 3}, 'c').assign(arrF).castTo(DataType.DOUBLE);

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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTanhXZ(Nd4jBackend backend) {
        INDArray arrC = Nd4j.linspace(-6, 6, 12, DataType.DOUBLE).reshape('c', 4, 3);
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastDiv(Nd4jBackend backend) {
        INDArray num = Nd4j.create(new double[] {1.00, 1.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 1.00, 1.00, 1.00, 1.00,
                2.00, 2.00, 2.00, 2.00, -1.00, -1.00, -1.00, -1.00, -2.00, -2.00, -2.00, -2.00, -1.00, -1.00,
                -1.00, -1.00, -2.00, -2.00, -2.00, -2.00}).reshape(2, 16);

        INDArray denom = Nd4j.create(new double[] {1.00, 1.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 1.00, 1.00, 1.00,
                1.00, 2.00, 2.00, 2.00, 2.00});

        INDArray expected = Nd4j.create(
                new double[] {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., -1., -1.,
                        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,},
                new long[] {2, 16});

        INDArray actual = Nd4j.getExecutioner().exec(new BroadcastDivOp(num, denom, num.dup(), -1));
        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastDiv2(){
        INDArray arr = Nd4j.ones(DataType.DOUBLE, 1, 64, 125, 125).muli(2);
        INDArray vec = Nd4j.ones(DataType.DOUBLE, 64).muli(2);

        INDArray exp = Nd4j.ones(DataType.DOUBLE, 1, 64, 125, 125);
        INDArray out = arr.like();

        for( int i=0; i<10; i++ ) {
            out.assign(0.0);
            Nd4j.getExecutioner().exec(new BroadcastDivOp(arr, vec, out, 1));
            assertEquals(exp, out);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastMult(Nd4jBackend backend) {
        INDArray num = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, -1.00, -2.00, -3.00,
                -4.00, -5.00, -6.00, -7.00, -8.00}).reshape(2, 8);

        INDArray denom = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00});

        INDArray expected = Nd4j.create(new double[] {1, 4, 9, 16, 25, 36, 49, 64, -1, -4, -9, -16, -25, -36, -49, -64},
                new long[] {2, 8});

        INDArray actual = Nd4j.getExecutioner().exec(new BroadcastMulOp(num, denom, num.dup(), -1));
        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastSub(Nd4jBackend backend) {
        INDArray num = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, -1.00, -2.00, -3.00,
                -4.00, -5.00, -6.00, -7.00, -8.00}).reshape(2, 8);

        INDArray denom = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00});

        INDArray expected = Nd4j.create(new double[] {0, 0, 0, 0, 0, 0, 0, 0, -2, -4, -6, -8, -10, -12, -14, -16},
                new long[] {2, 8});

        INDArray actual = Nd4j.getExecutioner().exec(new BroadcastSubOp(num, denom, num.dup(), -1));
        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastAdd(Nd4jBackend backend) {
        INDArray num = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, -1.00, -2.00, -3.00,
                -4.00, -5.00, -6.00, -7.00, -8.00}).reshape(2, 8);

        INDArray denom = Nd4j.create(new double[] {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00});

        INDArray expected = Nd4j.create(new double[] {2, 4, 6, 8, 10, 12, 14, 16, 0, 0, 0, 0, 0, 0, 0, 0,},
                new long[] {2, 8});
        INDArray dup = num.dup();
        INDArray actual = Nd4j.getExecutioner().exec(new BroadcastAddOp(num, denom, dup, -1));
        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDimension(Nd4jBackend backend) {
        INDArray test = Nd4j.create(Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data(), new long[] {2, 2});
        //row
        INDArray slice0 = test.slice(0, 1);
        INDArray slice02 = test.slice(1, 1);

        INDArray assertSlice0 = Nd4j.create(new double[] {1, 3});
        INDArray assertSlice02 = Nd4j.create(new double[] {2, 4});
        assertEquals(assertSlice0, slice0);
        assertEquals(assertSlice02, slice02);

        //column
        INDArray assertSlice1 = Nd4j.create(new double[] {1, 2});
        INDArray assertSlice12 = Nd4j.create(new double[] {3, 4});


        INDArray slice1 = test.slice(0, 0);
        INDArray slice12 = test.slice(1, 0);


        assertEquals(assertSlice1, slice1);
        assertEquals(assertSlice12, slice12);


        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24, DataType.DOUBLE).data(), new long[] {4, 3, 2});
        INDArray secondSliceFirstDimension = arr.slice(1, 1);
        assertEquals(secondSliceFirstDimension, secondSliceFirstDimension);


    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshape(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24, DataType.DOUBLE).data(), new long[] {4, 3, 2});
        INDArray reshaped = arr.reshape(2, 3, 4);
        assertEquals(arr.length(), reshaped.length());
        assertEquals(true, Arrays.equals(new long[] {4, 3, 2}, arr.shape()));
        assertEquals(true, Arrays.equals(new long[] {2, 3, 4}, reshaped.shape()));

    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDot() throws Exception {
        INDArray vec1 = Nd4j.create(new float[] {1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new float[] {1, 2, 3, 4});

        assertEquals(10.f, vec1.sumNumber().floatValue(), 1e-5);
        assertEquals(10.f, vec2.sumNumber().floatValue(), 1e-5);

        assertEquals(30, Nd4j.getBlasWrapper().dot(vec1, vec2), 1e-1);

        INDArray matrix = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray row = matrix.getRow(1);

        assertEquals(7.0f, row.sumNumber().floatValue(), 1e-5f);

        assertEquals(25, Nd4j.getBlasWrapper().dot(row, row), 1e-1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIdentity(Nd4jBackend backend) {
        INDArray eye = Nd4j.eye(5);
        assertTrue(Arrays.equals(new long[] {5, 5}, eye.shape()));
        eye = Nd4j.eye(5);
        assertTrue(Arrays.equals(new long[] {5, 5}, eye.shape()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTemp(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(new long[] {2, 2, 2}).castTo(DataType.DOUBLE);
//        System.out.println("In:\n" + in);
        INDArray permuted = in.permute(0, 2, 1); //Permute, so we get correct order after reshaping
        INDArray out = permuted.reshape(4, 2);
//        System.out.println("Out:\n" + out);

        int countZero = 0;
        for (int i = 0; i < 8; i++)
            if (out.getDouble(i) == 0.0)
                countZero++;
        assertEquals(countZero, 0);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeans(Nd4jBackend backend) {
        INDArray a = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray mean1 = a.mean(1);
        assertEquals(Nd4j.create(new double[] {1.5, 3.5}), mean1,getFailureMessage(backend));
        assertEquals(Nd4j.create(new double[] {2, 3}), a.mean(0),getFailureMessage(backend));
        assertEquals(2.5, Nd4j.linspace(1, 4, 4, DataType.DOUBLE).meanNumber().doubleValue(), 1e-1,getFailureMessage(backend));
        assertEquals(2.5, a.meanNumber().doubleValue(), 1e-1,getFailureMessage(backend));

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSums(Nd4jBackend backend) {
        INDArray a = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        assertEquals(Nd4j.create(new double[] {3, 7}), a.sum(1),getFailureMessage(backend));
        assertEquals(Nd4j.create(new double[] {4, 6}), a.sum(0),getFailureMessage(backend));
        assertEquals(10, a.sumNumber().doubleValue(), 1e-1,getFailureMessage(backend));


    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRSubi(Nd4jBackend backend) {
        INDArray n2 = Nd4j.ones(2);
        INDArray n2Assertion = Nd4j.zeros(2);
        INDArray nRsubi = n2.rsubi(1);
        assertEquals(n2Assertion, nRsubi);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcat(Nd4jBackend backend) {
        INDArray A = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape(2, 2, 2);
        INDArray B = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 2, 2);
        INDArray concat = Nd4j.concat(0, A, B);
        assertTrue(Arrays.equals(new long[] {5, 2, 2}, concat.shape()));

        INDArray columnConcat = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
        INDArray concatWith = Nd4j.zeros(2, 3);
        INDArray columnWiseConcat = Nd4j.concat(0, columnConcat, concatWith);
//        System.out.println(columnConcat);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcatHorizontally(Nd4jBackend backend) {
        INDArray rowVector = Nd4j.ones(1, 5);
        INDArray other = Nd4j.ones(1, 5);
        INDArray concat = Nd4j.hstack(other, rowVector);
        assertEquals(rowVector.rows(), concat.rows());
        assertEquals(rowVector.columns() * 2, concat.columns());
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArgMaxSameValues(Nd4jBackend backend) {
        //Here: assume that by convention, argmax returns the index of the FIRST maximum value
        //Thus, argmax(ones(...)) = 0 by convention
        INDArray arr = Nd4j.ones(DataType.DOUBLE,1,10);

        for (int i = 0; i < 10; i++) {
            double argmax = Nd4j.argMax(arr, 1).getDouble(0);
            //System.out.println(argmax);
            assertEquals(0.0, argmax, 0.0);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSoftmaxStability(Nd4jBackend backend) {
        INDArray input = Nd4j.create(new double[] {-0.75, 0.58, 0.42, 1.03, -0.61, 0.19, -0.37, -0.40, -1.42, -0.04}).reshape(1, -1).transpose();
//        System.out.println("Input transpose " + Shape.shapeToString(input.shapeInfo()));
        INDArray output = Nd4j.create(10, 1);
//        System.out.println("Element wise stride of output " + output.elementWiseStride());
        Nd4j.getExecutioner().exec(new SoftMax(input, output));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssignOffset(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(5, 5).castTo(DataType.DOUBLE);
        INDArray row = arr.slice(1);
        row.assign(1);
        assertEquals(Nd4j.ones(5).castTo(DataType.DOUBLE), row);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddScalar(Nd4jBackend backend) {
        INDArray div = Nd4j.valueArrayOf(new long[] {1, 4}, 4);
        INDArray rdiv = div.add(1);
        INDArray answer = Nd4j.valueArrayOf(new long[] {1, 4}, 5);
        assertEquals(answer, rdiv);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRdivScalar(Nd4jBackend backend) {
        INDArray div = Nd4j.valueArrayOf(new long[] {1, 4}, 4).castTo(DataType.DOUBLE);
        INDArray rdiv = div.rdiv(1);
        INDArray answer = Nd4j.valueArrayOf(new long[] {1, 4}, 0.25).castTo(DataType.DOUBLE);
        assertEquals(rdiv, answer);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRDivi(Nd4jBackend backend) {
        INDArray n2 = Nd4j.valueArrayOf(new long[] {1, 2}, 4).castTo(DataType.DOUBLE);
        INDArray n2Assertion = Nd4j.valueArrayOf(new long[] {1, 2}, 0.5).castTo(DataType.DOUBLE);
        INDArray nRsubi = n2.rdivi(2);
        assertEquals(n2Assertion, nRsubi);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testElementWiseAdd(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray linspace2 = linspace.dup();
        INDArray assertion = Nd4j.create(new double[][] {{2, 4}, {6, 8}});
        linspace.addi(linspace2);
        assertEquals(assertion, linspace);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSquareMatrix(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8, DataType.DOUBLE).data(), new long[] {2, 2, 2});
        INDArray eightFirstTest = n.vectorAlongDimension(0, 2);
        INDArray eightFirstAssertion = Nd4j.create(new double[] {1, 2});
        assertEquals(eightFirstAssertion, eightFirstTest);

        INDArray eightFirstTestSecond = n.vectorAlongDimension(1, 2);
        INDArray eightFirstTestSecondAssertion = Nd4j.create(new double[] {3, 4});
        assertEquals(eightFirstTestSecondAssertion, eightFirstTestSecond);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNumVectorsAlongDimension(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape(4, 3, 2);
        assertEquals(12, arr.vectorsAlongDimension(2));
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadCast(Nd4jBackend backend) {
        INDArray n = Nd4j.linspace(1, 4, 4, DataType.DOUBLE);
        INDArray broadCasted = n.broadcast(5, 4);
        for (int i = 0; i < broadCasted.rows(); i++) {
            INDArray row = broadCasted.getRow(i);
            assertEquals(n, broadCasted.getRow(i));
        }

        INDArray broadCast2 = broadCasted.getRow(0).broadcast(5, 4);
        assertEquals(broadCasted, broadCast2);


        INDArray columnBroadcast = n.reshape(4,1).broadcast(4, 5);
        for (int i = 0; i < columnBroadcast.columns(); i++) {
            INDArray column = columnBroadcast.getColumn(i);
            assertEquals(column, n);
        }

        INDArray fourD = Nd4j.create(1, 2, 1, 1);
        INDArray broadCasted3 = fourD.broadcast(1, 2, 36, 36);
        assertTrue(Arrays.equals(new long[] {1, 2, 36, 36}, broadCasted3.shape()));



        INDArray ones = Nd4j.ones(1, 1, 1).broadcast(2, 1, 1);
        assertArrayEquals(new long[] {2, 1, 1}, ones.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarBroadcast(Nd4jBackend backend) {
        INDArray fiveThree = Nd4j.ones(5, 3);
        INDArray fiveThreeTest = Nd4j.scalar(1.0).broadcast(5, 3);
        assertEquals(fiveThree, fiveThreeTest);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutRowGetRowOrdering(Nd4jBackend backend) {
        INDArray row1 = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray put = Nd4j.create(new double[] {5, 6});
        row1.putRow(1, put);


        INDArray row1Fortran = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray putFortran = Nd4j.create(new double[] {5, 6});
        row1Fortran.putRow(1, putFortran);
        assertEquals(row1, row1Fortran);
        INDArray row1CTest = row1.getRow(1);
        INDArray row1FortranTest = row1Fortran.getRow(1);
        assertEquals(row1CTest, row1FortranTest);



    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testElementWiseOps(Nd4jBackend backend) {
        INDArray n1 = Nd4j.scalar(1.0);
        INDArray n2 = Nd4j.scalar(2.0);
        INDArray nClone = n1.add(n2);
        assertEquals(Nd4j.scalar(3.0), nClone);
        assertFalse(n1.add(n2).equals(n1));

        INDArray n3 = Nd4j.scalar(3.0);
        INDArray n4 = Nd4j.scalar(4.0);
        INDArray subbed = n4.sub(n3);
        INDArray mulled = n4.mul(n3);
        INDArray div = n4.div(n3);

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));
        assertEquals(Nd4j.scalar(1.0), subbed);
        assertEquals(Nd4j.scalar(12.0), mulled);
        assertEquals(Nd4j.scalar(1.333333333333333333333), div);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNdArrayCreation(Nd4jBackend backend) {
        double delta = 1e-1;
        INDArray n1 = Nd4j.create(new double[] {0d, 1d, 2d, 3d}, new long[] {2, 2}, 'c');
        INDArray lv = n1.reshape(-1);
        assertEquals(0d, lv.getDouble(0), delta);
        assertEquals(1d, lv.getDouble(1), delta);
        assertEquals(2d, lv.getDouble(2), delta);
        assertEquals(3d, lv.getDouble(3), delta);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFlattenedWithOrder(Nd4jBackend backend) {
        int[] firstShape = {10, 3};
        int firstLen = ArrayUtil.prod(firstShape);
        int[] secondShape = {2, 7};
        int secondLen = ArrayUtil.prod(secondShape);
        int[] thirdShape = {3, 3};
        int thirdLen = ArrayUtil.prod(thirdShape);
        INDArray firstC = Nd4j.linspace(1, firstLen, firstLen, DataType.DOUBLE).reshape('c', firstShape);
        INDArray firstF = Nd4j.create(firstShape, 'f').assign(firstC);
        INDArray secondC = Nd4j.linspace(1, secondLen, secondLen, DataType.DOUBLE).reshape('c', secondShape);
        INDArray secondF = Nd4j.create(secondShape, 'f').assign(secondC);
        INDArray thirdC = Nd4j.linspace(1, thirdLen, thirdLen, DataType.DOUBLE).reshape('c', thirdShape);
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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLeakyRelu(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(-1, 1, 10, DataType.DOUBLE);
        double[] expected = new double[10];
        for (int i = 0; i < 10; i++) {
            double in = arr.getDouble(i);
            expected[i] = (in <= 0.0 ? 0.01 * in : in);
        }

        INDArray out = Nd4j.getExecutioner().exec(new LeakyReLU(arr, 0.01));

        INDArray exp = Nd4j.create(expected);
        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSoftmaxRow(Nd4jBackend backend) {
        for (int i = 0; i < 20; i++) {
            INDArray arr1 = Nd4j.zeros(1, 100);
            Nd4j.getExecutioner().execAndReturn(new SoftMax(arr1));
//            System.out.println(Arrays.toString(arr1.data().asFloat()));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLeakyRelu2(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(-1, 1, 10, DataType.DOUBLE);
        double[] expected = new double[10];
        for (int i = 0; i < 10; i++) {
            double in = arr.getDouble(i);
            expected[i] = (in <= 0.0 ? 0.01 * in : in);
        }

        INDArray out = Nd4j.getExecutioner().exec(new LeakyReLU(arr, 0.01));

//        System.out.println("Expected: " + Arrays.toString(expected));
//        System.out.println("Actual:   " + Arrays.toString(out.data().asDouble()));

        INDArray exp = Nd4j.create(expected);
        assertEquals(exp, out);
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
            assertEquals(in, dupc,msg);
            assertEquals(in, dupf,msg);
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

            assertEquals(in, dup,msg);
            assertEquals(in, dupc,msg);
            assertEquals(in, dupf,msg);
            assertEquals(dupc.ordering(), 'c',msg);
            assertEquals(dupf.ordering(), 'f',msg);
            assertEquals(in, dupany,msg);

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
    @Disabled
    public void largeInstantiation(Nd4jBackend backend) {
        Nd4j.ones((1024 * 1024 * 511) + (1024 * 1024 - 1)); // Still works; this can even be called as often as I want, allowing me even to spill over on disk
        Nd4j.ones((1024 * 1024 * 511) + (1024 * 1024)); // Crashes
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssignNumber(Nd4jBackend backend) {
        int nRows = 10;
        int nCols = 20;
        INDArray in = Nd4j.linspace(1, nRows * nCols, nRows * nCols, DataType.DOUBLE).reshape('c', new long[] {nRows, nCols});

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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumDifferentOrdersSquareMatrix(Nd4jBackend backend) {
        INDArray arrc = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray arrf = Nd4j.create(new long[] {2, 2}, 'f').assign(arrc);

        INDArray cSum = arrc.sum(0);
        INDArray fSum = arrf.sum(0);
        assertEquals(arrc, arrf);
        assertEquals(cSum, fSum); //Expect: 4,6. Getting [4, 4] for f order
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssignMixedC(Nd4jBackend backend) {
        int[] shape1 = {3, 2, 2, 2, 2, 2};
        int[] shape2 = {12, 8};
        int length = ArrayUtil.prod(shape1);

        assertEquals(ArrayUtil.prod(shape1), ArrayUtil.prod(shape2));

        INDArray arr = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape1);
        INDArray arr2c = Nd4j.create(shape2, 'c');
        INDArray arr2f = Nd4j.create(shape2, 'f');

//        log.info("2f data: {}", Arrays.toString(arr2f.data().asFloat()));

        arr2c.assign(arr);
//        System.out.println("--------------");
        arr2f.assign(arr);

        INDArray exp = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2);

//        log.info("arr data: {}", Arrays.toString(arr.data().asFloat()));
//        log.info("2c data: {}", Arrays.toString(arr2c.data().asFloat()));
//        log.info("2f data: {}", Arrays.toString(arr2f.data().asFloat()));
//        log.info("2c shape: {}", Arrays.toString(arr2c.shapeInfoDataBuffer().asInt()));
//        log.info("2f shape: {}", Arrays.toString(arr2f.shapeInfoDataBuffer().asInt()));
        assertEquals(exp, arr2c);
        assertEquals(exp, arr2f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDummy(Nd4jBackend backend) {
        INDArray arr2f = Nd4j.create(new double[] {1.0, 13.0, 25.0, 37.0, 49.0, 61.0, 73.0, 85.0, 2.0, 14.0, 26.0, 38.0,
                50.0, 62.0, 74.0, 86.0, 3.0, 15.0, 27.0, 39.0, 51.0, 63.0, 75.0, 87.0, 4.0, 16.0, 28.0, 40.0,
                52.0, 64.0, 76.0, 88.0, 5.0, 17.0, 29.0, 41.0, 53.0, 65.0, 77.0, 89.0, 6.0, 18.0, 30.0, 42.0,
                54.0, 66.0, 78.0, 90.0, 7.0, 19.0, 31.0, 43.0, 55.0, 67.0, 79.0, 91.0, 8.0, 20.0, 32.0, 44.0,
                56.0, 68.0, 80.0, 92.0, 9.0, 21.0, 33.0, 45.0, 57.0, 69.0, 81.0, 93.0, 10.0, 22.0, 34.0, 46.0,
                58.0, 70.0, 82.0, 94.0, 11.0, 23.0, 35.0, 47.0, 59.0, 71.0, 83.0, 95.0, 12.0, 24.0, 36.0, 48.0,
                60.0, 72.0, 84.0, 96.0}, new long[] {12, 8}, 'f');
//        log.info("arr2f shape: {}", Arrays.toString(arr2f.shapeInfoDataBuffer().asInt()));
//        log.info("arr2f data: {}", Arrays.toString(arr2f.data().asFloat()));
//        log.info("render: {}", arr2f);

//        log.info("----------------------");

        INDArray array = Nd4j.linspace(1, 96, 96, DataType.DOUBLE).reshape('c', 12, 8);
//        log.info("array render: {}", array);

//        log.info("----------------------");

        INDArray arrayf = array.dup('f');
//        log.info("arrayf render: {}", arrayf);
//        log.info("arrayf shape: {}", Arrays.toString(arrayf.shapeInfoDataBuffer().asInt()));
//        log.info("arrayf data: {}", Arrays.toString(arrayf.data().asFloat()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateDetached_1(Nd4jBackend backend) {
        val shape = new int[]{10};
        val dataTypes = new DataType[] {DataType.DOUBLE, DataType.BOOL, DataType.BYTE, DataType.UBYTE, DataType.SHORT, DataType.UINT16, DataType.INT, DataType.UINT32, DataType.LONG, DataType.UINT64, DataType.FLOAT, DataType.BFLOAT16, DataType.HALF};

        for(DataType dt : dataTypes){
            val dataBuffer = Nd4j.createBufferDetached(shape, dt);
            assertEquals(dt, dataBuffer.dataType());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateDetached_2(Nd4jBackend backend) {
        val shape = new long[]{10};
        val dataTypes = new DataType[] {DataType.DOUBLE, DataType.BOOL, DataType.BYTE, DataType.UBYTE, DataType.SHORT, DataType.UINT16, DataType.INT, DataType.UINT32, DataType.LONG, DataType.UINT64, DataType.FLOAT, DataType.BFLOAT16, DataType.HALF};

        for(DataType dt : dataTypes){
            val dataBuffer = Nd4j.createBufferDetached(shape, dt);
            assertEquals(dt, dataBuffer.dataType());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPairwiseMixedC(Nd4jBackend backend) {
        int[] shape2 = {12, 8};
        int length = ArrayUtil.prod(shape2);


        INDArray arr = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2);
        INDArray arr2c = arr.dup('c');
        INDArray arr2f = arr.dup('f');

        arr2c.addi(arr);
//        System.out.println("--------------");
        arr2f.addi(arr);

        INDArray exp = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2).mul(2.0);

        assertEquals(exp, arr2c);
        assertEquals(exp, arr2f);

//        log.info("2c data: {}", Arrays.toString(arr2c.data().asFloat()));
//        log.info("2f data: {}", Arrays.toString(arr2f.data().asFloat()));

        assertTrue(arrayNotEquals(arr2c.data().asFloat(), arr2f.data().asFloat(), 1e-5f));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPairwiseMixedF(Nd4jBackend backend) {
        int[] shape2 = {12, 8};
        int length = ArrayUtil.prod(shape2);


        INDArray arr = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2).dup('f');
        INDArray arr2c = arr.dup('c');
        INDArray arr2f = arr.dup('f');

        arr2c.addi(arr);
//        System.out.println("--------------");
        arr2f.addi(arr);

        INDArray exp = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2).dup('f').mul(2.0);

        assertEquals(exp, arr2c);
        assertEquals(exp, arr2f);

//        log.info("2c data: {}", Arrays.toString(arr2c.data().asFloat()));
//        log.info("2f data: {}", Arrays.toString(arr2f.data().asFloat()));

        assertTrue(arrayNotEquals(arr2c.data().asFloat(), arr2f.data().asFloat(), 1e-5f));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssign2D(Nd4jBackend backend) {
        int[] shape2 = {8, 4};

        int length = ArrayUtil.prod(shape2);

        INDArray arr = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2);
        INDArray arr2c = Nd4j.create(shape2, 'c');
        INDArray arr2f = Nd4j.create(shape2, 'f');

        arr2c.assign(arr);
//        System.out.println("--------------");
        arr2f.assign(arr);

        INDArray exp = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2);

        assertEquals(exp, arr2c);
        assertEquals(exp, arr2f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssign2D_2(Nd4jBackend backend) {
        int[] shape2 = {8, 4};

        int length = ArrayUtil.prod(shape2);

        INDArray arr = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2);
        INDArray arr2c = Nd4j.create(shape2, 'c');
        INDArray arr2f = Nd4j.create(shape2, 'f');
        INDArray z_f = Nd4j.create(shape2, 'f');
        INDArray z_c = Nd4j.create(shape2, 'c');

        Nd4j.getExecutioner().exec(new Set(arr2f, arr, z_f));

        Nd4j.getExecutioner().commit();

        Nd4j.getExecutioner().exec(new Set(arr2f, arr, z_c));

        INDArray exp = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape2);


//        System.out.println("Zf data: " + Arrays.toString(z_f.data().asFloat()));
//        System.out.println("Zc data: " + Arrays.toString(z_c.data().asFloat()));

        assertEquals(exp, z_f);
        assertEquals(exp, z_c);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssign3D_2(Nd4jBackend backend) {
        int[] shape3 = {8, 4, 8};

        int length = ArrayUtil.prod(shape3);

        INDArray arr = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape3).dup('f');
        INDArray arr3c = Nd4j.create(shape3, 'c');
        INDArray arr3f = Nd4j.create(shape3, 'f');

        Nd4j.getExecutioner().exec(new Set(arr3c, arr, arr3f));

        Nd4j.getExecutioner().commit();

        Nd4j.getExecutioner().exec(new Set(arr3f, arr, arr3c));

        INDArray exp = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape3);

        assertEquals(exp, arr3c);
        assertEquals(exp, arr3f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumDifferentOrders(Nd4jBackend backend) {
        INDArray arrc = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape('c', 3, 2);
        INDArray arrf = Nd4j.create(new double[6], new long[] {3, 2}, 'f').assign(arrc);

        assertEquals(arrc, arrf);
        INDArray cSum = arrc.sum(0);
        INDArray fSum = arrf.sum(0);
        assertEquals(cSum, fSum); //Expect: 0.51, 1.79; getting [0.51,1.71] for f order
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateUnitialized(Nd4jBackend backend) {

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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVarConst(Nd4jBackend backend) {
        INDArray x = Nd4j.linspace(1, 100, 100, DataType.DOUBLE).reshape(10, 10);
//        System.out.println(x);
        assertFalse(Double.isNaN(x.var(0).sumNumber().doubleValue()));
//        System.out.println(x.var(0));
        x.var(0);
        assertFalse(Double.isNaN(x.var(1).sumNumber().doubleValue()));
//        System.out.println(x.var(1));
        x.var(1);

//        System.out.println("=================================");
        // 2d array - all elements are the same
        INDArray a = Nd4j.ones(10, 10).mul(10);
//        System.out.println(a);
        assertFalse(Double.isNaN(a.var(0).sumNumber().doubleValue()));
//        System.out.println(a.var(0));
        a.var(0);
        assertFalse(Double.isNaN(a.var(1).sumNumber().doubleValue()));
//        System.out.println(a.var(1));
        a.var(1);

        // 2d array - constant in one dimension
//        System.out.println("=================================");
        INDArray nums = Nd4j.linspace(1, 10, 10, DataType.DOUBLE);
        INDArray b = Nd4j.ones(10, 10).mulRowVector(nums);
//        System.out.println(b);
        assertFalse(Double.isNaN((Double) b.var(0).sumNumber()));
//        System.out.println(b.var(0));
        b.var(0);
        assertFalse(Double.isNaN((Double) b.var(1).sumNumber()));
//        System.out.println(b.var(1));
        b.var(1);

//        System.out.println("=================================");
//        System.out.println(b.transpose());
        assertFalse(Double.isNaN((Double) b.transpose().var(0).sumNumber()));
//        System.out.println(b.transpose().var(0));
        b.transpose().var(0);
        assertFalse(Double.isNaN((Double) b.transpose().var(1).sumNumber()));
//        System.out.println(b.transpose().var(1));
        b.transpose().var(1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVPull1(Nd4jBackend backend) {
        int indexes[] = new int[] {0, 2, 4};
        INDArray array = Nd4j.linspace(1, 25, 25, DataType.DOUBLE).reshape(5, 5);
        INDArray assertion = Nd4j.createUninitialized(new long[] {3, 5}, 'f');
        for (int i = 0; i < 3; i++) {
            assertion.putRow(i, array.getRow(indexes[i]));
        }

        INDArray result = Nd4j.pullRows(array, 1, indexes, 'f');

        assertEquals(3, result.rows());
        assertEquals(5, result.columns());
        assertEquals(assertion, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsValidation1(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            Nd4j.pullRows(Nd4j.create(10, 10), 2, new int[] {0, 1, 2});

        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsValidation2(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            Nd4j.pullRows(Nd4j.create(10, 10), 1, new int[] {0, -1, 2});

        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsValidation3(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            Nd4j.pullRows(Nd4j.create(10, 10), 1, new int[] {0, 1, 10});

        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsValidation4(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            Nd4j.pullRows(Nd4j.create(3, 10), 1, new int[] {0, 1, 2, 3});

        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsValidation5(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            Nd4j.pullRows(Nd4j.create(3, 10), 1, new int[] {0, 1, 2}, 'e');

        });
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVPull2(Nd4jBackend backend) {
        val indexes = new int[] {0, 2, 4};
        INDArray array = Nd4j.linspace(1, 25, 25, DataType.DOUBLE).reshape(5, 5);
        INDArray assertion = Nd4j.createUninitialized(new long[] {3, 5}, 'c');
        for (int i = 0; i < 3; i++) {
            assertion.putRow(i, array.getRow(indexes[i]));
        }

        INDArray result = Nd4j.pullRows(array, 1, indexes, 'c');

        assertEquals(3, result.rows());
        assertEquals(5, result.columns());
        assertEquals(assertion, result);

//        System.out.println(assertion.toString());
//        System.out.println(result.toString());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompareAndSet1(Nd4jBackend backend) {
        INDArray array = Nd4j.zeros(25);

        INDArray assertion = Nd4j.zeros(25);

        array.putScalar(0, 0.1f);
        array.putScalar(10, 0.1f);
        array.putScalar(20, 0.1f);

        Nd4j.getExecutioner().exec(new CompareAndSet(array, 0.1, 0.0, 0.01));

        assertEquals(assertion, array);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReplaceNaNs(Nd4jBackend backend) {
        INDArray array = Nd4j.zeros(25);
        INDArray assertion = Nd4j.zeros(25);

        array.putScalar(0, Float.NaN);
        array.putScalar(10, Float.NaN);
        array.putScalar(20, Float.NaN);

        assertNotEquals(assertion, array);

        Nd4j.getExecutioner().exec(new ReplaceNans(array, 0.0));

//        System.out.println("Array After: " + array);

        assertEquals(assertion, array);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNaNEquality(Nd4jBackend backend) {
        INDArray array = Nd4j.zeros(25);
        INDArray assertion = Nd4j.zeros(25);

        array.putScalar(0, Float.NaN);
        array.putScalar(10, Float.NaN);
        array.putScalar(20, Float.NaN);

        assertNotEquals(assertion, array);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDistance1and2(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEqualsWithEps1(Nd4jBackend backend) {
        INDArray array1 = Nd4j.create(new double[] {0.5f, 1.5f, 2.5f, 3.5f, 4.5f});
        INDArray array2 = Nd4j.create(new double[] {0f, 1f, 2f, 3f, 4f});
        INDArray array3 = Nd4j.create(new double[] {0f, 1.000001f, 2f, 3f, 4f});


        assertFalse(array1.equalsWithEps(array2, Nd4j.EPS_THRESHOLD));
        assertTrue(array2.equalsWithEps(array3, Nd4j.EPS_THRESHOLD));
        assertTrue(array1.equalsWithEps(array2, 0.7f));
        assertEquals(array2, array3);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIMaxIAMax(Nd4jBackend backend) {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ALL);

        INDArray arr = Nd4j.create(new double[] {-0.24, -0.26, -0.07, -0.01});
        val iMax = new ArgMax(new INDArray[]{arr});
        val iaMax = new ArgAmax(new INDArray[]{arr.dup()});
        val imax = Nd4j.getExecutioner().exec(iMax)[0].getInt(0);
        val iamax = Nd4j.getExecutioner().exec(iaMax)[0].getInt(0);
//        System.out.println("IMAX: " + imax);
//        System.out.println("IAMAX: " + iamax);
        assertEquals(1, iamax);
        assertEquals(3, imax);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIMinIAMin(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[] {-0.24, -0.26, -0.07, -0.01});
        INDArray abs = Transforms.abs(arr);
        val iaMin = new ArgAmin(new INDArray[]{abs});
        val iMin = new ArgMin(new INDArray[]{arr.dup()});
        double imin = Nd4j.getExecutioner().exec(iMin)[0].getDouble(0);
        double iamin = Nd4j.getExecutioner().exec(iaMin)[0].getDouble(0);
//        System.out.println("IMin: " + imin);
//        System.out.println("IAMin: " + iamin);
        assertEquals(3, iamin, 1e-12);
        assertEquals(1, imin, 1e-12);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcast3d2d(Nd4jBackend backend) {
        char[] orders = {'c', 'f'};

        for (char orderArr : orders) {
            for (char orderbc : orders) {
//                System.out.println(orderArr + "\t" + orderbc);
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
                    assertEquals( bc12, subset,"Failed for subset [" + i + "] orders [" + orderArr + "/" + orderbc + "]");
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcast4d2d(Nd4jBackend backend) {
        char[] orders = {'c', 'f'};

        for (char orderArr : orders) {
            for (char orderbc : orders) {
//                System.out.println(orderArr + "\t" + orderbc);
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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsMax2Of3d(Nd4jBackend backend) {
        double[][][] slices = new double[3][][];
        boolean[][][] isMax = new boolean[3][][];

        slices[0] = new double[][] {{1, 10, 2}, {3, 4, 5}};
        slices[1] = new double[][] {{-10, -9, -8}, {-7, -6, -5}};
        slices[2] = new double[][] {{4, 3, 2}, {1, 0, -1}};

        isMax[0] = new boolean[][] {{false, true, false}, {false, false, false}};
        isMax[1] = new boolean[][] {{false, false, false}, {false, false, true}};
        isMax[2] = new boolean[][] {{true, false, false}, {false, false, false}};

        INDArray arr = Nd4j.create(3, 2, 3);
        INDArray expected = Nd4j.create(DataType.BOOL, 3, 2, 3);
        for (int i = 0; i < 3; i++) {
            arr.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).assign(Nd4j.create(slices[i]));
            val t = Nd4j.create(ArrayUtil.flatten(isMax[i]), new long[]{2, 3}, DataType.BOOL);
            val v = expected.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());
            v.assign(t);
        }

        val result = Nd4j.getExecutioner().exec(new IsMax(arr, Nd4j.createUninitialized(DataType.BOOL, arr.shape(), arr.ordering()), 1, 2))[0];

        assertEquals(expected, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsMax2of4d(Nd4jBackend backend) {

        Nd4j.getRandom().setSeed(12345);
        val s = new long[] {2, 3, 4, 5};
        INDArray arr = Nd4j.rand(s).castTo(DataType.DOUBLE);

        //Test 0,1
        INDArray exp = Nd4j.create(DataType.BOOL, s);
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

                subsetExp.putScalar(maxIdx, 1);
            }
        }

        INDArray actC = Nd4j.getExecutioner().exec(new IsMax(arr.dup('c'), Nd4j.createUninitialized(DataType.BOOL, arr.shape()),0, 1))[0];
        INDArray actF = Nd4j.getExecutioner().exec(new IsMax(arr.dup('f'), Nd4j.createUninitialized(DataType.BOOL, arr.shape(), 'f'), 0, 1))[0];

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

        actC = Nd4j.getExecutioner().exec(new IsMax(arr.dup('c'), arr.dup('c').ulike(), 2, 3))[0];
        actF = Nd4j.getExecutioner().exec(new IsMax(arr.dup('f'), arr.dup('f').ulike(), 2, 3))[0];

        assertEquals(exp, actC);
        assertEquals(exp, actF);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIMax2Of3d(Nd4jBackend backend) {
        double[][][] slices = new double[3][][];

        slices[0] = new double[][] {{1, 10, 2}, {3, 4, 5}};
        slices[1] = new double[][] {{-10, -9, -8}, {-7, -6, -5}};
        slices[2] = new double[][] {{4, 3, 2}, {1, 0, -1}};

        //Based on a c-order traversal of each tensor
        val imax = new long[] {1, 5, 0};

        INDArray arr = Nd4j.create(3, 2, 3).castTo(DataType.DOUBLE);
        for (int i = 0; i < 3; i++) {
            arr.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).assign(Nd4j.create(slices[i]));
        }

        INDArray out = Nd4j.exec(new ArgMax(arr, false,new long[]{1,2}))[0];

        assertEquals(DataType.LONG, out.dataType());

        INDArray exp = Nd4j.create(imax, new long[]{3}, DataType.LONG);

        assertEquals(exp, out);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIMax2of4d(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        val s = new long[] {2, 3, 4, 5};
        INDArray arr = Nd4j.rand(s).castTo(DataType.DOUBLE);

        //Test 0,1
        INDArray exp = Nd4j.create(DataType.LONG, 4, 5).castTo(DataType.DOUBLE);
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

        INDArray actC = Nd4j.getExecutioner().exec(new ArgMax(arr.dup('c'), false,new long[]{0,1}))[0].castTo(DataType.DOUBLE);
        INDArray actF = Nd4j.getExecutioner().exec(new ArgMax(arr.dup('f'),  false,new long[]{0,1}))[0].castTo(DataType.DOUBLE);
        //
        assertEquals(exp, actC);
        assertEquals(exp, actF);



        //Test 2,3
        exp = Nd4j.create(DataType.LONG, 2, 3);
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

        actC = Nd4j.getExecutioner().exec(new ArgMax(arr.dup('c'), false,new long[]{2, 3}))[0];
        actF = Nd4j.getExecutioner().exec(new ArgMax(arr.dup('f'), false,new long[]{2, 3}))[0];

        assertEquals(exp, actC);
        assertEquals(exp, actF);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadPermuteEquals(Nd4jBackend backend) {
        INDArray d3c = Nd4j.linspace(1, 5, 5, DataType.DOUBLE).reshape('c', 1, 5, 1);
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRemainder1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(10).assign(5.3).castTo(DataType.DOUBLE);
        INDArray y = Nd4j.create(10).assign(2.0).castTo(DataType.DOUBLE);
        INDArray exp = Nd4j.create(10).assign(-0.7).castTo(DataType.DOUBLE);

        INDArray result = x.remainder(2.0);
        assertEquals(exp, result);

        result = x.remainder(y);
        assertEquals(exp, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFMod1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(10).assign(5.3).castTo(DataType.DOUBLE);
        INDArray y = Nd4j.create(10).assign(2.0).castTo(DataType.DOUBLE);
        INDArray exp = Nd4j.create(10).assign(1.3).castTo(DataType.DOUBLE);

        INDArray result = x.fmod(2.0);
        assertEquals(exp, result);

        result = x.fmod(y);
        assertEquals(exp, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStrangeDups1(Nd4jBackend backend) {
        INDArray array = Nd4j.create(10).assign(0).castTo(DataType.DOUBLE);
        INDArray exp = Nd4j.create(10).assign(1.0f).castTo(DataType.DOUBLE);
        INDArray copy = null;

        for (int x = 0; x < array.length(); x++) {
            array.putScalar(x, 1f);
            copy = array.dup();
        }

        assertEquals(exp, array);
        assertEquals(exp, copy);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStrangeDups2(Nd4jBackend backend) {
        INDArray array = Nd4j.create(10).assign(0).castTo(DataType.DOUBLE);
        INDArray exp1 = Nd4j.create(10).assign(1.0f).castTo(DataType.DOUBLE);
        INDArray exp2 = Nd4j.create(10).assign(1.0f).putScalar(9, 0f).castTo(DataType.DOUBLE);
        INDArray copy = null;

        for (int x = 0; x < array.length(); x++) {
            copy = array.dup();
            array.putScalar(x, 1f);
        }

        assertEquals(exp1, array);
        assertEquals(exp2, copy);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReductionAgreement1(Nd4jBackend backend) {
        INDArray row = Nd4j.linspace(1, 3, 3, DataType.DOUBLE).reshape(1, 3);
        INDArray mean0 = row.mean(0);
        assertFalse(mean0 == row); //True: same object (should be a copy)

        INDArray col = Nd4j.linspace(1, 3, 3, DataType.DOUBLE).reshape(1, -1).transpose();
        INDArray mean1 = col.mean(1);
        assertFalse(mean1 == col);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpecialConcat1(Nd4jBackend backend) {
        for (int i = 0; i < 10; i++) {
            List<INDArray> arrays = new ArrayList<>();
            for (int x = 0; x < 10; x++) {
                arrays.add(Nd4j.create(1, 100).assign(x).castTo(DataType.DOUBLE));
            }

            INDArray matrix = Nd4j.specialConcat(0, arrays.toArray(new INDArray[0]));
            assertEquals(10, matrix.rows());
            assertEquals(100, matrix.columns());

            for (int x = 0; x < 10; x++) {
                assertEquals(x, matrix.getRow(x).meanNumber().doubleValue(), 0.1);
                assertEquals(arrays.get(x), matrix.getRow(x).reshape(1,matrix.size(1)));
            }
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpecialConcat2(Nd4jBackend backend) {
        List<INDArray> arrays = new ArrayList<>();
        for (int x = 0; x < 10; x++) {
            arrays.add(Nd4j.create(new double[] {x, x, x, x, x, x}).reshape(1, 6));
        }

        INDArray matrix = Nd4j.specialConcat(0, arrays.toArray(new INDArray[0]));
        assertEquals(10, matrix.rows());
        assertEquals(6, matrix.columns());

//        log.info("Result: {}", matrix);

        for (int x = 0; x < 10; x++) {
            assertEquals(x, matrix.getRow(x).meanNumber().doubleValue(), 0.1);
            assertEquals(arrays.get(x), matrix.getRow(x).reshape(1, matrix.size(1)));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutScalar1(Nd4jBackend backend) {
        INDArray array = Nd4j.create(10, 3, 96, 96).castTo(DataType.DOUBLE);

        for (int i = 0; i < 10; i++) {
//            log.info("Trying i: {}", i);
            array.tensorAlongDimension(i, 1, 2, 3).putScalar(1, 2, 3, 1);
        }
    }





    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testZ1(Nd4jBackend backend) {
        INDArray matrix = Nd4j.create(10, 10).assign(1.0).castTo(DataType.DOUBLE);

        INDArray exp = Nd4j.create(10).assign(10.0).castTo(DataType.DOUBLE);

        INDArray res = Nd4j.create(10).castTo(DataType.DOUBLE);
        INDArray sums = matrix.sum(res, 0);

        assertTrue(res == sums);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDupDelayed(Nd4jBackend backend) {
        if (!(Nd4j.getExecutioner() instanceof GridExecutioner))
            return;

//        Nd4j.getExecutioner().commit();
        val executioner = (GridExecutioner) Nd4j.getExecutioner();

//        log.info("Starting: -------------------------------");

        //log.info("Point A: [{}]", executioner.getQueueLength());

        INDArray in = Nd4j.zeros(10).castTo(DataType.DOUBLE);

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
            assertEquals(out.get(i), comp.get(i),"Failed at iteration: [" + i + "]");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarReduction1(Nd4jBackend backend) {
        val op = new Norm2(Nd4j.create(1).assign(1.0));
        double norm2 = Nd4j.getExecutioner().execAndReturn(op).getFinalResult().doubleValue();
        double norm1 = Nd4j.getExecutioner().execAndReturn(new Norm1(Nd4j.create(1).assign(1.0))).getFinalResult()
                .doubleValue();
        double sum = Nd4j.getExecutioner().execAndReturn(new Sum(Nd4j.create(1).assign(1.0))).getFinalResult()
                .doubleValue();

        assertEquals(1.0, norm2, 0.001);
        assertEquals(1.0, norm1, 0.001);
        assertEquals(1.0, sum, 0.001);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tesAbsReductions1(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {-1, -2, -3, -4}).castTo(DataType.DOUBLE);

        assertEquals(4, array.amaxNumber().intValue());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tesAbsReductions2(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {-1, -2, -3, -4}).castTo(DataType.DOUBLE);

        assertEquals(1, array.aminNumber().intValue());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tesAbsReductions3(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {-2, -2, 2, 2});

        assertEquals(2, array.ameanNumber().intValue());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tesAbsReductions4(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {-2, -2, 2, 3}).castTo(DataType.DOUBLE);
        assertEquals(1.0, array.sumNumber().doubleValue(), 1e-5);

        assertEquals(4, array.scan(Conditions.absGreaterThanOrEqual(0.0)).intValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tesAbsReductions5(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {-2, 0.0, 2, 2});

        assertEquals(3, array.scan(Conditions.absGreaterThan(0.0)).intValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewBroadcastComparison1(Nd4jBackend backend) {
        val initial = Nd4j.create(3, 5).castTo(DataType.DOUBLE);
        val mask = Nd4j.create(new double[] {5, 4, 3, 2, 1}).castTo(DataType.DOUBLE);
        val result = Nd4j.createUninitialized(DataType.BOOL, initial.shape());
        val exp = Nd4j.create(new boolean[] {true, true, true, false, false});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i);
        }

        Nd4j.getExecutioner().commit();
//        log.info("original: \n{}", initial);

        Nd4j.getExecutioner().exec(new BroadcastLessThan(initial, mask, result, 1));

        Nd4j.getExecutioner().commit();
//        log.info("Comparison ----------------------------------------------");
        for (int i = 0; i < initial.rows(); i++) {
            val row = result.getRow(i);
            assertEquals(exp, row,"Failed at row " + i);
//            log.info("-------------------");
        }
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewBroadcastComparison2(Nd4jBackend backend) {
        val initial = Nd4j.create(3, 5).castTo(DataType.DOUBLE);
        val mask = Nd4j.create(new double[] {5, 4, 3, 2, 1}).castTo(DataType.DOUBLE);
        val result = Nd4j.createUninitialized(DataType.BOOL, initial.shape());
        val exp = Nd4j.create(new boolean[] {false, false, false, true, true});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i);
        }

        Nd4j.getExecutioner().commit();


        Nd4j.getExecutioner().exec(new BroadcastGreaterThan(initial, mask, result, 1));



        for (int i = 0; i < initial.rows(); i++) {
            assertEquals(exp, result.getRow(i),"Failed at row " + i);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewBroadcastComparison3(Nd4jBackend backend) {
        val initial = Nd4j.create(3, 5).castTo(DataType.DOUBLE);
        val mask = Nd4j.create(new double[] {5, 4, 3, 2, 1}).castTo(DataType.DOUBLE);
        val result = Nd4j.createUninitialized(DataType.BOOL, initial.shape());
        val exp = Nd4j.create(new boolean[] {false, false, true, true, true});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i + 1);
        }

        Nd4j.getExecutioner().commit();


        Nd4j.getExecutioner().exec(new BroadcastGreaterThanOrEqual(initial, mask, result, 1));


        for (int i = 0; i < initial.rows(); i++) {
            assertEquals(exp, result.getRow(i),"Failed at row " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewBroadcastComparison4(Nd4jBackend backend) {
        val initial = Nd4j.create(3, 5).castTo(DataType.DOUBLE);
        val mask = Nd4j.create(new double[] {5, 4, 3, 2, 1}).castTo(DataType.DOUBLE);
        val result = Nd4j.createUninitialized(DataType.BOOL, initial.shape());
        val exp = Nd4j.create(new boolean[] {false, false, true, false, false});

        for (int i = 0; i < initial.columns(); i++) {
            initial.getColumn(i).assign(i + 1);
        }

        Nd4j.getExecutioner().commit();


        Nd4j.getExecutioner().exec(new BroadcastEqualTo(initial, mask, result, 1 ));


        for (int i = 0; i < initial.rows(); i++) {
            assertEquals( exp, result.getRow(i),"Failed at row " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadDup_1(Nd4jBackend backend) {
        INDArray haystack = Nd4j.create(new double[] {-0.84443557262, -0.06822254508, 0.74266910552, 0.61765557527, -0.77555125951,
                        -0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673,
                        0.62955373525, -0.31357592344, 1.03362500667, -0.59279078245, 1.1914824247})
                .reshape(3, 5).castTo(DataType.DOUBLE);
        INDArray needle = Nd4j.create(new double[] {-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673}).castTo(DataType.DOUBLE);

        val row = haystack.getRow(1);
        val drow = row.dup();

//        log.info("row shape: {}", row.shapeInfoDataBuffer());
        assertEquals(needle, drow);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_0(Nd4jBackend backend) {
        INDArray haystack = Nd4j.create(new double[] {-0.84443557262, -0.06822254508, 0.74266910552, 0.61765557527,
                        -0.77555125951, -0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130,
                        -1.25485503673, 0.62955373525, -0.31357592344, 1.03362500667, -0.59279078245, 1.1914824247})
                .reshape(3, 5).castTo(DataType.DOUBLE);
        INDArray needle = Nd4j.create(new double[] {-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130,
                -1.25485503673}).castTo(DataType.DOUBLE);

        INDArray reduced = Nd4j.getExecutioner().exec(new CosineDistance(haystack, needle, 1));

        INDArray exp = Nd4j.create(new double[] {0.577452, 0.0, 1.80182}).castTo(DataType.DOUBLE);
        assertEquals(exp, reduced);

        for (int i = 0; i < haystack.rows(); i++) {
            val row = haystack.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new CosineDistance(row, needle)).z().getDouble(0);
            assertEquals(reduced.getDouble(i), res, 1e-5,"Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduce3SignaturesEquality_1(Nd4jBackend backend) {
        val x = Nd4j.rand(DataType.DOUBLE, 3, 4, 5);
        val y = Nd4j.rand(DataType.DOUBLE, 3, 4, 5);

        val reduceOp = new ManhattanDistance(x, y, 0);
        val op = (Op) reduceOp;

        val z0 = Nd4j.getExecutioner().exec(reduceOp);
        val z1 = Nd4j.getExecutioner().exec(op);

        assertEquals(z0, z1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_1(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 10).castTo(DataType.DOUBLE);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(new double[] {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.getExecutioner().exec(new CosineSimilarity(initial, needle, 1));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            double res = Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(initial.getRow(i).dup(), needle))
                    .getFinalResult().doubleValue();
            assertEquals( reduced.getDouble(i), res, 0.001,"Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_2(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 10).castTo(DataType.DOUBLE);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.getExecutioner().exec(new ManhattanDistance(initial, needle, 1));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            double res = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(initial.getRow(i).dup(), needle))
                    .getFinalResult().doubleValue();
            assertEquals(reduced.getDouble(i), res, 0.001,"Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_3(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 10).castTo(DataType.DOUBLE);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.getExecutioner().exec(new EuclideanDistance(initial, needle, 1));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            INDArray x = initial.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(x, needle)).getFinalResult()
                    .doubleValue();
            assertEquals( reduced.getDouble(i), res, 0.001,"Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_3_NEG(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 10).castTo(DataType.DOUBLE);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.getExecutioner().exec(new EuclideanDistance(initial, needle, -1));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            INDArray x = initial.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(x, needle)).getFinalResult()
                    .doubleValue();
            assertEquals(reduced.getDouble(i), res, 0.001,"Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_3_NEG_2(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 10).castTo(DataType.DOUBLE);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.create(5).castTo(DataType.DOUBLE);
        Nd4j.getExecutioner().exec(new CosineSimilarity(initial, needle, reduced, -1));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            INDArray x = initial.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(x, needle)).getFinalResult()
                    .doubleValue();
            assertEquals(reduced.getDouble(i), res, 0.001,"Failed at " + i);
        }
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_4(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 6, 7).castTo(DataType.DOUBLE);
        for (int i = 0; i < 5; i++) {
            initial.tensorAlongDimension(i, 1, 2).assign(i + 1);
        }
        INDArray needle = Nd4j.create(6, 7).assign(1.0).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.getExecutioner().exec(new ManhattanDistance(initial, needle,  1,2));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < 5; i++) {
            double res = Nd4j.getExecutioner()
                    .execAndReturn(new ManhattanDistance(initial.tensorAlongDimension(i, 1, 2).dup(), needle))
                    .getFinalResult().doubleValue();
            assertEquals(reduced.getDouble(i), res, 0.001,"Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAtan2_1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(10).assign(-1.0).castTo(DataType.DOUBLE);
        INDArray y = Nd4j.create(10).assign(0.0).castTo(DataType.DOUBLE);
        INDArray exp = Nd4j.create(10).assign(Math.PI).castTo(DataType.DOUBLE);

        INDArray z = Transforms.atan2(x, y);

        assertEquals(exp, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAtan2_2(Nd4jBackend backend) {
        INDArray x = Nd4j.create(10).assign(1.0).castTo(DataType.DOUBLE);
        INDArray y = Nd4j.create(10).assign(0.0).castTo(DataType.DOUBLE);
        INDArray exp = Nd4j.create(10).assign(0.0).castTo(DataType.DOUBLE);

        INDArray z = Transforms.atan2(x, y);

        assertEquals(exp, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testJaccardDistance1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 1, 0, 0, 1, 0}).castTo(DataType.DOUBLE);
        INDArray y = Nd4j.create(new double[] {1, 1, 0, 1, 0, 0}).castTo(DataType.DOUBLE);

        double val = Transforms.jaccardDistance(x, y);

        assertEquals(0.75, val, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testJaccardDistance2(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 1, 0, 0, 1, 1}).castTo(DataType.DOUBLE);
        INDArray y = Nd4j.create(new double[] {1, 1, 0, 1, 0, 0}).castTo(DataType.DOUBLE);

        double val = Transforms.jaccardDistance(x, y);

        assertEquals(0.8, val, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHammingDistance1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 0, 0, 1, 0, 0}).castTo(DataType.DOUBLE);
        INDArray y = Nd4j.create(new double[] {0, 0, 0, 0, 1, 0}).castTo(DataType.DOUBLE);

        double val = Transforms.hammingDistance(x, y);

        assertEquals(2.0 / 6, val, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHammingDistance2(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 0, 0, 1, 0, 0});
        INDArray y = Nd4j.create(new double[] {0, 1, 0, 0, 1, 0});

        double val = Transforms.hammingDistance(x, y);

        assertEquals(3.0 / 6, val, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHammingDistance3(Nd4jBackend backend) {
        INDArray x = Nd4j.create(DataType.DOUBLE, 10, 6);
        for (int r = 0; r < x.rows(); r++) {
            val p = r % x.columns();
            x.getRow(r).putScalar(p, 1);
        }

        INDArray y = Nd4j.create(new double[] {0, 0, 0, 0, 1, 0});

        INDArray res = Nd4j.getExecutioner().exec(new HammingDistance(x, y, 1));
        assertEquals(10, res.length());

        for (int r = 0; r < x.rows(); r++) {
            if (r == 4) {
                assertEquals(0.0, res.getDouble(r), 1e-5,"Failed at " + r);
            } else {
                assertEquals(2.0 / 6, res.getDouble(r), 1e-5,"Failed at " + r);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances1(Nd4jBackend backend) {
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

                assertEquals(exp, res, 0.001,"Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances2(Nd4jBackend backend) {
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

                assertEquals( exp, res, 0.001,"Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances2_Large(Nd4jBackend backend) {
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

                assertEquals(exp, res, 0.001,"Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances3_Large(Nd4jBackend backend) {
        INDArray initialX = Nd4j.create(5, 2000);
        INDArray initialY = Nd4j.create(7, 2000);
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

                assertEquals(exp, res, 0.001,"Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances3_Large_Columns(Nd4jBackend backend) {
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

                assertEquals(exp, res, 0.001,"Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances4_Large_Columns(Nd4jBackend backend) {
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

                assertEquals(exp, res, 0.001,"Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances5_Large_Columns(Nd4jBackend backend) {
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

                assertEquals(exp, res, 0.001,"Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances3_Small_Columns(Nd4jBackend backend) {
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

                assertEquals(exp, res, 0.001,"Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances3(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(123);

        INDArray initialX = Nd4j.rand(5, 10).castTo(DataType.DOUBLE);
        INDArray initialY = initialX.mul(-1);

        INDArray result = Transforms.allCosineSimilarities(initialX, initialY, 1);

        assertEquals(5 * 5, result.length());

        for (int x = 0; x < initialX.rows(); x++) {

            INDArray rowX = initialX.getRow(x).dup();

            for (int y = 0; y < initialY.rows(); y++) {

                double res = result.getDouble(x, y);
                double exp = Transforms.cosineSim(rowX, initialY.getRow(y).dup());

                assertEquals(exp, res, 0.001,"Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedTransforms1(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEntropy1(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(1, 100).castTo(DataType.DOUBLE);

        double exp = MathUtils.entropy(x.data().asDouble());
        double res = x.entropyNumber().doubleValue();

        assertEquals(exp, res, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEntropy2(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(10, 100).castTo(DataType.DOUBLE);

        INDArray res = x.entropy(1);

        assertEquals(10, res.length());

        for (int t = 0; t < x.rows(); t++) {
            double exp = MathUtils.entropy(x.getRow(t).dup().data().asDouble());

            assertEquals(exp, res.getDouble(t), 1e-5);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEntropy3(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(1, 100).castTo(DataType.DOUBLE);

        double exp = getShannonEntropy(x.data().asDouble());
        double res = x.shannonEntropyNumber().doubleValue();

        assertEquals(exp, res, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEntropy4(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(1, 100).castTo(DataType.DOUBLE);

        double exp = getLogEntropy(x.data().asDouble());
        double res = x.logEntropyNumber().doubleValue();

        assertEquals(exp, res, 1e-5);
    }

    protected double getShannonEntropy(double[] array) {
        double ret = 0;
        for (double x : array) {
            ret += x * FastMath.log(2., x);
        }

        return -ret;
    }

    protected double getLogEntropy(double[] array) {
        return Math.log(MathUtils.entropy(array));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverse1(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        INDArray rev = Nd4j.reverse(array);

        assertEquals(exp, rev);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverse2(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

        INDArray rev = Nd4j.reverse(array);

        assertEquals(exp, rev);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverse3(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        INDArray rev = Nd4j.getExecutioner().exec(new Reverse(array, array.ulike()))[0];

        assertEquals(exp, rev);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverse4(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

        INDArray rev = Nd4j.getExecutioner().exec(new Reverse(array,array.ulike()))[0];

        assertEquals(exp, rev);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverse5(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

        INDArray rev = Transforms.reverse(array, true);

        assertEquals(exp, rev);
        assertFalse(rev == array);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverse6(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        INDArray exp = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

        INDArray rev = Transforms.reverse(array, false);

        assertEquals(exp, rev);
        assertTrue(rev == array);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeSortView1(Nd4jBackend backend) {
        INDArray matrix = Nd4j.create(10, 10);
        INDArray exp = Nd4j.linspace(0, 9, 10, DataType.DOUBLE);
        int cnt = 0;
        for (long i = matrix.rows() - 1; i >= 0; i--) {
            matrix.getRow((int) i).assign(cnt);
            cnt++;
        }

        Nd4j.sort(matrix.getColumn(0), true);

        assertEquals(exp, matrix.getColumn(0));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeSort1(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {9, 2, 1, 7, 6, 5, 4, 3, 8, 0});
        INDArray exp1 = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        INDArray exp2 = Nd4j.create(new double[] {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

        INDArray res = Nd4j.sort(array, true);

        assertEquals(exp1, res);

        res = Nd4j.sort(res, false);

        assertEquals(exp2, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeSort2(Nd4jBackend backend) {
        INDArray array = Nd4j.rand(1, 10000).castTo(DataType.DOUBLE);

        INDArray res = Nd4j.sort(array, true);
        INDArray exp = res.dup();

        res = Nd4j.sort(res, false);
        res = Nd4j.sort(res, true);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("Crashes")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testNativeSort3(Nd4jBackend backend) {
        int length = isIntegrationTests() ? 1048576 : 16484;
        INDArray array = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape(1, -1);
        INDArray exp = array.dup();
        Nd4j.shuffle(array, 0);

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, true);
        long time2 = System.currentTimeMillis();
        log.info("Time spent: {} ms", time2 - time1);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLongShapeDescriptor(){
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        INDArray arr = Nd4j.create(new float[]{1,2,3});

        val lsd = arr.shapeDescriptor();
        assertNotNull(lsd);     //Fails here on CUDA, OK on native/cpu
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverseSmall_1(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 10, 10, DataType.INT);
        val exp = array.dup(array.ordering());

        Transforms.reverse(array, false);
        Transforms.reverse(array, false);

        val jexp = exp.data().asInt();
        val jarr = array.data().asInt();
        assertArrayEquals(jexp, jarr);
        assertEquals(exp, array);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverseSmall_2(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 10, 10, DataType.INT);
        val exp = array.dup(array.ordering());

        val reversed = Transforms.reverse(array, true);
        val rereversed = Transforms.reverse(reversed, true);

        val jexp = exp.data().asInt();
        val jarr = rereversed.data().asInt();
        assertArrayEquals(jexp, jarr);
        assertEquals(exp, rereversed);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverseSmall_3(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 11, 11, DataType.INT);
        val exp = array.dup(array.ordering());

        Transforms.reverse(array, false);

        Transforms.reverse(array, false);

        val jexp = exp.data().asInt();
        val jarr = array.data().asInt();
        assertArrayEquals(jexp, jarr);
        assertEquals(exp, array);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverseSmall_4(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 11, 11, DataType.INT);
        val exp = array.dup(array.ordering());

        val reversed = Transforms.reverse(array, true);
        val rereversed = Transforms.reverse(reversed, true);

        val jexp = exp.data().asInt();
        val jarr = rereversed.data().asInt();
        assertArrayEquals(jexp, jarr);
        assertEquals(exp, rereversed);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverse_1(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 2017152, 2017152, DataType.INT);
        val exp = array.dup(array.ordering());

        Transforms.reverse(array, false);
        Transforms.reverse(array, false);

        val jexp = exp.data().asInt();
        val jarr = array.data().asInt();
        assertArrayEquals(jexp, jarr);
        assertEquals(exp, array);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverse_2(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 2017152, 2017152, DataType.INT);
        val exp = array.dup(array.ordering());

        val reversed = Transforms.reverse(array, true);
        val rereversed = Transforms.reverse(reversed, true);

        val jexp = exp.data().asInt();
        val jarr = rereversed.data().asInt();
        assertArrayEquals(jexp, jarr);
        assertEquals(exp, rereversed);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeSort3_1(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 2017152, 2017152, DataType.DOUBLE).reshape(1, -1);
        INDArray exp = array.dup();
        Transforms.reverse(array, false);

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, true);
        long time2 = System.currentTimeMillis();
        log.info("Time spent: {} ms", time2 - time1);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testNativeSortAlongDimension1(Nd4jBackend backend) {
        INDArray array = Nd4j.create(1000, 1000);
        INDArray exp1 = Nd4j.linspace(1, 1000, 1000, DataType.DOUBLE);
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

        val e = exp1.toDoubleVector();
        for (int r = 0; r < array.rows(); r++) {
            val d = res.getRow(r).dup();

            assertArrayEquals(e, d.toDoubleVector(), 1e-5);
            assertEquals(exp1, d,"Failed at " + r);
        }
    }

    protected boolean checkIfUnique(INDArray array, int iteration) {
        var jarray = array.data().asInt();
        var set = new HashSet<Integer>();

        for (val v : jarray) {
            if (set.contains(Integer.valueOf(v)))
                throw new IllegalStateException("Duplicate value found: [" + v + "] on iteration " + iteration);

            set.add(Integer.valueOf(v));
        }

        return true;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void shuffleTest(Nd4jBackend backend) {
        for (int e = 0; e < 5; e++) {
            //log.info("---------------------");
            val array = Nd4j.linspace(1, 1011, 1011, DataType.INT);

            checkIfUnique(array, e);
            Nd4j.getExecutioner().commit();

            Nd4j.shuffle(array, 0);
            Nd4j.getExecutioner().commit();

            checkIfUnique(array, e);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("Crashes")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testNativeSortAlongDimension3(Nd4jBackend backend) {
        INDArray array = Nd4j.create(2000,  2000);
        INDArray exp1 = Nd4j.linspace(1, 2000, 2000, DataType.DOUBLE);
        INDArray dps = exp1.dup();

        Nd4j.getExecutioner().commit();
        Nd4j.shuffle(dps, 0);

        assertNotEquals(exp1, dps);


        for (int r = 0; r < array.rows(); r++) {
            array.getRow(r).assign(dps);
        }

        val arow = array.getRow(0).toFloatVector();

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, 1, true);
        long time2 = System.currentTimeMillis();

        log.info("Time spent: {} ms", time2 - time1);

        val jexp = exp1.toFloatVector();
        for (int r = 0; r < array.rows(); r++) {
            val jrow = res.getRow(r).toFloatVector();
            //log.info("jrow: {}", jrow);
            assertArrayEquals(jexp, jrow, 1e-5f,"Failed at " + r);
            assertEquals( exp1, res.getRow(r),"Failed at " + r);
            //assertArrayEquals("Failed at " + r, exp1.data().asDouble(), res.getRow(r).dup().data().asDouble(), 1e-5);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("Crashes")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testNativeSortAlongDimension2(Nd4jBackend backend) {
        INDArray array = Nd4j.create(100, 10);
        INDArray exp1 = Nd4j.create(new double[] {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

        for (int r = 0; r < array.rows(); r++) {
            array.getRow(r).assign(Nd4j.create(new double[] {3, 8, 2, 7, 5, 6, 4, 9, 1, 0}));
        }

        INDArray res = Nd4j.sort(array, 1, false);

        for (int r = 0; r < array.rows(); r++) {
            assertEquals(exp1, res.getRow(r).dup(),"Failed at " + r);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPercentile1(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 10, 10, DataType.DOUBLE);
        Percentile percentile = new Percentile(50);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(50));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPercentile2(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 9, 9, DataType.DOUBLE);
        Percentile percentile = new Percentile(50);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(50));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPercentile3(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 9, 9, DataType.DOUBLE);
        Percentile percentile = new Percentile(75);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(75));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPercentile4(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 10, 10, DataType.DOUBLE);
        Percentile percentile = new Percentile(75);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(75));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPercentile5(Nd4jBackend backend) {
        val array = Nd4j.createFromArray(new int[]{1, 1982});
        val perc = array.percentileNumber(75);
        assertEquals(1982.f, perc.floatValue(), 1e-5f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadPercentile1(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 10, 10, DataType.DOUBLE);
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutiRowVector(Nd4jBackend backend) {
        INDArray matrix = Nd4j.createUninitialized(10, 10);
        INDArray exp = Nd4j.create(10, 10).assign(1.0);
        INDArray row = Nd4j.create(10).assign(1.0);

        matrix.putiRowVector(row);

        assertEquals(exp, matrix);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutiColumnsVector(Nd4jBackend backend) {
        INDArray matrix = Nd4j.createUninitialized(5, 10);
        INDArray exp = Nd4j.create(5, 10).assign(1.0);
        INDArray row = Nd4j.create(5, 1).assign(1.0);

        matrix.putiColumnVector(row);

        Nd4j.getExecutioner().commit();

        assertEquals(exp, matrix);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRsub1(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(5).assign(2.0);
        INDArray exp_0 = Nd4j.ones(5).assign(2.0);
        INDArray exp_1 = Nd4j.create(5).assign(-1);

        Nd4j.getExecutioner().commit();

        INDArray res = arr.rsub(1.0);

        assertEquals(exp_0, arr);
        assertEquals(exp_1, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastMin(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastMax(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastAMax(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastAMin(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testLogExpSum1(Nd4jBackend backend) {
        INDArray matrix = Nd4j.create(3, 3);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{1, 2, 3}));
        }

        INDArray res = Nd4j.getExecutioner().exec(new LogSumExp(matrix, false, 1))[0];

        for (int e = 0; e < res.length(); e++) {
            assertEquals(3.407605, res.getDouble(e), 1e-5);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testLogExpSum2(Nd4jBackend backend) {
        INDArray row = Nd4j.create(new double[]{1, 2, 3});

        double res = Nd4j.getExecutioner().exec(new LogSumExp(row))[0].getDouble(0);

        assertEquals(3.407605, res, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPow1(Nd4jBackend backend) {
        val argX = Nd4j.create(3).assign(2.0);
        val argY = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        val exp = Nd4j.create(new double[] {2.0, 4.0, 8.0});
        val res = Transforms.pow(argX, argY);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRDiv1(Nd4jBackend backend) {
        val argX = Nd4j.create(3).assign(2.0);
        val argY = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        val exp = Nd4j.create(new double[] {0.5, 1.0, 1.5});
        val res = argX.rdiv(argY);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEqualOrder1(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
        val arrayC = array.dup('c');
        val arrayF = array.dup('f');

        assertEquals(array, arrayC);
        assertEquals(array, arrayF);
        assertEquals(arrayC, arrayF);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatchTransform(Nd4jBackend backend) {
        val array = Nd4j.create(new double[] {1, 1, 1, 0, 1, 1},'c');
        val result = Nd4j.createUninitialized(DataType.BOOL, array.shape());
        val exp = Nd4j.create(new boolean[] {false, false, false, true, false, false});
        Op op = new MatchConditionTransform(array, result, 1e-5, Conditions.epsEquals(0.0));

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test4DSumView(Nd4jBackend backend) {
        INDArray labels = Nd4j.linspace(1, 160, 160, DataType.DOUBLE).reshape(2, 5, 4, 4);
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
    public void testReduction_Z1(Nd4jBackend backend) {
        val arrayX = Nd4j.create(10, 10, 10);

        val res = arrayX.max(1, 2);

        Nd4j.getExecutioner().commit();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduction_Z2(Nd4jBackend backend) {
        val arrayX = Nd4j.create(10, 10);

        val res = arrayX.max(0);

        Nd4j.getExecutioner().commit();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduction_Z3(Nd4jBackend backend) {
        val arrayX = Nd4j.create(200, 300);

        val res = arrayX.maxNumber().doubleValue();

        Nd4j.getExecutioner().commit();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSoftmaxZ1(Nd4jBackend backend) {
        val original = Nd4j.linspace(1, 100, 100, DataType.DOUBLE).reshape(10, 10);
        val reference = original.dup(original.ordering());
        val expected = original.dup(original.ordering());

        Nd4j.getExecutioner().execAndReturn((CustomOp) new SoftMax(expected, expected, -1));

        val result = Nd4j.getExecutioner().exec((CustomOp) new SoftMax(original, original.dup(original.ordering())))[0];

        assertEquals(reference, original);
        assertEquals(expected, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRDiv(Nd4jBackend backend) {
        val x = Nd4j.create(new double[]{2,2,2});
        val y = Nd4j.create(new double[]{4,6,8});
        val result = Nd4j.createUninitialized(DataType.DOUBLE, 3);

        assertEquals(DataType.DOUBLE, x.dataType());
        assertEquals(DataType.DOUBLE, y.dataType());
        assertEquals(DataType.DOUBLE, result.dataType());

        val op = DynamicCustomOp.builder("RDiv")
                .addInputs(x,y)
                .addOutputs(result)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(Nd4j.create(new double[]{2, 3, 4}), result);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIm2Col(Nd4jBackend backend) {
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


        val input = Nd4j.linspace(1, 2 * inY * inX, 2 * inY * inX, DataType.DOUBLE).reshape(2, 1, inY, inX);
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
                        .paddingMode(PaddingMode.SAME)
                        .build())

                .build();

        Nd4j.getExecutioner().exec(im2colOp);
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
            //final boolean eq = outerProduct.equalsWithEps(outerProductDuped, 1e-5);
            //assertTrue(eq);
            assertEquals(outerProductDuped, outerProduct);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeFailure(Nd4jBackend backend) {
        assertThrows(ND4JIllegalStateException.class,() -> {
            val a = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2,2);
            val b = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2,2);
            val score = a.mmul(b);
            val reshaped1 = score.reshape(2,100);
            val reshaped2 = score.reshape(2,1);
        });

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalar_1(Nd4jBackend backend) {
        val scalar = Nd4j.create(new float[]{2.0f}, new long[]{});

        assertTrue(scalar.isScalar());
        assertEquals(1, scalar.length());
        assertFalse(scalar.isMatrix());
        assertFalse(scalar.isVector());
        assertFalse(scalar.isRowVector());
        assertFalse(scalar.isColumnVector());

        assertEquals(2.0f, scalar.getFloat(0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalar_2(Nd4jBackend backend) {
        val scalar = Nd4j.scalar(2.0f);
        val scalar2 = Nd4j.scalar(2.0f);
        val scalar3 = Nd4j.scalar(3.0f);

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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVector_1(Nd4jBackend backend) {
        val vector = Nd4j.createFromArray(new float[] {1, 2, 3, 4, 5});
        val vector2 = Nd4j.createFromArray(new float[] {1, 2, 3, 4, 5});
        val vector3 = Nd4j.createFromArray(new float[] {1, 2, 3, 4, 6});

        assertFalse(vector.isScalar());
        assertEquals(5, vector.length());
        assertFalse(vector.isMatrix());
        assertTrue(vector.isVector());
        assertTrue(vector.isRowVector());
        assertFalse(vector.isColumnVector());

        assertEquals(vector, vector2);
        assertNotEquals(vector, vector3);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorScalar_2(Nd4jBackend backend) {
        val vector = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5});
        val scalar = Nd4j.scalar(2.0f);
        val exp = Nd4j.createFromArray(new float[]{3, 4, 5, 6, 7});

        vector.addi(scalar);

        assertEquals(exp, vector);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeScalar(Nd4jBackend backend) {
        val scalar = Nd4j.scalar(2.0f);
        val newShape = scalar.reshape(1, 1, 1, 1);

        assertEquals(4, newShape.rank());
        assertArrayEquals(new long[]{1, 1, 1, 1}, newShape.shape());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeVector(Nd4jBackend backend) {
        val vector = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5, 6});
        val newShape = vector.reshape(3, 2);

        assertEquals(2, newShape.rank());
        assertArrayEquals(new long[]{3, 2}, newShape.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTranspose1(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            val vector = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5, 6});

            assertArrayEquals(new long[]{6}, vector.shape());
            assertArrayEquals(new long[]{1}, vector.stride());

            val transposed = vector.transpose();

            assertArrayEquals(vector.shape(), transposed.shape());
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTranspose2(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            val scalar = Nd4j.scalar(2.f);

            assertArrayEquals(new long[]{}, scalar.shape());
            assertArrayEquals(new long[]{}, scalar.stride());

            val transposed = scalar.transpose();

            assertArrayEquals(scalar.shape(), transposed.shape());
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    //@Disabled
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
    public void testScalarSqueeze(Nd4jBackend backend) {
        val scalar = Nd4j.create(new float[]{2.0f}, new long[]{1, 1});
        val output = Nd4j.scalar(0.0f);
        val exp = Nd4j.scalar(2.0f);
        val op = DynamicCustomOp.builder("squeeze")
                .addInputs(scalar)
                .addOutputs(output)
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(new long[]{}, Shape.shape(shape.asLong()));

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarVectorSqueeze(Nd4jBackend backend) {
        val scalar = Nd4j.create(new float[]{2.0f}, new long[]{1});

        assertArrayEquals(new long[]{1}, scalar.shape());

        val output = Nd4j.scalar(0.0f);
        val exp = Nd4j.scalar(2.0f);
        val op = DynamicCustomOp.builder("squeeze")
                .addInputs(scalar)
                .addOutputs(output)
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(new long[]{}, Shape.shape(shape.asLong()));

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorSqueeze(Nd4jBackend backend) {
        val vector = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6}, new long[]{1, 6});
        val output = Nd4j.createFromArray(new float[] {0, 0, 0, 0, 0, 0});
        val exp = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5, 6});

        val op = DynamicCustomOp.builder("squeeze")
                .addInputs(vector)
                .addOutputs(output)
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(new long[]{6}, Shape.shape(shape.asLong()));

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatrixReshape(Nd4jBackend backend) {
        val matrix = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, new long[] {3, 3});
        val exp = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, new long[] {9});

        val reshaped = matrix.reshape(-1);

        assertArrayEquals(exp.shape(), reshaped.shape());
        assertEquals(exp, reshaped);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorScalarConcat(Nd4jBackend backend) {
        val vector = Nd4j.createFromArray(new float[] {1, 2});
        val scalar = Nd4j.scalar(3.0f);

        val output = Nd4j.createFromArray(new float[]{0, 0, 0});
        val exp = Nd4j.createFromArray(new float[]{1, 2, 3});

        val op = DynamicCustomOp.builder("concat")
                .addInputs(vector, scalar)
                .addOutputs(output)
                .addIntegerArguments(0) // axis
                .build();

        val shape = Nd4j.getExecutioner().calculateOutputShape(op).get(0);
        assertArrayEquals(exp.shape(), Shape.shape(shape.asLong()));

        Nd4j.getExecutioner().exec(op);

        assertArrayEquals(exp.shape(), output.shape());
        assertEquals(exp, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarPrint_1(Nd4jBackend backend) {
        val scalar = Nd4j.scalar(3.0f);

        Nd4j.exec(new PrintVariable(scalar, true));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testValueArrayOf_1(Nd4jBackend backend) {
        val vector = Nd4j.valueArrayOf(new long[] {5}, 2f, DataType.FLOAT);
        val exp = Nd4j.createFromArray(new float[]{2, 2, 2, 2, 2});

        assertArrayEquals(exp.shape(), vector.shape());
        assertEquals(exp, vector);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testValueArrayOf_2(Nd4jBackend backend) {
        val scalar = Nd4j.valueArrayOf(new long[] {}, 2f);
        val exp = Nd4j.scalar(2f);

        assertArrayEquals(exp.shape(), scalar.shape());
        assertEquals(exp, scalar);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayCreation(Nd4jBackend backend) {
        val vector = Nd4j.create(new float[]{1, 2, 3}, new long[] {3}, 'c');
        val exp = Nd4j.createFromArray(new float[]{1, 2, 3});

        assertArrayEquals(exp.shape(), vector.shape());
        assertEquals(exp, vector);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testACosh(){
        //http://www.wolframalpha.com/input/?i=acosh(x)

        INDArray in = Nd4j.linspace(1, 3, 20, DataType.DOUBLE);
        INDArray out = Nd4j.getExecutioner().exec(new ACosh(in.dup()));

        INDArray exp = Nd4j.create(in.shape());
        for( int i=0; i<in.length(); i++ ){
            double x = in.getDouble(i);
            double y = Math.log(x + Math.sqrt(x-1) * Math.sqrt(x+1));
            exp.putScalar(i, y);
        }

        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCosh(){
        //http://www.wolframalpha.com/input/?i=cosh(x)

        INDArray in = Nd4j.linspace(-2, 2, 20, DataType.DOUBLE);
        INDArray out = Transforms.cosh(in, true);

        INDArray exp = Nd4j.create(in.shape());
        for( int i=0; i<in.length(); i++ ){
            double x = in.getDouble(i);
            double y = 0.5 * (Math.exp(-x) + Math.exp(x));
            exp.putScalar(i, y);
        }

        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAtanh(){
        //http://www.wolframalpha.com/input/?i=atanh(x)

        INDArray in = Nd4j.linspace(-0.9, 0.9, 10, DataType.DOUBLE);
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLastIndex(){

        INDArray in = Nd4j.create(new double[][]{
                {1,1,1,0},
                {1,1,0,0}});

        INDArray exp0 = Nd4j.create(new long[]{1,1,0,-1}, new long[]{4}, DataType.LONG);
        INDArray exp1 = Nd4j.create(new long[]{2,1}, new long[]{2}, DataType.LONG);

        INDArray out0 = BooleanIndexing.lastIndex(in, Conditions.equals(1), 0);
        INDArray out1 = BooleanIndexing.lastIndex(in, Conditions.equals(1), 1);

        assertEquals(exp0, out0);
        assertEquals(exp1, out1);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduce3AlexBug(Nd4jBackend backend) {
        val arr = Nd4j.linspace(1,100,100, DataType.DOUBLE).reshape('f', 10, 10).dup('c');
        val arr2 = Nd4j.linspace(1,100,100, DataType.DOUBLE).reshape('c', 10, 10);
        val out = Nd4j.getExecutioner().exec(new EuclideanDistance(arr, arr2, 1));
        val exp = Nd4j.create(new double[] {151.93748, 128.86038, 108.37435, 92.22256, 82.9759, 82.9759, 92.22256, 108.37435, 128.86038, 151.93748});

        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistancesEdgeCase1(Nd4jBackend backend) {
        val x = Nd4j.create(400, 20).assign(2.0).castTo(Nd4j.defaultFloatingPointType());
        val y = Nd4j.ones(1, 20).castTo(Nd4j.defaultFloatingPointType());
        val z = Transforms.allEuclideanDistances(x, y, 1);

        val exp = Nd4j.create(400, 1).assign(4.47214);

        assertEquals(exp, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcat_1(Nd4jBackend backend) {
        for(char order : new char[]{'c', 'f'}) {

            INDArray arr1 = Nd4j.create(new double[]{1, 2}, new long[]{1, 2}, order);
            INDArray arr2 = Nd4j.create(new double[]{3, 4}, new long[]{1, 2}, order);

            INDArray out = Nd4j.concat(0, arr1, arr2);
            Nd4j.getExecutioner().commit();
            INDArray exp = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
            assertEquals(exp, out,String.valueOf(order));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
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

        assertEquals(expected, b.rdiv(Nd4j.scalar(2.0)));
        assertEquals(expected, b.rdivColumnVector(Nd4j.scalar(2)));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHalfStuff(Nd4jBackend backend) {
        if (!Nd4j.getExecutioner().getClass().getSimpleName().toLowerCase().contains("cuda"))
            return;

        val dtype = Nd4j.dataType();
        Nd4j.setDataType(DataType.HALF);

        val arr = Nd4j.ones(3, 3);
        arr.addi(2.0f);

        val exp = Nd4j.create(3, 3).assign(3.0f);

        assertEquals(exp, arr);

        Nd4j.setDataType(dtype);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Execution(ExecutionMode.SAME_THREAD)
    public void testInconsistentOutput(Nd4jBackend backend) {
        INDArray in = Nd4j.rand(1, 802816).castTo(DataType.DOUBLE);
        INDArray W = Nd4j.rand(802816, 1).castTo(DataType.DOUBLE);
        INDArray b = Nd4j.create(1).castTo(DataType.DOUBLE);
        INDArray out = fwd(in, W, b);

        for(int i = 0; i < 100;i++) {
            INDArray out2 = fwd(in, W, b);  //l.activate(inToLayer1, false, LayerWorkspaceMgr.noWorkspaces());
            assertEquals( out, out2,"Failed at iteration [" + i + "]");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test3D_create_1(Nd4jBackend backend) {
        val jArray = new float[2][3][4];

        fillJvmArray3D(jArray);

        val iArray = Nd4j.create(jArray);
        val fArray = ArrayUtil.flatten(jArray);

        assertArrayEquals(new long[]{2, 3, 4}, iArray.shape());

        assertArrayEquals(fArray, iArray.data().asFloat(), 1e-5f);

        int cnt = 0;
        for (val f : fArray)
            assertTrue(f > 0.0f,"Failed for element [" + cnt++ +"]");
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test4D_create_1(Nd4jBackend backend) {
        val jArray = new float[2][3][4][5];

        fillJvmArray4D(jArray);

        val iArray = Nd4j.create(jArray);
        val fArray = ArrayUtil.flatten(jArray);

        assertArrayEquals(new long[]{2, 3, 4, 5}, iArray.shape());

        assertArrayEquals(fArray, iArray.data().asFloat(), 1e-5f);

        int cnt = 0;
        for (val f : fArray)
            assertTrue(f > 0.0f,"Failed for element [" + cnt++ +"]");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcast_1(Nd4jBackend backend) {
        val array1 = Nd4j.linspace(1, 10, 10, DataType.DOUBLE).reshape(5, 1, 2).broadcast(5, 4, 2);
        val array2 = Nd4j.linspace(1, 20, 20, DataType.DOUBLE).reshape(5, 4, 1).broadcast(5, 4, 2);
        val exp = Nd4j.create(new double[] {2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 8.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 17.0f, 17.0f, 18.0f, 20.0f, 21.0f, 21.0f, 22.0f, 22.0f, 23.0f, 23.0f, 24.0f, 26.0f, 27.0f, 27.0f, 28.0f, 28.0f, 29.0f, 29.0f, 30.0f}).reshape(5,4,2);

        array1.addi(array2);

        assertEquals(exp, array1);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddiColumnEdge(){
        INDArray arr1 = Nd4j.create(1, 5);
        arr1.addiColumnVector(Nd4j.ones(1));
        assertEquals(Nd4j.ones(1,5), arr1);
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
    public void testTile_1(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRelativeError_1(Nd4jBackend backend) {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.ones(10, 10);
        val exp = Nd4j.ones(10, 10);

        Nd4j.getExecutioner().exec(new BinaryRelativeError(arrayX, arrayY, arrayX, 0.1));

        assertEquals(exp, arrayX);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBugMeshgridOnDoubleArray(Nd4jBackend backend) {
        Nd4j.meshgrid(Nd4j.create(new double[] { 1, 2, 3 }), Nd4j.create(new double[] { 4, 5, 6 }));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeshGrid(){

        INDArray x1 = Nd4j.create(new double[]{1,2,3,4}).reshape(1, -1);
        INDArray y1 = Nd4j.create(new double[]{5,6,7}).reshape(1, -1);

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
        assertArrayEquals(exp, out1);

        INDArray[] out2 = Nd4j.meshgrid(x1.transpose(), y1.transpose());
        assertArrayEquals(exp, out2);

        INDArray[] out3 = Nd4j.meshgrid(x1, y1.transpose());
        assertArrayEquals(exp, out3);

        INDArray[] out4 = Nd4j.meshgrid(x1.transpose(), y1);
        assertArrayEquals(exp, out4);

        //Test views:
        INDArray x2 = Nd4j.create(1,9).get(NDArrayIndex.all(), NDArrayIndex.interval(1,2,7, true))
                .assign(x1);
        INDArray y2 = Nd4j.create(1,7).get(NDArrayIndex.all(), NDArrayIndex.interval(1,2,5, true))
                .assign(y1);

        INDArray[] out5 = Nd4j.meshgrid(x2, y2);
        assertArrayEquals(exp, out5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAccumuationWithoutAxis_1(Nd4jBackend backend) {
        val array = Nd4j.create(3, 3).assign(1.0);

        val result = array.sum();

        assertEquals(1, result.length());
        assertEquals(9.0, result.getDouble(0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSummaryStatsEquality_1(Nd4jBackend backend) {
//        log.info("Datatype: {}", Nd4j.dataType());

        for(boolean biasCorrected : new boolean[]{false, true}) {

            INDArray indArray1 = Nd4j.rand(1, 4, 10).castTo(DataType.DOUBLE);
            double std = indArray1.stdNumber(biasCorrected).doubleValue();

            val standardDeviation = new org.apache.commons.math3.stat.descriptive.moment.StandardDeviation(biasCorrected);
            double std2 = standardDeviation.evaluate(indArray1.data().asDouble());
//            log.info("Bias corrected = {}", biasCorrected);
//            log.info("nd4j std: {}", std);
//            log.info("apache math3 std: {}", std2);

            assertEquals(std, std2, 1e-5);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeanEdgeCase_C(){
        INDArray arr = Nd4j.linspace(1, 30,30, DataType.DOUBLE).reshape(new int[]{3,10,1}).dup('c');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));

        assertEquals(exp, arr2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeanEdgeCase_F(){
        INDArray arr = Nd4j.linspace(1, 30,30, DataType.DOUBLE).reshape(new int[]{3,10,1}).dup('f');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));

        assertEquals(exp, arr2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeanEdgeCase2_C(){
        INDArray arr = Nd4j.linspace(1, 60,60, DataType.DOUBLE).reshape(new int[]{3,10,2}).dup('c');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
        exp.addi(arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)));
        exp.divi(2);


        assertEquals(exp, arr2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeanEdgeCase2_F(){
        INDArray arr = Nd4j.linspace(1, 60,60, DataType.DOUBLE).reshape(new int[]{3,10,2}).dup('f');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
        exp.addi(arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)));
        exp.divi(2);


        assertEquals(exp, arr2);
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
    public void testRndBloat16(Nd4jBackend backend) {
        INDArray x  = Nd4j.rand(DataType.BFLOAT16 , 'c', new long[]{5});
        assertTrue(x.sumNumber().floatValue() > 0);

        x = Nd4j.randn(DataType.BFLOAT16 , 10);
        assertTrue(x.sumNumber().floatValue() != 0.0);
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
    public void testVariance_4D_1(Nd4jBackend backend) {
        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataType.FLOAT);

        val x = Nd4j.ones(10, 20, 30, 40);
        val result = x.var(false, 0, 2, 3);

        Nd4j.getExecutioner().commit();

//        log.info("Result shape: {}", result.shapeInfoDataBuffer().asLong());

        Nd4j.setDataType(dtype);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTranspose_Custom(){

        INDArray arr = Nd4j.linspace(1,15, 15, DataType.DOUBLE).reshape(5,3);
        INDArray out = Nd4j.create(3,5);

        val op = DynamicCustomOp.builder("transpose")
                .addInputs(arr)
                .addOutputs(out)
                .build();

        Nd4j.getExecutioner().exec(op);

        val exp = arr.transpose();
        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Execution(ExecutionMode.SAME_THREAD)
    public void testRowColumnOpsRank1(Nd4jBackend backend) {

        for( int i = 0; i < 6; i++ ) {
            INDArray orig = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape('c', 3, 4);
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyShapeRank0(){
        Nd4j.getRandom().setSeed(12345);
        int[] s = new int[0];
        INDArray create = Nd4j.create(s).castTo(DataType.DOUBLE);
        INDArray zeros = Nd4j.zeros(s).castTo(DataType.DOUBLE);
        INDArray ones = Nd4j.ones(s).castTo(DataType.DOUBLE);
        INDArray uninit = Nd4j.createUninitialized(s).assign(0).castTo(DataType.DOUBLE);
        INDArray rand = Nd4j.rand(s).castTo(DataType.DOUBLE);

        INDArray tsZero = Nd4j.scalar(0.0).castTo(DataType.DOUBLE);
        INDArray tsOne = Nd4j.scalar(1.0).castTo(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(12345);
        INDArray tsRand = Nd4j.scalar(Nd4j.rand(new int[]{1,1}).getDouble(0)).castTo(DataType.DOUBLE);
        assertEquals(tsZero, create);
        assertEquals(tsZero, zeros);
        assertEquals(tsOne, ones);
        assertEquals(tsZero, uninit);
        assertEquals(tsRand, rand);


        Nd4j.getRandom().setSeed(12345);
        long[] s2 = new long[0];
        create = Nd4j.create(s2).castTo(DataType.DOUBLE);
        zeros = Nd4j.zeros(s2).castTo(DataType.DOUBLE);
        ones = Nd4j.ones(s2).castTo(DataType.DOUBLE);
        uninit = Nd4j.createUninitialized(s2).assign(0).castTo(DataType.DOUBLE);
        rand = Nd4j.rand(s2).castTo(DataType.DOUBLE);

        assertEquals(tsZero, create);
        assertEquals(tsZero, zeros);
        assertEquals(tsOne, ones);
        assertEquals(tsZero, uninit);
        assertEquals(tsRand, rand);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarView_1(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 5, 5, DataType.DOUBLE);
        val exp = Nd4j.create(new double[]{1.0, 2.0, 5.0, 4.0, 5.0});
        val scalar = array.getScalar(2);

        assertEquals(3.0, scalar.getDouble(0), 1e-5);
        scalar.addi(2.0);

        assertEquals(exp, array);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarView_2(Nd4jBackend backend) {
        val array = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        val exp = Nd4j.create(new double[]{1.0, 2.0, 5.0, 4.0}).reshape(2, 2);
        val scalar = array.getScalar(1, 0);

        assertEquals(3.0, scalar.getDouble(0), 1e-5);
        scalar.addi(2.0);

        assertEquals(exp, array);
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
    public void testIndexesIteration_1(Nd4jBackend backend) {
        val arrayC = Nd4j.linspace(1,  60,  60, DataType.DOUBLE).reshape(3, 4, 5);
        val arrayF = arrayC.dup('f');

        val iter = new NdIndexIterator(arrayC.ordering(), arrayC.shape());
        while (iter.hasNext()) {
            val idx = iter.next();

            val c = arrayC.getDouble(idx);
            val f = arrayF.getDouble(idx);

            assertEquals(c, f, 1e-5);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIndexesIteration_2(Nd4jBackend backend) {
        val arrayC = Nd4j.linspace(1,  60,  60, DataType.DOUBLE).reshape(3, 4, 5);
        val arrayF = arrayC.dup('f');

        val iter = new NdIndexIterator(arrayC.ordering(), arrayC.shape());
        while (iter.hasNext()) {
            val idx = iter.next();

            var c = arrayC.getDouble(idx);
            var f = arrayF.getDouble(idx);

            arrayC.putScalar(idx,  c + 1.0);
            arrayF.putScalar(idx, f + 1.0);

            c = arrayC.getDouble(idx);
            f = arrayF.getDouble(idx);

            assertEquals(c, f, 1e-5);
        }
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPairwiseScalar_1(Nd4jBackend backend) {
        val exp_1 = Nd4j.create(new double[]{2.0, 3.0, 4.0}, new long[]{3});
        val exp_2 = Nd4j.create(new double[]{0.0, 1.0, 2.0}, new long[]{3});
        val exp_3 = Nd4j.create(new double[]{1.0, 2.0, 3.0}, new long[]{3});
        val arrayX = Nd4j.create(new double[]{1.0, 2.0, 3.0}, new long[]{3});
        val arrayY = Nd4j.scalar(1.0);

        val arrayZ_1 = arrayX.add(arrayY);
        assertEquals(exp_1, arrayZ_1);

        val arrayZ_2 = arrayX.sub(arrayY);
        assertEquals(exp_2, arrayZ_2);

        val arrayZ_3 = arrayX.div(arrayY);
        assertEquals(exp_3, arrayZ_3);

        val arrayZ_4 = arrayX.mul(arrayY);
        assertEquals(exp_3, arrayZ_4);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLTOE_1(Nd4jBackend backend) {
        val x = Nd4j.create(new double[]{1.0, 2.0, 3.0, -1.0});
        val y = Nd4j.create(new double[]{2.0, 2.0, 3.0, -2.0});

        val ex = Nd4j.create(new double[]{1.0, 2.0, 3.0, -1.0});
        val ey = Nd4j.create(new double[]{2.0, 2.0, 3.0, -2.0});

        val ez = Nd4j.create(new boolean[]{true, true, true, false});
        val z = Transforms.lessThanOrEqual(x, y, true);

        assertEquals(ex, x);
        assertEquals(ey, y);

        assertEquals(ez, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGTOE_1(Nd4jBackend backend) {
        val x = Nd4j.create(new double[]{1.0, 2.0, 3.0, -1.0});
        val y = Nd4j.create(new double[]{2.0, 2.0, 3.0, -2.0});

        val ex = Nd4j.create(new double[]{1.0, 2.0, 3.0, -1.0});
        val ey = Nd4j.create(new double[]{2.0, 2.0, 3.0, -2.0});

        val ez = Nd4j.create(new boolean[]{false, true, true, true}, new long[]{4}, DataType.BOOL);
        val z = Transforms.greaterThanOrEqual(x, y, true);

        val str = ez.toString();
//        log.info("exp: {}", str);

        assertEquals(ex, x);
        assertEquals(ey, y);

        assertEquals(ez, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastInvalid() {
        assertThrows(IllegalStateException.class,() -> {
            INDArray arr1 = Nd4j.ones(3,4,1);

            //Invalid op: y must match x/z dimensions 0 and 2
            INDArray arrInvalid = Nd4j.create(3,12);
            Nd4j.getExecutioner().exec(new BroadcastMulOp(arr1, arrInvalid, arr1, 0, 2));
            fail("Excepted exception on invalid input");
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGet(){
        //https://github.com/eclipse/deeplearning4j/issues/6133
        INDArray m = Nd4j.linspace(0,99,100, DataType.DOUBLE).reshape('c', 10,10);
        INDArray exp = Nd4j.create(new double[]{5, 15, 25, 35, 45, 55, 65, 75, 85, 95}, new int[]{10});
        INDArray col = m.getColumn(5);

        for(int i=0; i<10; i++ ){
            col.slice(i);
//            System.out.println(i + "\t" + col.slice(i));
        }

        //First element: index 5
        //Last element: index 95
        //91 total elements
        assertEquals(5, m.getDouble(5), 1e-6);
        assertEquals(95, m.getDouble(95), 1e-6);
        assertEquals(91, col.data().length());

        assertEquals(exp, col);
        assertEquals(exp.toString(), col.toString());
        assertArrayEquals(exp.toDoubleVector(), col.toDoubleVector(), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhere1(){

        INDArray arr = Nd4j.create(new boolean[][]{{false,true,false},{false,false,true},{false,false,true}});
        INDArray[] exp = new INDArray[]{
                Nd4j.createFromArray(new long[]{0,1,2}),
                Nd4j.createFromArray(new long[]{1,2,2})};

        INDArray[] act = Nd4j.where(arr, null, null);

        assertArrayEquals(exp, act);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhere2(){

        INDArray arr = Nd4j.create(DataType.BOOL, 3,3,3);
        arr.putScalar(0,1,0,1.0);
        arr.putScalar(1,2,1,1.0);
        arr.putScalar(2,2,1,1.0);
        INDArray[] exp = new INDArray[]{
                Nd4j.createFromArray(new long[]{0,1,2}),
                Nd4j.createFromArray(new long[]{1,2,2}),
                Nd4j.createFromArray(new long[]{0,1,1})
        };

        INDArray[] act = Nd4j.where(arr, null, null);

        assertArrayEquals(exp, act);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhere3(){
        INDArray arr = Nd4j.create(new boolean[][]{{false,true,false},{false,false,true},{false,false,true}});
        INDArray x = Nd4j.valueArrayOf(3, 3, 1.0);
        INDArray y = Nd4j.valueArrayOf(3, 3, 2.0);
        INDArray exp = Nd4j.create(new double[][]{
                {1,2,1},
                {1,1,2},
                {1,1,2}});

        INDArray[] act = Nd4j.where(arr, x, y);
        assertEquals(1, act.length);

        assertEquals(exp, act[0]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereEmpty(){
        INDArray inArray = Nd4j.zeros(2, 3);
        inArray.putScalar(0, 0, 10.0f);
        inArray.putScalar(1, 2, 10.0f);

        INDArray mask1 = inArray.match(1, Conditions.greaterThanOrEqual(1));

        assertEquals(1, mask1.castTo(DataType.INT).maxNumber().intValue()); // ! Not Empty Match

        INDArray[] matchIndexes = Nd4j.where(mask1, null, null);

        assertArrayEquals(new int[] {0, 1}, matchIndexes[0].toIntVector());
        assertArrayEquals(new int[] {0, 2}, matchIndexes[1].toIntVector());

        INDArray mask2 = inArray.match(11, Conditions.greaterThanOrEqual(11));

        assertEquals(0, mask2.castTo(DataType.INT).maxNumber().intValue());

        INDArray[] matchIndexes2 = Nd4j.where(mask2, null, null);
        for( int i = 0; i < matchIndexes2.length; i++) {
            assertTrue(matchIndexes2[i].isEmpty());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarEquality_1(Nd4jBackend backend) {
        val x = Nd4j.scalar(1.0f);
        val e = Nd4j.scalar(3.0f);

        x.addi(2.0f);

        assertEquals(e, x);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStack(){
        INDArray in = Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape(3,4);
        INDArray in2 = in.add(100);

        for( int i=-3; i<3; i++ ){
            INDArray out = Nd4j.stack(i, in, in2);
            long[] expShape;
            switch (i){
                case -3:
                case 0:
                    expShape = new long[]{2,3,4};
                    break;
                case -2:
                case 1:
                    expShape = new long[]{3,2,4};
                    break;
                case -1:
                case 2:
                    expShape = new long[]{3,4,2};
                    break;
                default:
                    throw new RuntimeException(String.valueOf(i));
            }
            assertArrayEquals(expShape, out.shape(),String.valueOf(i));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutSpecifiedIndex(){
        long[][] ss = new long[][]{{3,4}, {3,4,5}, {3,4,5,6}};
        long[][] st = new long[][]{{4,4}, {4,4,5}, {4,4,5,6}};
        long[][] ds = new long[][]{{1,4}, {1,4,5}, {1,4,5,6}};

        for( int test=0; test<ss.length; test++ ) {
            long[] shapeSource = ss[test];
            long[] shapeTarget = st[test];
            long[] diffShape = ds[test];

            final INDArray source = Nd4j.ones(shapeSource);
            final INDArray target = Nd4j.zeros(shapeTarget);

            final INDArrayIndex[] targetIndexes = new INDArrayIndex[shapeTarget.length];
            Arrays.fill(targetIndexes, NDArrayIndex.all());
            int[] arr = new int[(int) shapeSource[0]];
            for (int i = 0; i < arr.length; i++) {
                arr[i] = i;
            }
            targetIndexes[0] = new SpecifiedIndex(arr);

            // Works
            //targetIndexes[0] = NDArrayIndex.interval(0, shapeSource[0]);

            target.put(targetIndexes, source);
            final INDArray expected = Nd4j.concat(0, Nd4j.ones(shapeSource), Nd4j.zeros(diffShape));
            assertEquals(expected, target,"Expected array to be set!");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutSpecifiedIndices2d(){

        INDArray arr = Nd4j.create(3,4);
        INDArray toPut = Nd4j.create(new double[]{1,2,3,4}, new int[]{2,2}, 'c');
        INDArrayIndex[] indices = new INDArrayIndex[]{
                NDArrayIndex.indices(0,2),
                NDArrayIndex.indices(1,3)} ;

        INDArray exp = Nd4j.create(new double[][]{
                {0,1,0,2},
                {0,0,0,0},
                {0,3,0,4}});

        arr.put(indices, toPut);
        assertEquals(exp, arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutSpecifiedIndices3d() {

        INDArray arr = Nd4j.create(2,3,4);
        INDArray toPut = Nd4j.create(new double[]{1,2,3,4}, new int[]{1,2,2}, 'c');
        INDArrayIndex[] indices = new INDArrayIndex[]{
                NDArrayIndex.point(1),
                NDArrayIndex.indices(0,2),
                NDArrayIndex.indices(1,3)};

        INDArray exp = Nd4j.create(2,3,4);
        exp.putScalar(1, 0, 1, 1);
        exp.putScalar(1, 0, 3, 2);
        exp.putScalar(1, 2, 1, 3);
        exp.putScalar(1, 2, 3, 4);

        arr.put(indices, toPut);
        assertEquals(exp, arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpecifiedIndexArraySize1(Nd4jBackend backend) {
        long[] shape = {2, 2, 2, 2};
        INDArray in = Nd4j.create(shape);
        INDArrayIndex[] idx1 = new INDArrayIndex[]{NDArrayIndex.all(), new SpecifiedIndex(0), NDArrayIndex.all(), NDArrayIndex.all()};

        INDArray arr = in.get(idx1);
        long[] expShape = new long[]{2,1,2,2};
        assertArrayEquals(expShape, arr.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTransposei(){
        INDArray arr = Nd4j.linspace(1,12,12).reshape('c',3,4);

        INDArray ti = arr.transposei();
        assertArrayEquals(new long[]{4,3}, ti.shape());
        assertArrayEquals(new long[]{4,3}, arr.shape());

        assertTrue(arr == ti);  //Should be same object
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStatistics_1(Nd4jBackend backend) {
        val array = Nd4j.createFromArray(new float[] {-1.0f, 0.0f, 1.0f});
        val stats = Nd4j.getExecutioner().inspectArray(array);

        assertEquals(1, stats.getCountPositive());
        assertEquals(1, stats.getCountNegative());
        assertEquals(1, stats.getCountZero());
        assertEquals(0.0f, stats.getMeanValue(), 1e-5);
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
    public void testInvalidOrder(){

        try {
            Nd4j.create(new int[]{1}, 'z');
            fail("Expected failure");
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().toLowerCase().contains("order"));
        }

        try {
            Nd4j.zeros(1, 'z');
            fail("Expected failure");
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().toLowerCase().contains("order"));
        }

        try {
            Nd4j.zeros(new int[]{1}, 'z');
            fail("Expected failure");
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().toLowerCase().contains("order"));
        }

        try {
            Nd4j.create(new long[]{1}, 'z');
            fail("Expected failure");
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().toLowerCase().contains("order"));
        }

        try {
            Nd4j.rand('z', 1, 1).castTo(DataType.DOUBLE);
            fail("Expected failure");
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().toLowerCase().contains("order"));
        }

        try {
            Nd4j.createUninitialized(new int[]{1}, 'z');
            fail("Expected failure");
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().toLowerCase().contains("order"));
        }

        try {
            Nd4j.createUninitialized(new long[]{1}, 'z');
            fail("Expected failure");
        } catch (IllegalArgumentException e){
            assertTrue(e.getMessage().toLowerCase().contains("order"));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssignValid(){
        INDArray arr1 = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);
        INDArray arr2 = Nd4j.create(3,4);
        arr2.assign(arr1);
        assertEquals(arr1, arr2);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyCasting(){
        for(val from : DataType.values()) {
            if (from == DataType.UTF8 || from == DataType.UNKNOWN || from == DataType.COMPRESSED)
                continue;

            for(val to : DataType.values()){
                if (to == DataType.UTF8 || to == DataType.UNKNOWN || to == DataType.COMPRESSED)
                    continue;

                INDArray emptyFrom = Nd4j.empty(from);
                INDArray emptyTo = emptyFrom.castTo(to);

                String str = from + " -> " + to;

                assertEquals(from, emptyFrom.dataType(),str);
                assertTrue(emptyFrom.isEmpty(),str);
                assertEquals(0, emptyFrom.length(),str);

                assertEquals(to, emptyTo.dataType(),str);
                assertTrue(emptyTo.isEmpty(),str);
                assertEquals(0, emptyTo.length(),str);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVStackRank1(){
        List<INDArray> list = new ArrayList<>();
        list.add(Nd4j.linspace(1,3,3, DataType.DOUBLE));
        list.add(Nd4j.linspace(4,6,3, DataType.DOUBLE));
        list.add(Nd4j.linspace(7,9,3, DataType.DOUBLE));

        INDArray out = Nd4j.vstack(list);
        INDArray exp = Nd4j.createFromArray(new double[][]{
                {1,2,3},
                {4,5,6},
                {7,8,9}});
        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAxpyOpRows(){
        INDArray arr = Nd4j.create(1,4).assign(2.0f);
        INDArray ones = Nd4j.ones(1,4).assign(3.0f);

        Nd4j.exec(new Axpy(arr, ones, arr, 10.0, 4));

        INDArray exp = Nd4j.valueArrayOf(new long[]{1,4}, 23.0);

        assertEquals(exp, arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyArray(Nd4jBackend backend) {
        INDArray empty = Nd4j.empty(DataType.INT);
        assertEquals(empty.toString(), "[]");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLinspaceWithStep() {
        double lower = -0.9, upper = 0.9, step = 0.2;
        INDArray in = Nd4j.linspace(DataType.DOUBLE,lower,step,10);
        for (int i = 0; i < 10; ++i) {
            assertEquals(lower + step * i, in.getDouble(i), 1e-5);
        }

        step = 0.3;
        INDArray stepped = Nd4j.linspace(DataType.DOUBLE, lower, step, 10);
        for (int i = 0; i < 10; ++i) {
            assertEquals(lower + i * step, stepped.getDouble(i),1e-5);
        }

        lower = 0.9;
        upper = -0.9;
        step = -0.2;
        in = Nd4j.linspace(lower, upper, 10, DataType.DOUBLE);
        for (int i = 0; i < 10; ++i) {
            assertEquals(lower + i * step, in.getDouble(i),  1e-5);
        }

        stepped = Nd4j.linspace(DataType.DOUBLE, lower, step, 10);
        for (int i = 0; i < 10; ++i) {
            assertEquals(lower + i * step, stepped.getDouble(i),  1e-5);
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLinspaceWithStepForIntegers(){

        long lower = -9, upper = 9, step = 2;
        INDArray in = Nd4j.linspace(lower, upper, 10, DataType.LONG);
        for (int i = 0; i < 10; ++i) {
            assertEquals(lower + step * i, in.getInt(i));
        }

        INDArray stepped = Nd4j.linspace(DataType.INT, lower, 10, step);
        for (int i = 0; i < 10; ++i) {
            assertEquals(lower + i * step, stepped.getInt(i));
        }

        lower = 9;
        upper = -9;
        step = -2;
        in = Nd4j.linspace(lower, upper, 10, DataType.INT);
        for (int i = 0; i < 10; ++i) {
            assertEquals(lower + i * step, in.getInt(i));
        }
        lower = 9;
        step = -2;
        INDArray stepped2 = Nd4j.linspace(DataType.INT, lower, 10, step);
        for (int i = 0; i < 10; ++i) {
            assertEquals(lower + i * step, stepped2.getInt(i));
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled()
    public void testArangeWithStep(Nd4jBackend backend) {
        int begin = -9, end = 9, step = 2;
        INDArray in = Nd4j.arange(begin, end, step);
        assertEquals(in.getInt(0), -9);
        assertEquals(in.getInt(1), -7);
        assertEquals(in.getInt(2), -5);
        assertEquals(in.getInt(3), -3);
        assertEquals(in.getInt(4), -1);
        assertEquals(in.getInt(5), 1);
        assertEquals(in.getInt(6), 3);
        assertEquals(in.getInt(7), 5);
        assertEquals(in.getInt(8), 7);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("Crashes")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testRollingMean(Nd4jBackend backend) {
        val wsconf = WorkspaceConfiguration.builder()
                .initialSize(4L * (32*128*256*256 + 32*128 + 10*1024*1024))
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policySpill(SpillPolicy.FAIL)
                .build();

        String wsName = "testRollingMeanWs";
        try {
            System.gc();
            int iterations1 = isIntegrationTests() ? 5 : 2;
            for (int e = 0; e < 5; e++) {
                try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsconf, wsName)) {
                    val array = Nd4j.create(DataType.FLOAT, 32, 128, 256, 256);
                    array.mean(2, 3);
                }
            }

            int iterations = isIntegrationTests() ? 20 : 3;
            val timeStart = System.nanoTime();
            for (int e = 0; e < iterations; e++) {
                try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsconf, wsName)) {
                    val array = Nd4j.create(DataType.FLOAT, 32, 128, 256, 256);
                    array.mean(2, 3);
                }
            }
            val timeEnd = System.nanoTime();
            log.info("Average time: {} ms", (timeEnd - timeStart) / (double) iterations / (double) 1000 / (double) 1000);
        } finally {
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testZerosRank1(Nd4jBackend backend) {
        Nd4j.zeros(new int[] { 2 }, DataType.DOUBLE);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeEnforce(){

        INDArray arr = Nd4j.create(new long[]{2,2}, 'c');
        INDArray arr2 = arr.reshape('c', true, 4, 1);

        INDArray arr1a = Nd4j.create(new long[]{2,3}, 'c').get(NDArrayIndex.all(), NDArrayIndex.interval(0,2));
        INDArray arr3 = arr1a.reshape('c', false, 4,1);
        boolean isView = arr3.isView();
        assertFalse(isView);     //Should be copy

        try{
            INDArray arr4 = arr1a.reshape('c', true, 4,1);
            fail("Expected exception");
        } catch (ND4JIllegalStateException e){
            assertTrue(e.getMessage().contains("Unable to reshape array as view"),e.getMessage());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRepeatSimple(){

        INDArray arr = Nd4j.createFromArray(new double[][]{
                {1,2,3},{4,5,6}});

        INDArray r0 = arr.repeat(0, 2);

        INDArray exp0 = Nd4j.createFromArray(new double[][]{
                {1,2,3},
                {1,2,3},
                {4,5,6},
                {4,5,6}});

        assertEquals(exp0, r0);


        INDArray r1 = arr.repeat(1, 2);
        INDArray exp1 = Nd4j.createFromArray(new double[][]{
                {1,1,2,2,3,3},{4,4,5,5,6,6}});
        assertEquals(exp1, r1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowsEdgeCaseView(){

        INDArray arr = Nd4j.linspace(0, 9, 10, DataType.DOUBLE).reshape('f', 5, 2).dup('c');    //0,1,2... along columns
        INDArray view = arr.getColumn(0);
        assertEquals(Nd4j.createFromArray(0.0, 1.0, 2.0, 3.0, 4.0), view);
        int[] idxs = new int[]{0,2,3,4};

        INDArray out = Nd4j.pullRows(view.reshape(5, 1), 1, idxs);
        INDArray exp = Nd4j.createFromArray(new double[]{0,2,3,4}).reshape(4, 1);

        assertEquals(exp, out);   //Failing here
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsFailure(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class,() -> {
            val idxs = new int[]{0,2,3,4};
            val out = Nd4j.pullRows(Nd4j.createFromArray(0.0, 1.0, 2.0, 3.0, 4.0), 0, idxs);
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRepeatStrided(Nd4jBackend backend) {

        // Create a 2D array (shape 5x5)
        INDArray array = Nd4j.arange(25).reshape(5, 5);

        // Get first column (shape 5x1)
        INDArray slice = array.get(NDArrayIndex.all(), NDArrayIndex.point(0)).reshape(5,1);

        // Repeat column on sliced array (shape 5x3)
        INDArray repeatedSlice = slice.repeat(1, (long) 3);

        // Same thing but copy array first
        INDArray repeatedDup = slice.dup().repeat(1, (long) 3);

        // Check result
        assertEquals(repeatedSlice, repeatedDup);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeshgridDtypes(Nd4jBackend backend) {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        Nd4j.meshgrid(Nd4j.create(new double[] { 1, 2, 3 }), Nd4j.create(new double[] { 4, 5, 6 }));

        Nd4j.meshgrid(Nd4j.createFromArray(1, 2, 3), Nd4j.createFromArray(4, 5, 6));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetColumnRowVector(){
        INDArray arr = Nd4j.create(1,4);
        INDArray col = arr.getColumn(0);
//        System.out.println(Arrays.toString(col.shape()));
        assertArrayEquals(new long[]{1}, col.shape());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyArrayReuse(){
        //Empty arrays are immutable - no point creating them multiple times
        INDArray ef1 = Nd4j.empty(DataType.FLOAT);
        INDArray ef2 = Nd4j.empty(DataType.FLOAT);
        assertTrue(ef1 == ef2);       //Should be exact same object

        INDArray el1 = Nd4j.empty(DataType.LONG);
        INDArray el2 = Nd4j.empty(DataType.LONG);
        assertTrue(el1 == el2);       //Should be exact same object
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMaxViewF(){
        INDArray arr = Nd4j.create(DataType.DOUBLE, new long[]{8,2}, 'f').assign(999);

        INDArray view = arr.get(NDArrayIndex.interval(3,5), NDArrayIndex.all());
        view.assign(Nd4j.createFromArray(new double[][]{{1,2},{3,4}}));

        assertEquals(Nd4j.create(new double[]{3,4}), view.max(0));
        assertEquals(Nd4j.create(new double[]{2,4}), view.max(1));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMin2(){
        INDArray x = Nd4j.createFromArray(new double[][]{
                {-999,       0.2236,    0.7973,    0.0962},
                { 0.7231,    0.3381,   -0.7301,    0.9115},
                {-0.5094,    0.9749,   -2.1340,    0.6023}});

        INDArray out = Nd4j.create(DataType.DOUBLE, 4);
        Nd4j.exec(DynamicCustomOp.builder("reduce_min")
                .addInputs(x)
                .addOutputs(out)
                .addIntegerArguments(0)
                .build());

        INDArray exp = Nd4j.createFromArray(-999, 0.2236, -2.1340, 0.0962);
        assertEquals(exp, out); //Fails here


        INDArray out1 = Nd4j.create(DataType.DOUBLE, 3);
        Nd4j.exec(DynamicCustomOp.builder("reduce_min")
                .addInputs(x)
                .addOutputs(out1)
                .addIntegerArguments(1)
                .build());

        INDArray exp1 = Nd4j.createFromArray(-999, -0.7301, -2.1340);
        assertEquals(exp1, out1); //This is OK
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutRowValidation(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class,() -> {
            val matrix = Nd4j.create(5, 10);
            val row = Nd4j.create(25);

            matrix.putRow(1, row);
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutColumnValidation(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class,() -> {
            val matrix = Nd4j.create(5, 10);
            val column = Nd4j.create(25);

            matrix.putColumn(1, column);
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateF(){
        char origOrder = Nd4j.order();
        try {
            Nd4j.factory().setOrder('f');


            INDArray arr = Nd4j.createFromArray(new double[][]{{1, 2, 3}, {4, 5, 6}});
            INDArray arr2 = Nd4j.createFromArray(new float[][]{{1, 2, 3}, {4, 5, 6}});
            INDArray arr3 = Nd4j.createFromArray(new int[][]{{1, 2, 3}, {4, 5, 6}});
            INDArray arr4 = Nd4j.createFromArray(new long[][]{{1, 2, 3}, {4, 5, 6}});
            INDArray arr5 = Nd4j.createFromArray(new short[][]{{1, 2, 3}, {4, 5, 6}});
            INDArray arr6 = Nd4j.createFromArray(new byte[][]{{1, 2, 3}, {4, 5, 6}});

            INDArray exp = Nd4j.create(2, 3);
            exp.putScalar(0, 0, 1.0);
            exp.putScalar(0, 1, 2.0);
            exp.putScalar(0, 2, 3.0);
            exp.putScalar(1, 0, 4.0);
            exp.putScalar(1, 1, 5.0);
            exp.putScalar(1, 2, 6.0);

            assertEquals(exp, arr);
            assertEquals(exp.castTo(DataType.FLOAT), arr2);
            assertEquals(exp.castTo(DataType.INT), arr3);
            assertEquals(exp.castTo(DataType.LONG), arr4);
            assertEquals(exp.castTo(DataType.SHORT), arr5);
            assertEquals(exp.castTo(DataType.BYTE), arr6);
        } finally {
            Nd4j.factory().setOrder(origOrder);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduceKeepDimsShape(){
        INDArray arr = Nd4j.create(3,4);
        INDArray out = arr.sum(true, 1);
        assertArrayEquals(new long[]{3, 1}, out.shape());

        INDArray out2 = arr.sum(true, 0);
        assertArrayEquals(new long[]{1, 4}, out2.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSliceRow(){
        double[] data = new double[]{15.0, 16.0};
        INDArray vector = Nd4j.createFromArray(data).reshape(1,2);
        INDArray slice = vector.slice(0);
//        System.out.println(slice.shapeInfoToString());
        assertEquals(vector.reshape(2), slice);
        slice.assign(-1);
        assertEquals(Nd4j.createFromArray(-1.0, -1.0).reshape(1,2), vector);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSliceMatrix(){
        INDArray arr = Nd4j.arange(4).reshape(2,2);
        arr.slice(0);
        arr.slice(1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarEq(Nd4jBackend backend){
        INDArray scalarRank2 = Nd4j.scalar(10.0).reshape(1,1);
        INDArray scalarRank1 = Nd4j.scalar(10.0).reshape(1);
        INDArray scalarRank0 = Nd4j.scalar(10.0);

        assertNotEquals(scalarRank0, scalarRank2);
        assertNotEquals(scalarRank0, scalarRank1);
        assertNotEquals(scalarRank1, scalarRank2);
        assertEquals(scalarRank0, scalarRank0.dup());
        assertEquals(scalarRank1, scalarRank1.dup());
        assertEquals(scalarRank2, scalarRank2.dup());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetWhereINDArray(Nd4jBackend backend) {
        INDArray input = Nd4j.create(new double[] { 1, -3, 4, 8, -2, 5 });
        INDArray comp = Nd4j.create(new double[]{2, -3, 1, 1, -2, 1 });
        INDArray expected = Nd4j.create(new double[] { 4, 8, 5 });
        INDArray actual = input.getWhere(comp, Conditions.greaterThan(1));

        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetWhereNumber(Nd4jBackend backend) {
        INDArray input = Nd4j.create(new double[] { 1, -3, 4, 8, -2, 5 });
        INDArray expected = Nd4j.create(new double[] { 8, 5 });
        INDArray actual = input.getWhere(4, Conditions.greaterThan(1));

        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testType1(Nd4jBackend backend) throws IOException {
        for (int i = 0; i < 10; ++i) {
            INDArray in1 = Nd4j.rand(DataType.DOUBLE, new int[]{100, 100}).castTo(DataType.DOUBLE);
            File dir = testDir.toFile();
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(dir,"test.bin")));
            oos.writeObject(in1);

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(dir,"test.bin")));
            INDArray in2 = null;
            try {
                in2 = (INDArray) ois.readObject();
            } catch(ClassNotFoundException e) {

            }

            assertEquals(in1, in2);
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOnes(Nd4jBackend backend){
        INDArray arr = Nd4j.ones();
        INDArray arr2 = Nd4j.ones(DataType.LONG);
        assertEquals(0, arr.rank());
        assertEquals(1, arr.length());
        assertEquals(0, arr2.rank());
        assertEquals(1, arr2.length());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testZeros(Nd4jBackend backend){
        INDArray arr = Nd4j.zeros();
        INDArray arr2 = Nd4j.zeros(DataType.LONG);
        assertEquals(0, arr.rank());
        assertEquals(1, arr.length());
        assertEquals(0, arr2.rank());
        assertEquals(1, arr2.length());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
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
            } catch(ClassNotFoundException e) {

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
            } catch(ClassNotFoundException e) {

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
            } catch(ClassNotFoundException e) {

            }

            assertEquals(in1, in2);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToXMatrix(){

        List<long[]> shapes = Arrays.asList(new long[]{3, 4}, new long[]{3, 1}, new long[]{1,3});
        for(long[] shape : shapes){
            long length = ArrayUtil.prodLong(shape);
            INDArray orig = Nd4j.arange(length).castTo(DataType.DOUBLE).reshape(shape);
            for(DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.INT,
                    DataType.LONG, DataType.SHORT, DataType.UBYTE, DataType.UINT16, DataType.UINT32, DataType.UINT64, DataType.BFLOAT16}) {
                INDArray arr = orig.castTo(dt);

                float[][] fArr = arr.toFloatMatrix();
                double[][] dArr = arr.toDoubleMatrix();
                int[][] iArr = arr.toIntMatrix();
                long[][] lArr = arr.toLongMatrix();

                INDArray f = Nd4j.createFromArray(fArr).castTo(dt);
                INDArray d = Nd4j.createFromArray(dArr).castTo(dt);
                INDArray i = Nd4j.createFromArray(iArr).castTo(dt);
                INDArray l = Nd4j.createFromArray(lArr).castTo(dt);

                assertEquals(arr, f);
                assertEquals(arr, d);
                assertEquals(arr, i);
                assertEquals(arr, l);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToXVector(){

        List<long[]> shapes = Arrays.asList(new long[]{3}, new long[]{3, 1}, new long[]{1,3});
        for(long[] shape : shapes){
            INDArray orig = Nd4j.arange(3).castTo(DataType.DOUBLE).reshape(shape);
            for(DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.INT,
                    DataType.LONG, DataType.SHORT, DataType.UBYTE, DataType.UINT16, DataType.UINT32, DataType.UINT64, DataType.BFLOAT16}) {
                INDArray arr = orig.castTo(dt);

                float[] fArr = arr.toFloatVector();
                double[] dArr = arr.toDoubleVector();
                int[] iArr = arr.toIntVector();
                long[] lArr = arr.toLongVector();

                INDArray f = Nd4j.createFromArray(fArr).castTo(dt).reshape(shape);
                INDArray d = Nd4j.createFromArray(dArr).castTo(dt).reshape(shape);
                INDArray i = Nd4j.createFromArray(iArr).castTo(dt).reshape(shape);
                INDArray l = Nd4j.createFromArray(lArr).castTo(dt).reshape(shape);

                assertEquals(arr, f);
                assertEquals(arr, d);
                assertEquals(arr, i);
                assertEquals(arr, l);
            }
        }
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumEdgeCase(){
        INDArray row = Nd4j.create(1,3);
        INDArray sum = row.sum(0);
        assertArrayEquals(new long[]{3}, sum.shape());

        INDArray twoD = Nd4j.create(2,3);
        INDArray sum2 = twoD.sum(0);
        assertArrayEquals(new long[]{3}, sum2.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMedianEdgeCase(){
        INDArray rowVec = Nd4j.rand(DataType.FLOAT, 1, 10);
        INDArray median = rowVec.median(0);
        assertEquals(rowVec.reshape(10), median);

        INDArray colVec = Nd4j.rand(DataType.FLOAT, 10, 1);
        median = colVec.median(1);
        assertEquals(colVec.reshape(10), median);

        //Non-edge cases:
        rowVec.median(1);
        colVec.median(0);

        //full array case:
        rowVec.median();
        colVec.median();
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
    public void testCreateDtypes(Nd4jBackend backend) {
        int[] sliceShape = new int[] {9};
        float[] arrays = new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        double [] arrays_double = new double[] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

        INDArray x = Nd4j.create( sliceShape, arrays, arrays );
        assertEquals(DataType.FLOAT, x.dataType());

        INDArray xd = Nd4j.create( sliceShape, arrays_double, arrays_double  );
        assertEquals(DataType.DOUBLE, xd.dataType());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateShapeValidation(){
        try {
            Nd4j.create(new double[]{1, 2, 3}, new int[]{1, 1});
            fail();
        } catch (Exception t){
            assertTrue(t.getMessage().contains("length"));
        }

        try {
            Nd4j.create(new float[]{1, 2, 3}, new int[]{1, 1});
            fail();
        } catch (Exception t){
            assertTrue(t.getMessage().contains("length"));
        }

        try {
            Nd4j.create(new byte[]{1, 2, 3}, new long[]{1, 1}, DataType.BYTE);
            fail();
        } catch (Exception t){
            assertTrue(t.getMessage().contains("length"));
        }

        try {
            Nd4j.create(new double[]{1, 2, 3}, new int[]{1, 1}, 'c');
            fail();
        } catch (Exception t){
            assertTrue(t.getMessage().contains("length"));
        }
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBatchToSpace(Nd4jBackend backend) {

        INDArray out = Nd4j.create(DataType.FLOAT, 2, 4, 5);
        DynamicCustomOp c = new BatchToSpaceND();

        c.addInputArgument(
                Nd4j.rand(DataType.FLOAT, new int[]{4, 4, 3}),
                Nd4j.createFromArray(1, 2),
                Nd4j.createFromArray(new int[][]{ new int[]{0, 0}, new int[]{0, 1} })
        );
        c.addOutputArgument(out);
        Nd4j.getExecutioner().exec(c);

        List<DataBuffer> l = c.calculateOutputShape();


        //from [4,4,3] to [2,4,6] then crop to [2,4,5]
    }
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToFromByteArray() throws IOException {
        // simple test to get rid of toByteArray and fromByteArray compiler warnings.
        INDArray x = Nd4j.arange(10);
        byte[] xb = Nd4j.toByteArray(x);
        INDArray y = Nd4j.fromByteArray(xb);
        assertEquals(x,y);
    }

    private static INDArray fwd(INDArray input, INDArray W, INDArray b){
        INDArray ret = Nd4j.createUninitialized(input.size(0), W.size(1)).castTo(DataType.DOUBLE);
        input.mmuli(W, ret);
        ret.addiRowVector(b);
        return ret;
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVStackHStack1d(Nd4jBackend backend) {
        INDArray rowVector1 = Nd4j.create(new double[]{1,2,3});
        INDArray rowVector2 = Nd4j.create(new double[]{4,5,6});

        INDArray vStack = Nd4j.vstack(rowVector1, rowVector2);      //Vertical stack:   [3]+[3] to [2,3]
        INDArray hStack = Nd4j.hstack(rowVector1, rowVector2);      //Horizontal stack: [3]+[3] to [6]

        assertEquals(Nd4j.createFromArray(1.0,2,3,4,5,6).reshape('c', 2, 3), vStack);
        assertEquals(Nd4j.createFromArray(1.0,2,3,4,5,6), hStack);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduceAll_1(Nd4jBackend backend) {
        val x = Nd4j.empty(DataType.FLOAT);
        val e = Nd4j.scalar(true);
        val z = Nd4j.exec(new All(x));

        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduceAll_2(Nd4jBackend backend) {
        val x = Nd4j.ones(DataType.FLOAT, 0);
        val e = Nd4j.scalar(true);
        val z = Nd4j.exec(new All(x));

        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduceAll_3(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 0);
        assertEquals(1, x.rank());

        val e = Nd4j.scalar(true);
        val z = Nd4j.exec(new All(x, 0));

        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarEqualsNoResult(){
        INDArray out = Nd4j.exec(new ScalarEquals(Nd4j.createFromArray(-2, -1, 0, 1, 2), null, 0));
        INDArray exp = Nd4j.createFromArray(false, false, true, false, false);
        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutOverwrite(){
        INDArray arr = Nd4j.create(DataType.DOUBLE, 10);
        arr.putScalar(0, 10);
        System.out.println(arr);
        INDArray arr2 = Nd4j.createFromArray(3.0, 3.0, 3.0);
        val view = arr.get(new INDArrayIndex[]{NDArrayIndex.interval(1, 4)});
        view.assign(arr2);
        System.out.println(arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyReshapingMinus1(){
        INDArray arr0 = Nd4j.create(DataType.FLOAT, 2, 0);
        INDArray arr1 = Nd4j.create(DataType.FLOAT, 0, 1, 2);

        INDArray out0 = Nd4j.exec(new Reshape(arr0, Nd4j.createFromArray(2, 0, -1)))[0];
        INDArray out1 = Nd4j.exec(new Reshape(arr1, Nd4j.createFromArray(-1, 1)))[0];
        INDArray out2 = Nd4j.exec(new Reshape(arr1, Nd4j.createFromArray(10, -1)))[0];

        assertArrayEquals(new long[]{2, 0, 1}, out0.shape());
        assertArrayEquals(new long[]{0, 1}, out1.shape());
        assertArrayEquals(new long[]{10, 0}, out2.shape());
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
        System.out.println(Arrays.toString(ret[0].shape()));
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LARGE_RESOURCES)
    @Tag(TagNames.LONG_TEST)
    public void testCreateBufferFromByteBuffer(Nd4jBackend backend){

        for(DataType dt : DataType.values()){
            if(dt == DataType.COMPRESSED || dt == DataType.UTF8 || dt == DataType.UNKNOWN)
                continue;

            int lengthBytes = 256;
            int lengthElements = lengthBytes / dt.width();
            ByteBuffer bb = ByteBuffer.allocateDirect(lengthBytes);

            DataBuffer db = Nd4j.createBuffer(bb, dt, lengthElements);
            INDArray arr = Nd4j.create(db, new long[]{lengthElements});

            arr.toStringFull();
            arr.toString();

            for(DataType dt2 : DataType.values()) {
                if (dt2 == DataType.COMPRESSED || dt2 == DataType.UTF8 || dt2 == DataType.UNKNOWN)
                    continue;
                INDArray a2 = arr.castTo(dt2);
                a2.toStringFull();
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateBufferFromByteBufferViews(){

        for(DataType dt : DataType.values()){
            if(dt == DataType.COMPRESSED || dt == DataType.UTF8 || dt == DataType.UNKNOWN)
                continue;
//            System.out.println(dt);

            int lengthBytes = 256;
            int lengthElements = lengthBytes / dt.width();
            ByteBuffer bb = ByteBuffer.allocateDirect(lengthBytes);

            DataBuffer db = Nd4j.createBuffer(bb, dt, lengthElements);
            INDArray arr = Nd4j.create(db, new long[]{lengthElements/2, 2});

            arr.toStringFull();

            INDArray view = arr.get(NDArrayIndex.all(), NDArrayIndex.point(0));
            INDArray view2 = arr.get(NDArrayIndex.point(1), NDArrayIndex.all());

            view.toStringFull();
            view2.toStringFull();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTypeCastingToString(){

        for(DataType dt : DataType.values()) {
            if (dt == DataType.COMPRESSED || dt == DataType.UTF8 || dt == DataType.UNKNOWN)
                continue;
            INDArray a1 = Nd4j.create(dt, 10);
            for(DataType dt2 : DataType.values()) {
                if (dt2 == DataType.COMPRESSED || dt2 == DataType.UTF8 || dt2 == DataType.UNKNOWN)
                    continue;

                INDArray a2 = a1.castTo(dt2);
                a2.toStringFull();
            }
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShape0Casts(){
        for(DataType dt : DataType.values()){
            if(!dt.isNumerical())
                continue;

            INDArray a1 = Nd4j.create(dt, 1,0,2);

            for(DataType dt2 : DataType.values()){
                if(!dt2.isNumerical())
                    continue;
                INDArray a2 = a1.castTo(dt2);

                assertArrayEquals(a1.shape(), a2.shape());
                assertEquals(dt2, a2.dataType());
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSmallSort(){
        INDArray arr = Nd4j.createFromArray(0.5, 0.4, 0.1, 0.2);
        INDArray expected = Nd4j.createFromArray(0.1, 0.2, 0.4, 0.5);
        INDArray sorted = Nd4j.sort(arr, true);
        assertEquals(expected, sorted);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
