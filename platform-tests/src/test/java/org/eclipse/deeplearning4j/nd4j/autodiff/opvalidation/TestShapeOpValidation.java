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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.math3.linear.LUDecomposition;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.custom.Tri;
import org.nd4j.linalg.api.ops.custom.Triu;
import org.nd4j.linalg.api.ops.impl.shape.DiagPart;
import org.nd4j.linalg.api.ops.impl.shape.MergeMaxIndex;
import org.nd4j.linalg.api.ops.impl.shape.Permute;
import org.nd4j.linalg.api.ops.impl.shape.SequenceMask;
import org.nd4j.linalg.api.ops.impl.shape.SizeAt;
import org.nd4j.linalg.api.ops.impl.shape.Transpose;
import org.nd4j.linalg.api.ops.impl.shape.Unstack;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Fill;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.primitives.Triple;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.shade.guava.collect.Lists;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class TestShapeOpValidation extends BaseOpValidation {



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcat(Nd4jBackend backend, TestInfo testInfo) {
        int[] concatDim = new int[]{0, 0, 0};
        List<List<int[]>> origShapes = new ArrayList<>();
        origShapes.add(Arrays.asList(new int[]{3, 4}, new int[]{5, 4}));
        origShapes.add(Arrays.asList(new int[]{1, 2, 3}, new int[]{1, 2, 3}, new int[]{2, 2, 3}));
        origShapes.add(Arrays.asList(new int[]{1, 2, 3, 4}, new int[]{2, 2, 3, 4}));

        List<String> failed = new ArrayList<>();

        for (int i = 0; i < concatDim.length; i++) {

            SameDiff sd = SameDiff.create();
            List<int[]> shapes = origShapes.get(i);

            SDVariable[] toConcat = new SDVariable[shapes.size()];
            INDArray[] orig = new INDArray[shapes.size()];
            for (int j = 0; j < shapes.size(); j++) {
                orig[j] = Nd4j.rand(DataType.DOUBLE, shapes.get(j));
                toConcat[j] = sd.var("concat-in-" + String.valueOf(j), orig[j]);
            }

            SDVariable sdConcat = sd.concat("c", 0, toConcat);
            SDVariable stdev = sd.standardDeviation("out", sdConcat, true);

            String msg = "i=" + i + ", concatDim=" + concatDim[i];
            TestCase tc = new TestCase(sd);
            tc.testName(msg)
                    .expectedOutput("c", Nd4j.concat(concatDim[i], orig));

            String error = OpValidation.validate(tc);
            if(error != null){
                failed.add(testInfo.getTestMethod().get().getName());
            }
        }

        assertEquals( 0, failed.size(),failed.toString());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeGradient(Nd4jBackend backend) {
        //https://github.com/eclipse/deeplearning4j/issues/6873

        int[] origShape = new int[]{3, 4, 5};

        List<String> failed = new ArrayList<>();

        for (long[] toShape : new long[][]{{3, 4 * 5}, {3 * 4, 5}, {1, 3 * 4 * 5}, {3 * 4 * 5, 1}}) {
            for(char order : new char[]{'c','f'}){
                INDArray inArr = Nd4j.rand(DataType.DOUBLE, origShape, order).muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable reshape = sd.reshape(in, toShape);
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", reshape, true);

                INDArray out = stdev.eval();
                INDArray expOut = in.getArr().std(true, Integer.MAX_VALUE);

                String msg = "toShape=" + Arrays.toString(toShape) + ", order=" + order;
                TestCase tc = new TestCase(sd);
                tc.testName(msg)
                        .expectedOutput("out", expOut);

                String error = OpValidation.validate(tc);
                if(error != null) {
                    failed.add(error);
                }
            }
        }

        assertEquals(0, failed.size(),failed.toString());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermuteGradient(Nd4jBackend backend) {
        int[] origShape = new int[]{3, 4, 5};

        List<String> failed = new ArrayList<>();

        for (long[] perm : new long[][]{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}) {
            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, origShape, DataType.DOUBLE)) {
                String msg = "permute=" + Arrays.toString(perm) + ", source=" + p.getSecond();
                System.out.println(msg);

                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable permute = sd.permute(in, perm);
                // Print iArgs immediately after creation
                System.out.println("  iArgs after creation: " + java.util.Arrays.toString(((Permute) sd.getOps().get(permute.getCreator().getOwnName()).getOp()).iArgs()));
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", permute, true);

                INDArray exp = inArr.permute(perm);
                INDArray expOut = in.getArr().std(true, Integer.MAX_VALUE);

                TestCase tc = new TestCase(sd);
                tc.testName(msg)
                        .expected("out", expOut)
                        .expected(permute, exp);

                // Debug: Compare expected vs actual permute output
                // Check the iArguments in the permute op
                try {
                    Permute permuteOp = (Permute) sd.getOps().get(permute.getCreator().getOwnName()).getOp();
                    System.out.println("  Permute op iArguments: " + Arrays.toString(permuteOp.iArgs()));
                } catch (Exception e) {
                    System.out.println("  ERROR getting iArgs: " + e.getMessage());
                }

                Map<String, INDArray> outputs = sd.output((Map<String, INDArray>)null, permute.name());
                INDArray actual = outputs.get(permute.name());
                System.out.println("  Expected shape: " + Arrays.toString(exp.shape()) + ", Actual shape: " + Arrays.toString(actual.shape()));
                if (!exp.equals(actual)) {
                    System.out.println("  Shapes match: " + Arrays.equals(exp.shape(), actual.shape()));
                    System.out.println("  Expected [0,0,0]: " + exp.getDouble(0, 0, 0) + ", Actual [0,0,0]: " + actual.getDouble(0, 0, 0));
                    if (exp.shape()[1] > 1 && exp.shape()[2] > 0) {
                        System.out.println("  Expected [0,1,0]: " + exp.getDouble(0, 1, 0) + ", Actual [0,1,0]: " + actual.getDouble(0, 1, 0));
                    }
                }

                String error = OpValidation.validate(tc, true);
                if(error != null){
                    System.out.println("VALIDATION FAILED: " + error);
                    failed.add(msg);
                }
            }
        }

        assertEquals(0, failed.size(),failed.toString());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRank(Nd4jBackend backend) {

        List<long[]> inShape = Arrays.asList(null, new long[]{1}, new long[]{6}, new long[]{3,4}, new long[]{3,4,5});

        for( long[] shape : inShape){

            SameDiff sd = SameDiff.create();
            SDVariable var;
            if(shape == null){
                var = sd.var("in", Nd4j.scalar(1.0));
            } else {
                var = sd.var("in", Nd4j.create(DataType.DOUBLE, shape));
            }

            SDVariable rank = sd.rank(var);

            INDArray expRank = Nd4j.scalar(DataType.INT64, shape == null ? 0 : shape.length);
            String msg = "Rank " + (shape == null ? 0 : shape.length);
            String err = OpValidation.validate(new TestCase(sd)
                    .gradientCheck(false)
                    .expected(rank, expRank));

            assertNull(err);
        }
    }

    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @ParameterizedTest()
    public void testExpandDimsOutofBounds(Nd4jBackend backend) {
        assertThrows(ND4JIllegalStateException.class,() -> {
            INDArray v1 = Nd4j.zeros(1, 1);
            INDArray v2 = Nd4j.base().expandDims(v1, 3); // crashes
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNegativeIndicesExpandDims(Nd4jBackend backend) {
        INDArray v1 = Nd4j.ones(2, 2);
        INDArray v2 = Nd4j.expandDims(v1, -1); // throws exception
        assertArrayEquals(new long[]{2,2,1},v2.shape());
    }

    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @ParameterizedTest
    public void scalarExpandDims(Nd4jBackend backend) {
        INDArray v1 = Nd4j.scalar(0); // shape is []
        INDArray v2 = Nd4j.expandDims(v1, -1); // throws exception
        System.out.println(java.util.Arrays.toString(v2.shape())); // shape should now be [1]
    }


    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @ParameterizedTest
    public void testExpandDimsSameDiff(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        SameDiff sd = SameDiff.create();
        SDVariable v1 = sd.zero(null, 1, 1);
        SDVariable v2 = sd.expandDims(v1, -1);
        assertArrayEquals(new long[]{1,1,1},v2.eval().shape());
        System.out.println(v1.shape().eval()); // [1, 1]
        System.out.println(v2.shape().eval()); // should be [1, 1, 1] but is [1, 1]
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExpandDimsGradient(Nd4jBackend backend) {
        val origShape = new long[]{3, 4};

        List<String> failed = new ArrayList<>();

        boolean first = true;
        for (int i = 0; i < 3; i++) {

            long[] expExpandShape;
            switch (i) {
                case 0:
                    expExpandShape = new long[]{1, 3, 4};
                    break;
                case 1:
                    expExpandShape = new long[]{3, 1, 4};
                    break;
                case 2:
                    expExpandShape = new long[]{3, 4, 1};
                    break;
                default:
                    throw new RuntimeException();
            }

            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAllTestMatricesWithShape(origShape[0], origShape[1], 12345, DataType.DOUBLE)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable expand = sd.expandDims(in, i);
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", expand, true);

                Map<String,INDArray> m = sd.outputAll(null);
                INDArray expOut = in.getArr().std(true);

                assertArrayEquals(expExpandShape, m.get(expand.name()).shape());
                INDArray expExpand = inArr.dup('c').reshape(expExpandShape);

                String msg = "expandDim=" + i + ", source=" + p.getSecond();
                log.info("Starting: " + msg);

                TestCase tc = new TestCase(sd);
                tc.testName(msg)
                        .expectedOutput("out", expOut)
                        .expectedOutput(expand.name(), expExpand);

                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(error);
                }
            }
        }
        assertEquals(0, failed.size(),failed.toString());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSqueezeGradient(Nd4jBackend backend,TestInfo testInfo) {
        val origShape = new long[]{3, 4, 5};

        List<String> failed = new ArrayList<>();

        for (int i = 0; i < 3; i++) {

            val shape = origShape.clone();
            shape[i] = 1;

            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, shape, DataType.DOUBLE)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable squeeze = sd.squeeze(in, i);
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", squeeze, true);

                long[] expShapePostSqueeze;
                switch (i) {
                    case 0:
                        expShapePostSqueeze = new long[]{4, 5};
                        break;
                    case 1:
                        expShapePostSqueeze = new long[]{3, 5};
                        break;
                    case 2:
                        expShapePostSqueeze = new long[]{3, 4};
                        break;
                    default:
                        throw new RuntimeException();
                }

                INDArray exp = inArr.dup('c').reshape('c', expShapePostSqueeze);

                Map<String,INDArray> m = sd.outputAll(null);

                INDArray squeezed = m.get(squeeze.name());
//                assertArrayEquals(expShapePostSqueeze, squeezed.shape());

                INDArray out = m.get(stdev.name());
                INDArray expOut = in.getArr().std(true, Integer.MAX_VALUE);
                assertEquals(expOut, out);

                String msg = "squeezeDim=" + i + ", source=" + p.getSecond();
                TestCase tc = new TestCase(sd)
                        .testName(msg)
                        .expected(squeeze.name(), exp)
                        .expectedOutput("out", expOut);


                String error = OpValidation.validate(tc, true);
                if(error != null){
                    failed.add(testInfo.getTestMethod().get().getName());
                }
            }
        }

        assertEquals( 0, failed.size(),failed.toString());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSliceGradient(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        //Order here: original shape, begin, size
        List<Triple<int[], int[], int[]>> testCases = new ArrayList<>();
        testCases.add(new Triple<>(new int[]{3, 4}, new int[]{0, 0}, new int[]{3, 4}));
        testCases.add(new Triple<>(new int[]{3, 4}, new int[]{1, 1}, new int[]{2, 2}));
        testCases.add(new Triple<>(new int[]{3, 4}, new int[]{1, 2}, new int[]{2, 2}));
        testCases.add(new Triple<>(new int[]{3, 4, 5}, new int[]{0, 0, 0}, new int[]{3, 4, 5}));
        testCases.add(new Triple<>(new int[]{3, 4, 5}, new int[]{1, 1, 1}, new int[]{2, 3, 4}));

        Map<Integer,INDArrayIndex[]> indices = new HashMap<>();
        indices.put(0, new INDArrayIndex[]{all(), all()});
        indices.put(1, new INDArrayIndex[]{interval(1,3), interval(1,3)});
        indices.put(2, new INDArrayIndex[]{interval(1,3), interval(2,4)});
        indices.put(3, new INDArrayIndex[]{all(), all(), all()});
        indices.put(4, new INDArrayIndex[]{interval(1,3), interval(1,4), interval(1,5)});

        List<String> failed = new ArrayList<>();

        for (int i = 0; i < testCases.size(); i++) {
            Triple<int[], int[], int[]> t = testCases.get(i);
            int[] os = t.getFirst();
            int[] b = t.getSecond();
            int[] e = t.getThird();
            int prod = ArrayUtil.prod(os);
            INDArray arr = Nd4j.linspace(1, prod, prod, DataType.DOUBLE).reshape(os);

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", arr);
            SDVariable slice = sd.slice(in, b, e);
            SDVariable stdev = sd.standardDeviation(slice, true);

            String msg = "i=" + i + ": inShape=" + Arrays.toString(os) + ", begin=" + Arrays.toString(b) + ", end=" + Arrays.toString(e);
            log.info("Starting test: " + msg);

            TestCase tc = new TestCase(sd).testName(msg);

            if(indices.containsKey(i)){
                tc.expected(slice, arr.get(indices.get(i)).dup());
            }

            String error = OpValidation.validate(tc, true);
            if(error != null){
                failed.add(error);
            }
        }

        assertEquals(0, failed.size(),failed.toString());
    }


    @Builder(builderClassName = "Builder")
    @Data
    private static class SSCase {
        private long[] shape;
        private long[] begin;
        private long[] end;
        private long[] strides;
        private int beginMask;
        private int endMask;
        private int ellipsisMask;
        private int newAxisMask;
        private int shrinkAxisMask;

        public static class Builder {

            public Builder shape(long... shape) {
                this.shape = shape;
                return this;
            }

            public Builder begin(long... begin) {
                this.begin = begin;
                return this;
            }

            public Builder end(long... end) {
                this.end = end;
                return this;
            }

            public Builder strides(long... strides) {
                this.strides = strides;
                return this;
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSliceGradient(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        //Order here: original shape, begin, size
        List<SSCase> testCases = new ArrayList<>();
        testCases.add(SSCase.builder().shape(3, 4).begin(0, 0).end(3, 4).strides(1, 1).build());
        testCases.add(SSCase.builder().shape(3, 4).begin(1, 1).end(2, 3).strides(1, 1).build());
        testCases.add(SSCase.builder().shape(3, 4).begin(-999, 0).end(3, 4).strides(1, 1).beginMask(1).build());
        testCases.add(SSCase.builder().shape(3, 4).begin(1, 1).end(3, -999).strides(1, 1).endMask(1 << 1).build());
        testCases.add(SSCase.builder().shape(3, 4).begin(-999, 0).end(-999, 4).strides(1, 1).beginMask(1).endMask(1).build());

        testCases.add(SSCase.builder().shape(3, 4, 5).begin(0, 0, 0).end(3, 4, 5).strides(1, 1, 1).build());
        testCases.add(SSCase.builder().shape(3, 4, 5).begin(1, 2, 3).end(3, 4, 5).strides(1, 1, 1).build());
        testCases.add(SSCase.builder().shape(3, 4, 5).begin(0, 0, 0).end(3, 3, 5).strides(1, 2, 2).build());
        testCases.add(SSCase.builder().shape(3, 4, 5).begin(1, -999, 1).end(3, 3, 4).strides(1, 1, 1).beginMask(1 << 1).build());
        testCases.add(SSCase.builder().shape(3, 4, 5).begin(1, -999, 1).end(3, 3, -999).strides(1, 1, 1).beginMask(1 << 1).endMask(1 << 2).build());
        testCases.add(SSCase.builder().shape(3, 4, 5).begin(1, 2).end(3, 4).strides(1, 1).ellipsisMask(1 << 1).build());   //[1:3,...,2:4]
        testCases.add(SSCase.builder().shape(3, 4, 5).begin(1, -999, 1, 2).end(3, -999, 3, 4).strides(1, -999, 1, 2).newAxisMask(1 << 1).build());
        testCases.add(SSCase.builder().shape(3, 4, 5).begin(1, 0, 1).end(3, -999, 4).strides(1, 1, 1).shrinkAxisMask(1 << 1).build());
        testCases.add(SSCase.builder().shape(3, 4, 5).begin(1, 1, 1).end(3, -999, 4).strides(1, 1, 1).shrinkAxisMask(1 << 1).build());

        Map<Integer,INDArrayIndex[]> indices = new HashMap<>();
        indices.put(0, new INDArrayIndex[]{all(), all()});
        indices.put(1, new INDArrayIndex[]{interval(1,2), interval(1,3)});
        indices.put(2, new INDArrayIndex[]{interval(0,3), interval(0,4)});
        indices.put(3, new INDArrayIndex[]{interval(1,3), interval(1,4)});

        indices.put(5, new INDArrayIndex[]{all(), all(), all()});
        indices.put(7, new INDArrayIndex[]{interval(0,1,3), interval(0,2,3), interval(0,2,5)});


        List<String> failed = new ArrayList<>();

        for (int i = 0; i < testCases.size(); i++) {
            SSCase t = testCases.get(i);
            INDArray arr = Nd4j.rand(t.getShape());

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", arr);
            SDVariable slice = sd.stridedSlice(in, t.getBegin(), t.getEnd(), t.getStrides(), t.getBeginMask(),
                    t.getEndMask(), t.getEllipsisMask(), t.getNewAxisMask(), t.getShrinkAxisMask());
            SDVariable stdev = sd.standardDeviation(slice, true);

            String msg = "i=" + i + ": " + t;
            log.info("Starting test: " + msg);

            TestCase tc = new TestCase(sd);
            tc.testName(msg);

            if(indices.containsKey(i)){
                tc.expected(slice, arr.get(indices.get(i)).dup());
            }

            String error = OpValidation.validate(tc, true);
            if(error != null){
                failed.add(error);
            }
        }
        assertEquals( 0, failed.size(),failed.toString());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMerge(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int t = 0; t < 3; t++) {
            for (int numArrays : new int[]{3, 1}) {
                for (long[] shape : new long[][]{{1}, {3, 4}, {3, 4, 5}}) {


                    SameDiff sd = SameDiff.create();
                    SDVariable[] arr = new SDVariable[numArrays];

                    for (int i = 0; i < numArrays; i++) {
                        INDArray randArr = Nd4j.rand(shape);
                        // Force sync by reading a value - ensures device data is copied to host
                        randArr.getDouble(0);
                        arr[i] = sd.var(String.valueOf(i), randArr);
                    }

                    // Compute expected by explicitly extracting and summing input arrays
                    INDArray[] inputArrays = new INDArray[numArrays];
                    for (int i = 0; i < numArrays; i++) {
                        inputArrays[i] = arr[i].getArr();
                    }

                    INDArray exp;
                    SDVariable merge;
                    String name;
                    switch (t) {
                        case 0:
                            name = "mergeAdd";
                            merge = sd.math().mergeAdd(arr);
                            // Compute expected: sum all arrays
                            exp = inputArrays[0].dup();
                            for (int i = 1; i < numArrays; i++) {
                                exp = exp.add(inputArrays[i]);
                            }
                            break;
                        case 1:
                            name = "mergeMax";
                            merge = sd.math().mergeMax(arr);
                            exp = inputArrays[0].dup();
                            for( int i=1; i<numArrays; i++ ){
                                exp = Transforms.max(exp, inputArrays[i], true);
                            }
                            break;
                        case 2:
                            name = "mergeAvg";
                            merge = sd.math().mergeAvg(arr);
                            // Compute expected: average all arrays
                            exp = inputArrays[0].dup();
                            for (int i = 1; i < numArrays; i++) {
                                exp = exp.add(inputArrays[i]);
                            }
                            exp = exp.div(numArrays);
                            break;
                        default:
                            throw new RuntimeException();
                    }

                    String msg = name + " - numArrays=" + numArrays + ", shape=" + Arrays.toString(shape);
                    SDVariable loss;
                    if(shape.length > 1){
                        loss = sd.standardDeviation("loss", merge, true);
                    } else {
                        loss = sd.mean("loss", merge);
                    }


                    TestCase tc = new TestCase(sd)
                            .expected(merge, exp)
                            .testName(msg);
                    String error = OpValidation.validate(tc, true);
                    if(error != null){
                        failed.add(msg + " - " + error);
                    }
                }
            }
        }

        assertEquals(0, failed.size(),failed.toString());
    }

    @Override
    public long getTimeoutMilliseconds() {
        return Long.MAX_VALUE;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStack(Nd4jBackend backend,TestInfo testInfo) {
        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        List<long[]> origShape = Arrays.asList(
                new long[]{1},
                new long[]{1, 1},
                new long[]{3, 4},
                new long[]{3, 4, 5},
                new long[]{3, 4, 5, 6}
        );

        for (long[] shape : origShape) {
            for (int axis = 0; axis <= shape.length; axis++) {
                for (int numInputs : new int[]{1, 3}) {

                    long[] expOutShape = new long[shape.length + 1];
                    int x = 0;
                    for (int i = 0; i <= shape.length; i++) {
                        if (i == axis) {
                            expOutShape[i] = numInputs;
                        } else {
                            expOutShape[i] = shape[x++];
                        }
                    }


                    SameDiff sd = SameDiff.create();

                    SDVariable[] in = new SDVariable[numInputs];
                    INDArray[] inArr = new INDArray[numInputs];
                    for (int i = 0; i < numInputs; i++) {
                        inArr[i] = Nd4j.rand(shape);
                        in[i] = sd.var(String.valueOf(i), inArr[i]);
                    }

                    INDArray expStack = null;
                    if(Arrays.equals(new long[]{3,4}, shape)) {
                        if(axis == 0){
                            INDArray out = Nd4j.create(numInputs, 3, 4);
                            for( int i = 0; i < numInputs; i++) {
                                out.get(point(i), all(), all()).assign(inArr[i]);
                            }
                            expStack = out;
                        } else if(axis == 1) {
                            INDArray out = Nd4j.create(3, numInputs, 4);
                            for( int i = 0; i<numInputs; i++) {
                                out.get(all(), point(i), all()).assign(inArr[i]);
                            }
                            expStack = out;
                        } else {
                            INDArray out = Nd4j.create(3, 4, numInputs);
                            for( int i = 0; i < numInputs; i++) {
                                out.get(all(), all(), point(i)).assign(inArr[i]);
                            }
                            expStack = out;
                        }
                    }

                    SDVariable stack = sd.stack(axis, in);

                    INDArray out = stack.eval();
                    assertArrayEquals(expOutShape, out.shape());

                    if (ArrayUtil.prodLong(shape) == 1) {
                        SDVariable loss = sd.sum("loss", stack);
                    } else {
                        SDVariable loss = sd.standardDeviation("loss", stack, true);
                    }

                    String msg = Arrays.toString(shape) + ", axis=" + axis + ", numInputs=" + numInputs;

                    TestCase tc = new TestCase(sd);
                    if(expStack != null){
                        tc.expected(stack, expStack);
                    }

                    String error = OpValidation.validate(tc);
                    if(error != null){
                        failed.add(testInfo.getTestMethod().get().getName());
                    }
                }
            }
        }

        assertEquals(0, failed.size(),failed.toString());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testUnStack(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        List<long[]> unstackedShape = Arrays.asList(
                new long[]{1},
                new long[]{1, 1},
                new long[]{3, 4},
                new long[]{3, 4, 5},
                new long[]{3, 4, 5, 6}
        );

        for (long[] shape : unstackedShape) {
            for (int axis = 0; axis <= shape.length; axis++) {
//                for (int numInputs : new int[]{1, 3}) {
                for (int numInputs : new int[]{3}) {

                    long[] stackedShape = new long[shape.length + 1];
                    int x = 0;
                    for (int i = 0; i <= shape.length; i++) {
                        if (i == axis) {
                            stackedShape[i] = numInputs;
                        } else {
                            stackedShape[i] = shape[x++];
                        }
                    }


                    SameDiff sd = SameDiff.create();
                    INDArray in = Nd4j.rand(stackedShape);
                    SDVariable var = sd.var("var", in);

                    SDVariable[] unstacked = sd.unstack(var, axis, numInputs);

                    INDArray[] unstackedExp = null;
                    if(Arrays.equals(new long[]{3,4}, shape)){
                        unstackedExp = new INDArray[numInputs];
                        if(axis == 0){
                            for(int i=0; i<numInputs; i++ ){
                                unstackedExp[i] = in.get(point(i), all(), all());
                            }
                        } else if(axis == 1){
                            for(int i=0; i<numInputs; i++ ){
                                unstackedExp[i] = in.get(all(), point(i), all());
                            }
                        } else {
                            for(int i=0; i<numInputs; i++ ){
                                unstackedExp[i] = in.get(all(), all(), point(i));
                            }
                        }
                    }

                    //for gradient check, need to combine to single scalar output...
                    SDVariable merged = sd.math().mergeAvg(unstacked);

                    if (ArrayUtil.prodLong(stackedShape) == 1 || ArrayUtil.prodLong(shape) == 1) {
                        SDVariable loss = sd.sum("loss", merged);
                    } else {
                        SDVariable loss = sd.standardDeviation("loss", merged, true);
                    }

                    String msg = "Unstacked shape = " + Arrays.toString(shape) + ", stacked shape = " + Arrays.toString(stackedShape)
                            + ", axis=" + axis + ", numInputs=" + numInputs;

                    Map<String,INDArray> m = sd.outputAll(null);
                    for (SDVariable v : unstacked) {
                        assertArrayEquals(shape, m.get(v.name()).shape(),msg);
                    }

                    TestCase tc = new TestCase(sd).testName(msg);
                    if (unstackedExp != null) {
                        for( int i=0; i<numInputs; i++ ){
                            tc.expected(unstacked[i], unstackedExp[i]);
                        }
                    }
                    String error = OpValidation.validate(tc, true);
                    if(error != null){
                        failed.add(error);
                    }
                }
            }
        }

        assertEquals( 0, failed.size(),failed.toString());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTile(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        List<int[]> tileArg = Arrays.asList(
                new int[]{1},
                new int[]{5},
                new int[]{3,4},
                new int[]{2,3},
                new int[]{2,3,4}
        );

        INDArray[] orig = new INDArray[tileArg.size()];
        orig[0] = Nd4j.valueArrayOf(new long[]{1}, 3.0);
        orig[1] = Nd4j.valueArrayOf(new long[]{1}, 3.0);
        orig[2] = Nd4j.valueArrayOf(new long[]{1,1}, 3.0);
        orig[3] = Nd4j.rand(2,2).muli(10);
        orig[4] = Nd4j.rand(new int[]{3,4,5}).muli(10);

        INDArray[] exp = new INDArray[tileArg.size()];
        exp[0] = Nd4j.create(new double[]{3});
        exp[1] = Nd4j.create(new double[]{3,3,3,3,3});
        exp[2] = Nd4j.valueArrayOf(new long[]{3,4}, 3.0);
        exp[3] = Nd4j.create(2*2, 2*3);
        for( int i=0; i<2; i++ ){
            for( int j=0; j<3; j++ ){
                exp[3].get(interval(2*i,2*(i+1)), interval(2*j,2*(j+1))).assign(orig[3]);
            }
        }
        exp[4] = Nd4j.create(3*2, 4*3, 5*4);
        for( int i=0; i<2; i++ ){
            for( int j=0; j<3; j++ ){
                for( int k=0; k<4; k++ ) {
                    exp[4].get(interval(3 * i, 3 * (i + 1)), interval(4 * j, 4 * (j + 1)), interval(5*k, 5*(k+1))).assign(orig[4]);
                }
            }
        }

        List<String> failed = new ArrayList<>();

        for (int i = 0; i < tileArg.size(); i++) {
            int[] tArg = tileArg.get(i);
            INDArray inArr = orig[i];
            log.info("Starting test {} - shape {}, tile arg {}", i, Arrays.toString(inArr.shape()), Arrays.toString(tArg));

            SameDiff sd = SameDiff.create();
            SDVariable var = sd.var("in", inArr);
            SDVariable tile = sd.tile(var, tArg);

            if(exp[i].length() == 1 || inArr.length() == 1){
                SDVariable loss = sd.sum("loss", tile);
            } else {
                SDVariable loss = sd.standardDeviation("loss", tile, true);
            }

            String msg = "Shape=" + Arrays.toString(inArr.shape()) + " - tile=" + Arrays.toString(tArg);

            TestCase tc = new TestCase(sd)
                    .expected(tile, exp[i])
                    //Tile op seems unusually sensitive - but testTileBp and testTileBp2 seem to verify it's correctness...
                    .gradCheckMinAbsError(5e-3)
                    .gradCheckMaxRelativeError(5e-3);
            String error = OpValidation.validate(tc);
            if(error != null){
                failed.add(msg + " - " + error);
            }
        }

        assertEquals( 0, failed.size(),failed.toString());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTileBp(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray in = Nd4j.create(1,2,3);   //Values aren't used in backprop, just shape
        int[] tile = new int[]{2,3,4};

        int[] outShape = new int[]{1*2, 2*3, 3*4};
        int length = ArrayUtil.prod(outShape);
        INDArray gradAtOut = Nd4j.rand(outShape);

        INDArray gradAtInExp = Nd4j.create(in.shape());
        for(int i=0; i<tile[0]; i++ ){
            for( int j=0; j<tile[1]; j++){
                for( int k=0; k<tile[2]; k++ ){
                    INDArray subset = gradAtOut.get(NDArrayIndex.interval(i*1, (i+1)*1), NDArrayIndex.interval(j*2, (j+1)*2), NDArrayIndex.interval(k*3, (k+1)*3));
                    gradAtInExp.addi(subset);
                }
            }
        }

        DynamicCustomOp op = DynamicCustomOp.builder("tile_bp")
                .addInputs(in, gradAtOut)
                .addOutputs(gradAtInExp)
                .addIntegerArguments(tile)
                .build();
        OpTestCase otc = new OpTestCase(op)
                .expectedOutput(0, gradAtInExp);

        String err = OpValidation.validate(otc);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTileBp2(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray in = Nd4j.create(3,4,5);   //Values aren't used in backprop, just shape
        int[] tile = new int[]{2,3,4};

        int[] outShape = new int[]{3*2, 4*3, 5*4};
        int length = ArrayUtil.prod(outShape);
        INDArray gradAtOut = Nd4j.rand(outShape);

        INDArray gradAtInExp = Nd4j.create(in.shape());
        for(int i=0; i<tile[0]; i++ ){
            for( int j=0; j<tile[1]; j++){
                for( int k=0; k<tile[2]; k++ ){
                    INDArray subset = gradAtOut.get(NDArrayIndex.interval(i*3, (i+1)*3), NDArrayIndex.interval(j*4, (j+1)*4), NDArrayIndex.interval(k*5, (k+1)*5));
                    gradAtInExp.addi(subset);
                }
            }
        }

        DynamicCustomOp op = DynamicCustomOp.builder("tile_bp")
                .addInputs(in, gradAtOut)
                .addOutputs(gradAtInExp)
                .addIntegerArguments(tile)
                .build();
        OpTestCase otc = new OpTestCase(op)
                .expectedOutput(0, gradAtInExp);

        String err = OpValidation.validate(otc);
        assertNull(err);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshape(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(-5, 6, 12)).reshape(3, 4).castTo(DataType.DOUBLE);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result1 = sameDiff.reshape(x, 4, 3);
        SDVariable loss = sameDiff.standardDeviation(result1, true);

        INDArray exp = arr.dup('c').reshape('c', 4,3);

        String err = OpValidation.validate(new TestCase(sameDiff)
                .expectedOutput(result1.name(), exp));

        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshape2(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        int[] origShape = new int[]{3, 4, 5};

        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape(origShape);

        for (int[] toShape : new int[][]{{3, 4 * 5}, {3 * 4, 5}, {1, 3 * 4 * 5}, {3 * 4 * 5, 1}}) {
            INDArray exp = inArr.reshape(toShape);

            INDArray out = Nd4j.create(toShape);
            Nd4j.getExecutioner().exec(DynamicCustomOp.builder("reshape")
                    .addInputs(inArr)
                    .addOutputs(out)
                    .addIntegerArguments(-'c')
                    .addIntegerArguments(toShape)
                    .build());

            assertEquals(exp, out);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTranspose(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(1,4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.transpose(x);
        SDVariable loss = sameDiff.standardDeviation(result, true);

        String err = OpValidation.validate(new TestCase(sameDiff).expectedOutput(result.name(), arr.transpose()));
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTransposeOp(Nd4jBackend backend) {

        INDArray arr = Nd4j.linspace(1,15, 15).reshape(5,3);
        INDArray out = Nd4j.create(Nd4j.defaultFloatingPointType(), new long[]{3,5}, 'c');

        OpTestCase op = new OpTestCase(new Transpose(arr, out));
        INDArray exp = arr.transpose();
        op.expectedOutput(0, exp.dup('f'));
        String err = OpValidation.validate(op);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShape(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        val shape = new long[]{2, 3};
        SDVariable x = sameDiff.var("x", shape);
        SDVariable result = sameDiff.shape(x).castTo(DataType.DOUBLE);
        SDVariable loss = sameDiff.standardDeviation(result, true);

        String err = OpValidation.validate(new TestCase(sameDiff)
                .gradientCheck(false)
                .expected(result, Nd4j.create(new double[]{2,3}, new long[]{2})));

        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSize(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        val shape = new long[]{2, 3};
        SDVariable x = sameDiff.var("x", DataType.FLOAT, shape);
        SDVariable result = sameDiff.size(x);

        String err = OpValidation.validate(new TestCase(sameDiff)
                .gradientCheck(false)
                .expected(result, Nd4j.scalar(6L)));

        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDiagShapeFn(Nd4jBackend backend) {
        INDArray i = Nd4j.linspace(1, 16, 16).reshape(4,4);

        OpTestCase op = new OpTestCase(new DiagPart(i, null));

        INDArray exp = Nd4j.create(new double[]{1,6,11,16}, new long[]{4});
        op.expectedOutput(0, exp);

        String err = OpValidation.validate(op);
        assertNull(err);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermute(Nd4jBackend backend) {
        INDArray in = Nd4j.linspace(1, 60, 60).reshape(3,4,5);
        INDArray exp = in.permute(0,1,2);   //No op

        assertEquals(in, exp);

        INDArray out = Nd4j.create(3,4,5);
        OpTestCase op = new OpTestCase(new Permute(in,out,0,1,2));
        op.expectedOutput(0, exp);

        assertNull(OpValidation.validate(op));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermute2(Nd4jBackend backend) {
        for (long[] perm : new long[][]{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}) {
            INDArray in = Nd4j.linspace(1, 60, 60).reshape(3,4,5);
            INDArray exp = in.permute(perm).dup('c');

            int[] outShape = new int[3];
            for( int i = 0; i < 3; i++) {
                outShape[i] = (int)in.size(perm[i]);
            }

            INDArray out = Nd4j.create(outShape);
            OpTestCase op = new OpTestCase(new Permute(in, out, perm));
            op.expectedOutput(0, exp);

            assertNull(OpValidation.validate(op));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConstant(Nd4jBackend backend) {

        //Case 0: no shape
        SameDiff sd = SameDiff.create();
        INDArray ia = Nd4j.create(new double[]{1,2,3});
        SDVariable in = sd.var(ia);
        SDVariable loss = in.std(true);

        assertNull(OpValidation.validate(new TestCase(sd).expected(in, ia)));

        //Case 1: shape is provided + scalar

        sd = SameDiff.create();
        ia = Nd4j.scalar(3.0);
        in = sd.var(ia);
        SDVariable constant = sd.constant(Nd4j.create(DataType.FLOAT, 3,4,5));
        INDArray exp = Nd4j.valueArrayOf(new long[]{3,4,5}, 3.0);
        loss = constant.std(true);

        assertNull(OpValidation.validate(new TestCase(sd)
                .gradientCheck(false)
                .expected(constant, Nd4j.create(DataType.FLOAT, 3,4,5))));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testUnstackEdgeCase2(Nd4jBackend backend) {
        for( int i=0; i<3; i++ ) {

            INDArray arr = Nd4j.rand(new long[]{1, 1, 1});

            val shapes = Nd4j.getExecutioner().calculateOutputShape(
                    new Unstack(arr, null, i));

            assertEquals(1, shapes.size());
            assertArrayEquals(new long[]{1, 1}, Shape.shape(shapes.get(0).asLong()));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void invertPermutation(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new float[] {3, 4, 0, 2, 1}).castTo(DataType.INT);
        INDArray expOut = Nd4j.create(new float[] {2, 4, 3, 0, 1}).castTo(DataType.INT);

        SDVariable input = sd.var("in", DataType.INT, 1, 5);
        sd.associateArrayWithVariable(ia, input);
        SDVariable out = sd.invertPermutation(input);

        assertNull(OpValidation.validate(new TestCase(sd)
                .gradientCheck(false)   //Integer indices in/out
                .expected(out, expOut)));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGatherNd(Nd4jBackend backend) {

        List<INDArray> indices = new ArrayList<>();
        List<INDArray> params = new ArrayList<>();
        List<INDArray> expected = new ArrayList<>();


        indices.add(Nd4j.create(new double[][]{{0,0},{1,1}}).castTo(DataType.INT));
        params.add(Nd4j.create(new double[][]{{1,2},{3,4}}));
        expected.add(Nd4j.create(new double[]{1,4}));

        indices.add(Nd4j.create(new double[][]{{1},{0}}).castTo(DataType.INT));
        params.add(Nd4j.create(new double[][]{{1,2},{3,4}}));
        expected.add(Nd4j.create(new double[][]{{3,4},{1,2}}));

        indices.add(Nd4j.create(new double[][]{{0,1},{1,0}}).castTo(DataType.INT));
        params.add(Nd4j.create(new double[][][]{{{10,20},{30,40}},
                {{11, 21}, {31,41}}}));
        expected.add(Nd4j.create(new double[][]{{30,40},{11,21}}));

        for( int i = 0; i < indices.size(); i++) {
            SameDiff sd = SameDiff.create();
            SDVariable p = sd.var("p", params.get(i));
            SDVariable ind = sd.constant("i", indices.get(i));
            SDVariable g = sd.gatherNd(p, ind);

            INDArray exp = expected.get(i);

            String err = OpValidation.validate(new TestCase(sd)
                    .expected(g, exp)
                    .gradientCheck(false)); //Grad not implemented
            assertNull(err);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReverseSequence(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        float[] input_data = new float[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                0, 0, 0,
                0, 0, 0,

                1, 2, 3,
                4, 5, 6,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
        };
        float[] expected_output = new float[]{
                7, 8, 9,
                4, 5, 6,
                1, 2, 3,
                0, 0, 0,
                0, 0, 0,

                4, 5, 6,
                1, 2, 3,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
        };
        INDArray arr1 = Nd4j.create(input_data, new long[]{2, 5, 3}).castTo(DataType.DOUBLE);
        INDArray seqLenArr = Nd4j.createFromArray(3, 2);
        SDVariable x = sameDiff.constant("x", arr1);
        SDVariable seq_lengths = sameDiff.constant("seq_lengths", seqLenArr);
        SDVariable result = sameDiff.reverseSequence(x, seq_lengths, 1, 0);
        INDArray expected = Nd4j.create(expected_output, new long[]{2, 5, 3}).castTo(DataType.DOUBLE);
        assertArrayEquals(arr1.shape(), result.eval().shape());
        assertEquals(expected, result.eval());

        SDVariable loss = sameDiff.standardDeviation(result, true);
        String err = OpValidation.validate(new TestCase(sameDiff)
                .expected(result.name(), expected)
                .gradientCheck(false));
        assertNull(err);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("MatrixDeterminant does not have a gradient yet.")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testMatrixDeterminant(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(3,3);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", in);
        SDVariable md = sd.math().matrixDeterminant(var);

        double d = new LUDecomposition(CheckUtil.convertToApacheMatrix(in)).getDeterminant();


        INDArray outExp = Nd4j.scalar(d);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(md.name(), outExp));
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("MatrixDeterminant does not have a gradient yet.")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testDeterminant22(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.create(new double[][]{{1, 2.5}, {3.5, 4.5}});


        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", in);
        SDVariable md = sd.math().matrixDeterminant(var);

        double d = new LUDecomposition(CheckUtil.convertToApacheMatrix(in)).getDeterminant();
        double d2 = in.getDouble(0,0) * in.getDouble(1,1) - in.getDouble(1,0) * in.getDouble(0,1);
        assertEquals(d, d2, 1e-5);


        INDArray outExp = Nd4j.scalar(d);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(md.name(), outExp));
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("MatrixDeterminant does not have a gradient yet.")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testMatrixDeterminant3(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(3,3);
        //System.out.println(in.shapeInfoToString());   //Rank: 2,Offset: 0 Order: c Shape: [3,3],  stride: [3,1]
        //System.out.println(Arrays.toString(in.data().asFloat())); //[0.27620894, 0.21801452, 0.062078513, 7.348895E-4, 0.24149609, 0.4948205, 0.93483436, 0.52035654, 0.30292067]

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", in);
        SDVariable md = sd.math().matrixDeterminant(var);

        double d = new LUDecomposition(CheckUtil.convertToApacheMatrix(in)).getDeterminant();

        //https://en.wikipedia.org/wiki/Determinant
        double[][] a = in.toDoubleMatrix();
        double d2 = a[0][0] * a[1][1] * a[2][2]
                + a[0][1] * a[1][2] * a[2][0]
                + a[0][2] * a[1][0] * a[2][1]
                - a[0][2] * a[1][1] * a[2][0]
                - a[0][1] * a[1][0] * a[2][2]
                - a[0][0] * a[1][2] * a[2][1];
        assertEquals(d, d2, 1e-6);          //Manual calc and Apache commons both match:    0.03589524995561552

        INDArray outExp = Nd4j.scalar(d);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(md.name(), outExp));
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("MatrixDeterminant does not have a gradient yet.")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testMatrixDeterminant4(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(4,4);
        //System.out.println(in.shapeInfoToString());   //Rank: 2,Offset: 0 Order: c Shape: [4,4],  stride: [4,1]
        //System.out.println(Arrays.toString(in.data().asFloat())); //[0.27620894, 0.21801452, 0.062078513, 7.348895E-4, 0.24149609, 0.4948205, 0.93483436, 0.52035654, 0.30292067, 0.3289706, 0.7977864, 0.03180518, 0.1455722, 0.90352905, 0.9405744, 0.0048329555]

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", in);
        SDVariable md = sd.math().matrixDeterminant(var);

        double d = new LUDecomposition(CheckUtil.convertToApacheMatrix(in)).getDeterminant();   //-0.06713878100086641
        //System.out.println(d);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(md.name(), Nd4j.scalar(d)));
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSegmentOps(Nd4jBackend backend) {
        //https://github.com/eclipse/deeplearning4j/issues/6952
        INDArray s = Nd4j.create(new double[]{0,0,0,1,2,2,3,3}, new long[]{8}).castTo(DataType.INT);
        INDArray d = Nd4j.create(new double[]{5,1,7,2,3,4,1,3}, new long[]{8});
        int numSegments = 4;

        List<String> failed = new ArrayList<>();

        for(String op : new String[]{"max", "min", "mean", "prod", "sum",
                "umax", "umin", "umean", "uprod", "usum", "usqrtn"}) {
            log.info("Starting test: {}", op);

            if(op.startsWith("u")){
                //Unsorted segment cases
                s = Nd4j.create(new double[]{3,1,0,0,2,0,3,2}, new long[]{8}).castTo(DataType.INT);
                d = Nd4j.create(new double[]{1,2,5,7,3,1,3,4}, new long[]{8});
            }

            SameDiff sd = SameDiff.create();
            SDVariable data = sd.var("data", d);
            SDVariable segments = sd.constant("segments", s);

            SDVariable sm;
            INDArray exp;
            switch (op){
                case "max":
                    sm = sd.segmentMax(data, segments);
                    exp = Nd4j.create(new double[]{7, 2, 4, 3});
                    break;
                case "min":
                    sm = sd.segmentMin(data, segments);
                    exp = Nd4j.create(new double[]{1, 2, 3, 1});
                    break;
                case "mean":
                    sm = sd.segmentMean(data, segments);
                    exp = Nd4j.create(new double[]{4.3333333333, 2, 3.5, 2});
                    break;
                case "prod":
                    sm = sd.segmentProd(data, segments);
                    exp = Nd4j.create(new double[]{35, 2, 12, 3});
                    break;
                case "sum":
                    sm = sd.segmentSum(data, segments);
                    exp = Nd4j.create(new double[]{13, 2, 7, 4});
                    break;
                case "umax":
                    sm = sd.unsortedSegmentMax(data, segments, numSegments);
                    exp = Nd4j.create(new double[]{7, 2, 4, 3});
                    break;
                case "umin":
                    sm = sd.unsortedSegmentMin(data, segments, numSegments);
                    exp = Nd4j.create(new double[]{1, 2, 3, 1});
                    break;
                case "umean":
                    sm = sd.unsortedSegmentMean(data, segments, numSegments);
                    exp = Nd4j.create(new double[]{4.3333333333, 2, 3.5, 2});
                    break;
                case "uprod":
                    sm = sd.unsortedSegmentProd(data, segments, numSegments);
                    exp = Nd4j.create(new double[]{35, 2, 12, 3});
                    break;
                case "usum":
                    sm = sd.unsortedSegmentSum(data, segments, numSegments);
                    exp = Nd4j.create(new double[]{13, 2, 7, 4});
                    break;
                case "usqrtn":
                    sm = sd.unsortedSegmentSqrtN(data, segments, numSegments);
                    exp = Nd4j.create(new double[]{(5 + 7 + 1)/Math.sqrt(3), 2, (3 + 4)/ Math.sqrt(2), (1 + 3) / Math.sqrt(2)});
                    break;
                default:
                    throw new RuntimeException();
            }

            SDVariable loss = sm.std(true);
            sd.addLossVariable(loss);

            TestCase tc = new TestCase(sd)
                    .testName(op)
                    .expected(sm, exp)
                    .gradientCheck(true)
                    .gradCheckSkipVariables(segments.name());

            String err = OpValidation.validate(tc);
            if(err != null)
                failed.add(err);
        }

        assertEquals(0, failed.size(),failed.toString());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSegmentMean(Nd4jBackend backend) {
        INDArray x = Nd4j.linspace(DataType.FLOAT, 1, 18, 1).reshape(6, 3);
        INDArray segmentIds = Nd4j.createFromArray(0, 0, 1, 1, 2, 2);

        INDArray out = Nd4j.create(DataType.FLOAT, 3, 3);

        Nd4j.exec(DynamicCustomOp.builder("segment_mean")
                .addInputs(x, segmentIds)
                .addOutputs(out)
                .build());

        INDArray exp = out.like();
        exp.putRow(0, x.getRow(0).add(x.getRow(1)).muli(0.5));
        exp.putRow(1, x.getRow(2).add(x.getRow(3)).muli(0.5));
        exp.putRow(2, x.getRow(4).add(x.getRow(5)).muli(0.5));

        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSequenceMask(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.createFromArray(new int[] {1, 3, 2});
        // arr is not trainable, so it's constant in model
        SDVariable lengths = sameDiff.constant(arr);

        // Test with static max len
        int maxlen = 5;
        INDArray expected = Nd4j.create(new float[] {
                1.f,     0.f,     0.f,    0.f,   0.f,
                1.f,     1.f,     1.f,    0.f,   0.f,
                1.f,     1.f,     0.f,    0.f,   0.f
        }).reshape(3,5);
        INDArray[] ret = Nd4j.exec(new SequenceMask(arr, maxlen, DataType.FLOAT));
        SDVariable result1 = sameDiff.sequenceMask(lengths, maxlen, DataType.FLOAT);
        assertArrayEquals(expected.shape(), result1.eval().shape());
        assertEquals(expected, result1.eval());

        SDVariable loss = sameDiff.standardDeviation(result1, true);

        String err = OpValidation.validate(new TestCase(sameDiff)
                .expected(result1, expected)
                .gradientCheck(false));
        assertNull(err);

        // Test with dynamic maxlen
        lengths = sameDiff.constant("lengths2", arr);
        SDVariable maxLen = sameDiff.constant("maxLen", Nd4j.scalar(5));
        SDVariable result2 = sameDiff.sequenceMask(lengths, maxLen, DataType.FLOAT);
//        assertArrayEquals(expected.shape(), result2.eval().shape());
        assertEquals(expected, result2.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeshGrid(Nd4jBackend backend) {
        List<String> failed = new ArrayList<>();

        for( int rank=2; rank<=4; rank++ ){
            SameDiff sd = SameDiff.create();

            SDVariable[] arr = new SDVariable[rank];
            String[] names = new String[rank];
            for( int i=0; i<rank; i++ ){
                INDArray in = Nd4j.linspace(1,3+i, 3+i).reshape(3+i).castTo(DataType.DOUBLE);
                arr[i] = sd.var("in"+i, in);
                names[i] = "meshgrid-" + i;
            }
            SDVariable[] meshgrid = sd.math().meshgrid(names, arr, false);

            TestCase tc = new TestCase(sd);

            long[] shape;
            if(rank == 2){
                shape = new long[]{3,4};
            } else if(rank == 3) {
                shape = new long[]{3,4,5};
            } else {
                shape = new long[]{3,4,5,6};
            }
            INDArray[] exp = new INDArray[shape.length];    //Nd4j.create(shape);
            for( int i=0; i<exp.length; i++ ){
                exp[i] = Nd4j.create(DataType.DOUBLE, shape);
                long nTensors = exp[i].tensorsAlongDimension(i);
                for( long j=0; j<nTensors; j++ ){
                    INDArray tad = exp[i].tensorAlongDimension((int)j, i);
                    tad.assign(arr[i].getArr());
                }

                tc.expected(meshgrid[i], exp[i]);
            }

            SDVariable loss = null;
            for( int i=0; i<rank; i++ ){
                if(i == 0)
                    loss = meshgrid[i].std(true);
                else {
                    loss = loss.add("loss-" + i, meshgrid[i].std(true));
                }
            }

            String err = OpValidation.validate(tc, true);
            if(err != null)
                failed.add(err);
        }

        assertEquals( 0, failed.size(),failed.toString());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("Adam: 5/11/22: invalid with latest indexing changes")
    public void testGather(Nd4jBackend backend) {
        List<INDArray> inArrs = new ArrayList<>();
        List<Integer> axis = new ArrayList<>();
        List<INDArray> indices = new ArrayList<>();

        inArrs.add(Nd4j.linspace(1,48,48).reshape(2,4,3,2));
        indices.add(Nd4j.create(new double[]{1,0}).castTo(DataType.INT));
        axis.add(-1);

        for(int i = 0; i < inArrs.size(); i++) {

            INDArray in = inArrs.get(i);
            INDArray idx = indices.get(i);
            int a = axis.get(i);
            int aNorm = (a >= 0 ? a : a + in.rank());

            INDArray expOut;
            if(idx.rank() == 0) {
                INDArrayIndex[] get = new INDArrayIndex[in.rank()];
                for( int j = 0; j < aNorm; j++) {
                    get[j] = NDArrayIndex.all();
                }
                get[aNorm] = NDArrayIndex.point(idx.getInt(0));
                for( int j = aNorm + 1; j <  in.rank(); j++) {
                    get[j] = NDArrayIndex.all();
                }
                expOut = in.get(get);
            } else if (idx.rank() == 1) {
                long[] shape = in.shape().clone();
                shape[aNorm] = idx.length();
                expOut = Nd4j.create(shape);

                INDArrayIndex[] get = new INDArrayIndex[in.rank()];
                INDArrayIndex[] put = new INDArrayIndex[in.rank()];
                for( int j = 0; j < aNorm; j++) {
                    get[j] = NDArrayIndex.all();
                    put[j] = NDArrayIndex.all();
                }
                for( int j = aNorm + 1; j < in.rank(); j++) {
                    get[j] = NDArrayIndex.all();
                    put[j] = NDArrayIndex.all();
                }

                for(int j = 0; j < idx.length(); j++) {
                    get[aNorm] = NDArrayIndex.point(idx.getInt(j));
                    put[aNorm] = NDArrayIndex.point(j);
                    expOut.put(put, in.get(get));
                }
            } else {
                throw new RuntimeException("Rank 2+ tests not yet implemented");
            }


            SameDiff sd = SameDiff.create();
            SDVariable sdIn = sd.var("in", in);
            SDVariable sdIdx = sd.constant("idx", idx);
            SDVariable gather = sd.gather(sdIn, sdIdx, a);

            SDVariable loss = gather.std(true);

            String err = OpValidation.validate(new TestCase(sd)
                    .expected(gather, expOut)
                    .gradCheckSkipVariables("idx"));

            assertNull(err);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGatherSimple(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.create(new float[]{1, 2, 3, 4}, new long[]{2, 2});
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.gather(x, new int[]{1, 0}, 1);
        INDArray expected = Nd4j.create(new float[]{2, 1, 4, 3}, new long[]{2, 2});
        assertEquals(expected, result.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGatherNdSingle(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(DataType.DOUBLE, 1, 24, 24)).reshape(2, 3, 4);
        INDArray arr2 = Nd4j.create(new float[]{1, 2, 3, 0, 1, 3, 1, 0, 2}, new long[]{3, 3}).castTo(DataType.INT);
        SDVariable x = sameDiff.var("x", arr1);
        SDVariable idxs = sameDiff.constant("idxs", arr2);
        SDVariable result = sameDiff.gatherNd(x, idxs);
        // build expected output array
        INDArray expected  = Nd4j.zeros(3);
        for (int i=0; i<3; i++){
            INDArray idx = arr2.get(point(i), NDArrayIndex.all());
            expected.putScalar(i, arr1.get(point(idx.getInt(0)),
                    point(idx.getInt(1)),
                    point(idx.getInt(2))).getDouble(0));
        }
        assertEquals(expected, result.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStack2(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 6, 6)).reshape(3, 2);
        INDArray arr2 = Transforms.sigmoid(Nd4j.linspace(7, 12, 6)).reshape(3, 2);
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable result = sameDiff.stack(1, x1, x2);
        assertArrayEquals(new long[]{3, 2, 2}, result.eval().shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParallelStack(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 6, 6)).reshape(3, 2);
        INDArray arr2 = Transforms.sigmoid(Nd4j.linspace(7, 12, 6)).reshape(3, 2);
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable result = sameDiff.stack(0, new SDVariable[]{x1, x2});
        assertArrayEquals(new long[]{2, 3, 2}, result.eval().shape());
        assertEquals(Nd4j.concat(0, arr1, arr2).reshape(2, 3, 2), result.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testUnStack2(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Nd4j.zeros(3, 2);
        INDArray arr2 = Nd4j.ones(3, 2);
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable stacked = sameDiff.stack(0, x1, x2);
        SDVariable[] result = sameDiff.unstack(stacked, 0, 2);
        assertEquals(arr1, result[0].eval());
        assertEquals(arr2, result[1].eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermuteSimple(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 6, 6).reshape(2, 3));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.permute(x, 1, 0);
        Map<String,INDArray> m = sameDiff.outputAll(null);
        assertArrayEquals(new long[]{3, 2}, m.get(result.name()).shape());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcat2(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(1,4);
        INDArray arr2 = Transforms.sigmoid(Nd4j.linspace(4, 8, 4)).reshape(1,4);
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable result = sameDiff.concat(0, x1, x2);
        assertArrayEquals(new long[]{2, 4}, result.eval().shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTile2(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(1,4));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.tile(x, new int[]{2, 2});
        assertArrayEquals(new long[]{2, 8}, result.eval().shape());
        INDArray arr2 = Nd4j.concat(0, arr, arr);  // (1, 4), (1, 4) -> (2, 4)
        INDArray expected = Nd4j.concat(1, arr2, arr2);  // (2, 4), (2, 4) -> (2, 8)
        assertEquals(expected, result.eval());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSlice2d(Nd4jBackend backend) {
        INDArray inArr = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice_full = sd.slice(in, new int[]{0, 0}, new int[]{3, 4});
        SDVariable subPart = sd.slice(in, new int[]{1, 2}, new int[]{2, 2});

        Map<String,INDArray> m = sd.outputAll(Collections.emptyMap());

        assertEquals(inArr, m.get(slice_full.name()));
        assertEquals(inArr.get(interval(1, 3), interval(2, 4)), m.get(subPart.name()));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSlice3d(Nd4jBackend backend) {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice_full = sd.slice(in, new int[]{0, 0, 0}, new int[]{3, 4, 5});
        SDVariable subPart = sd.slice(in, new int[]{1, 2, 3}, new int[]{2, 2, 1});

        Map<String,INDArray> m = sd.outputAll(null);

        assertEquals(inArr, m.get(slice_full.name()));
        assertEquals(inArr.get(interval(1, 3), interval(2, 4), interval(3, 4)), m.get(subPart.name()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSlice2dBasic(Nd4jBackend backend) {
        INDArray inArr = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice_full = sd.stridedSlice(in,new long[]{0, 0},new long[]{3, 4},new long[]{1, 1});
        SDVariable subPart = sd.stridedSlice(in,new long[]{1, 2},new long[]{3, 4},new long[]{1, 1});
        // SDVariable subPart2 = sd.stridedSlice(in,new long[]{0, 0},new long[]{4, 5},new long[]{2, 2});

        sd.outputAll(null);

        assertEquals(inArr, slice_full.getArr());
        assertEquals(inArr.get(interval(1, 3), interval(2, 4)), subPart.getArr());
        // assertEquals(inArr.get(interval(0, 2, 4), interval(0, 2, 5)), subPart2.getArr());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSliceBeginEndMask(Nd4jBackend backend) {
        INDArray inArr = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice1 = sd.stridedSlice(in,new long[]{-999, 0},new long[]{2, 4},new long[]{1, 1}, 1 << 1, 0, 0, 0, 0);
        SDVariable slice2 = sd.stridedSlice(in,new long[]{1, 0},new long[]{-999, 4},new long[]{1, 1}, 0, 1, 0, 0, 0);

        sd.outputAll(null);

        assertEquals(inArr.get(NDArrayIndex.interval(0, 2), NDArrayIndex.all()), slice1.getArr());
        assertEquals(inArr.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all()), slice2.getArr());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSliceEllipsisMask(Nd4jBackend backend) {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);

        //[1:3,...] -> [1:3,:,:]
        SDVariable slice = sd.stridedSlice(in,new long[]{1},new long[]{3},new long[]{1}, 0, 0, 1 << 1, 0, 0);
        //[1:3,...,1:4] -> [1:3,:,1:4]
        SDVariable slice2 = sd.stridedSlice(in,new long[]{1, 1},new long[]{3, 4},new long[]{1, 1}, 0, 0, 1 << 1, 0, 0);

        sd.outputAll(Collections.emptyMap());

        assertEquals(inArr.get(interval(1, 3), all(), all()), slice.getArr());
        assertEquals(inArr.get(interval(1, 3), all(), all()), slice2.getArr());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSliceNewAxisMask(Nd4jBackend backend) {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice = sd.stridedSlice(in,new long[]{-999, 0, 0, 0},new long[]{-999, 3, 4, 5},new long[]{-999, 1, 1, 1}, 0, 0, 0, 1, 0);

        INDArray out = slice.eval();

        assertArrayEquals(new long[]{1, 3, 4, 5}, out.shape());
        assertEquals(inArr, out.get(point(0), all(), all(), all()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSliceNewAxisMask2(Nd4jBackend backend) {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice = sd.stridedSlice(in,new long[]{1, 1, -999, 1},new long[]{3, 3, -999, 4},new long[]{1, 1, -999, 1}, 0, 0, 0, 1 << 2, 0);
        INDArray out = slice.eval();

        assertArrayEquals(new long[]{2, 2, 1, 3}, slice.getArr().shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSliceShrinkAxisMask(Nd4jBackend backend) {

        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice = sd.stridedSlice(in,new long[]{0, 0, 0},new long[]{-999, 4, 5},new long[]{1, 1, 1}, 0, 0, 0, 0, 1);
        SDVariable slice2 = sd.stridedSlice(in,new long[]{2, 0, 0},new long[]{-999, 4, 5},new long[]{1, 1, 1}, 0, 0, 0, 0, 1);
        SDVariable slice3 = sd.stridedSlice(in,new long[]{1, 2, 1},new long[]{-999, -999, 5},new long[]{1, 1, 1}, 0, 0, 0, 0, 1 | 1 << 1);

        sd.outputAll(null);

        assertEquals(inArr.get(point(0), all(), all()), slice.getArr());
        assertEquals(inArr.get(point(2), all(), all()), slice2.getArr());
        // For C-order [3,4,5] array with values 1-60: element [i,j,k] = i*20 + j*5 + k + 1
        // At [1,2,1:5]: values 32, 33, 34, 35
        INDArray expectedSlice3 = Nd4j.createFromArray(32.0, 33.0, 34.0, 35.0);
        assertEquals(expectedSlice3, slice3.getArr());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSizeAt_1(Nd4jBackend backend) {
        val array = Nd4j.create(10, 20, 30);
        val exp = Nd4j.scalar(DataType.LONG, 20);

        val op = new SizeAt(array, 1);

        Nd4j.getExecutioner().exec(op);

        val output = op.outputArguments().get(0);

        assertEquals(exp, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEye(Nd4jBackend backend) {
        int[] rows = new int[]{3,3,3,3};
        int[] cols = new int[]{3,2,2,2};
        int[][] batch = new int[][]{null, null, {4}, {3,3}};
        INDArray[] expOut = new INDArray[4];

        expOut[0] = Nd4j.eye(3);
        expOut[1] = Nd4j.create(new double[][]{{1,0},{0,1},{0,0}});
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
            log.info("Starting: " + i);
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSplit1(Nd4jBackend backend) {
        INDArray in = Nd4j.linspace(1,10,10).reshape(10);
        INDArray axis = Nd4j.scalar(-1);

        INDArray out1 = Nd4j.create(new long[]{5});
        INDArray out2 = Nd4j.create(new long[]{5});

        // Hardcode expected values to avoid CUDA issues with get() indexing
        // linspace(1,10,10) = [1,2,3,4,5,6,7,8,9,10]
        INDArray exp1 = Nd4j.createFromArray(1.0, 2.0, 3.0, 4.0, 5.0);
        INDArray exp2 = Nd4j.createFromArray(6.0, 7.0, 8.0, 9.0, 10.0);

        assertNull(OpValidation.validate(new OpTestCase(DynamicCustomOp.builder("split")
                .addInputs(axis, in)
                .addOutputs(out1, out2)
                .addIntegerArguments(2)
                .build()).expectedOutput(0, exp1).expectedOutput(1,exp2)));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSplit2(Nd4jBackend backend) {
        INDArray in = Nd4j.linspace(1,24,24).reshape(3,8);
        INDArray axis = Nd4j.scalar(-1);

        INDArray out1 = Nd4j.create(new long[]{3,4}, 'c');
        INDArray out2 = Nd4j.create(new long[]{3,4}, 'c');

        INDArray exp1 = in.get(NDArrayIndex.all(), NDArrayIndex.interval(0,4)).dup('c');
        INDArray exp2 = in.get(NDArrayIndex.all(), NDArrayIndex.interval(4,8)).dup('c');

        assertNull(OpValidation.validate(new OpTestCase(DynamicCustomOp.builder("split")
                .addInputs(axis, in)
                .addOutputs(out1, out2)
                .addIntegerArguments(2)
                .build()).expectedOutput(0, exp1).expectedOutput(1,exp2)));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDistancesExec(Nd4jBackend backend) {
        //https://github.com/eclipse/deeplearning4j/issues/7001
        for(String s : new String[]{"euclidean", "manhattan", "cosinesim", "cosinedist", "jaccard"}) {
            log.info("Starting: {}", s);
            INDArray defaultTestCase = Nd4j.create(4, 4);
            defaultTestCase.putRow(0, Nd4j.create(new float[]{0, 2, -2, 0}));
            defaultTestCase.putRow(1, Nd4j.create(new float[]{0, 1, -1, 0}));
            defaultTestCase.putRow(2, Nd4j.create(new float[]{0, -1, 1, 0}));
            defaultTestCase.putRow(3, Nd4j.create(new float[]{0, -2, 2, 0}));
            long singleEmbeddingSize = defaultTestCase.size(1) / 2L;

            // Split vectors
            INDArray x = defaultTestCase.get(NDArrayIndex.all(), NDArrayIndex.interval(0, singleEmbeddingSize));
            INDArray y = defaultTestCase.get(NDArrayIndex.all(), NDArrayIndex.interval(singleEmbeddingSize, defaultTestCase.size(1)));

            log.info(y.shapeInfoToString());

            SameDiff sd = SameDiff.create();
            sd.enableDebugMode();

            SDVariable xSd = sd.var("x", x);
            SDVariable ySd = sd.var("y", y);

            ySd = ySd.add(ySd);
            SDVariable dist;
            switch (s){
                case "euclidean":
                    dist = sd.math().euclideanDistance(s, ySd, xSd, 0);
                    break;
                case "manhattan":
                    dist = sd.math().manhattanDistance(s, ySd, xSd, 0);
                    break;
                case "cosinesim":
                    dist = sd.math().cosineSimilarity(s, ySd, xSd, 0);
                    break;
                case "cosinedist":
                    dist = sd.math().cosineDistance(s, ySd, xSd, 0);
                    break;
                case "jaccard":
                    dist = sd.math().jaccardDistance(s, ySd, xSd, 0);
                    break;
                default:
                    throw new RuntimeException();
            }

            SDVariable loss = dist.sum();


//            log.info(sd.summary());
            sd.output(Collections.emptyMap(), Lists.newArrayList(s));
            sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReductionShape(Nd4jBackend backend) {

        INDArray shape = Nd4j.createFromArray(4,2);
        INDArray axis = Nd4j.scalar(0);

        DynamicCustomOp op = DynamicCustomOp.builder("evaluate_reduction_shape")
                .addInputs(shape,axis)
                .addBooleanArguments(true) //keepdim = true
                .build();

        List<DataBuffer> list = op.calculateOutputShape();
        long[] shape2 = Shape.shape(list.get(0).asLong());
        long[] s = shape2;
        long[] exp = new long[]{2};         //(4,2).reduce(0,keepDims=true) -> [1,2] requires output array shape [2] here

        assertArrayEquals(exp, s);  //Fails - actual shape [1]
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void gatherTest(Nd4jBackend backend) {
        INDArray in = Nd4j.createFromArray(new double[][]{
                {1,2,3,4,5},
                {6,7,8,9,10},
                {11,12,13,14,15}});
        INDArray indices = Nd4j.createFromArray(2);
        INDArray axis = Nd4j.scalar(0);

        DynamicCustomOp op = DynamicCustomOp.builder("gather")
                .addInputs(in, indices, axis)
                .build();

        List<DataBuffer> shapeList = op.calculateOutputShape();
        long[] shape = Shape.shape(shapeList.get(0).asLong());
        long[] expShape = new long[]{1,5};
        assertArrayEquals(expShape, shape);     //Fails: actual shape: [5]
    }
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSliceShape(Nd4jBackend backend) {

        INDArray arr = Nd4j.arange(0, 25).reshape(1,5,5).castTo(DataType.INT);

        INDArray begin = Nd4j.createFromArray(0, 1, 2);
        INDArray size = Nd4j.createFromArray(-1, -1, -1);

        DynamicCustomOp op = DynamicCustomOp.builder("slice")
                .addInputs(arr, begin, size)
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        long[] shape = Shape.shape(l.get(0).asLong());
        long[] shapeExp = new long[]{1,4,3};

        assertArrayEquals(shapeExp, shape);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereAllFalse(Nd4jBackend backend) {
        INDArray in = Nd4j.create(DataType.BOOL, 1917);
        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(in)
                .build();
        List<DataBuffer> l = op.calculateOutputShape();
        INDArray[] out = Nd4j.exec(op);
        long[] shape = Shape.shape(l.get(0).asLong());
        boolean isEmpty = Shape.isEmpty(l.get(0).asLong());
        assertEquals(1, out.length);
        assertTrue(out[0].isEmpty());
        assertTrue(isEmpty);    //Not empty, but should be
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGatherScalar(Nd4jBackend backend) {
        INDArray in = Nd4j.linspace(100, 200, 100, DataType.FLOAT).reshape(100);
        INDArray indices = Nd4j.scalar(0);
        INDArray axis = Nd4j.scalar(0);

        DynamicCustomOp op = DynamicCustomOp.builder("gather")
                .addInputs(in, indices, axis)
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        long[] shape = Shape.shape(l.get(0).asLong());
        assertArrayEquals(new long[0], shape);

        INDArray arr = Nd4j.createFromDescriptor(l.get(0));

        op.addOutputArgument(arr);

        Nd4j.exec(op);

        INDArray exp = Nd4j.scalar(DataType.FLOAT, 100);
        assertEquals(exp, arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCastEmpty(Nd4jBackend backend) {
        INDArray emptyLong = Nd4j.empty(DataType.LONG);
        int dtype = 9;  //INT = 9 - https://github.com/eclipse/deeplearning4j/blob/master/libnd4j/include/array/DataType.h
        DynamicCustomOp op = DynamicCustomOp.builder("cast")
                .addInputs(emptyLong)
                .addIntegerArguments(dtype)
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        long[] shape = Shape.shape(l.get(0).asLong());
        boolean isEmpty = Shape.isEmpty(l.get(0).asLong());
        assertEquals(0, shape.length);
        assertTrue(isEmpty);
    }
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGatherEmpty(Nd4jBackend backend) {
    /*
    tf.reset_default_graph()
    inputFloat = tf.constant([], shape=[0,2,3], dtype=tf.float32)
    emptyInt = tf.constant([], shape=[0], dtype=tf.int32)

    gather = tf.gather(params=inputFloat, indices=emptyInt)

    sess = tf.Session()
    out = sess.run([gather])
    print(out[0].shape)
    print(out[0]);

    > (0, 2, 3)
    > []
     */
        INDArray emptyFloat = Nd4j.create(DataType.FLOAT, 0, 2, 3);
        INDArray emptyInt = Nd4j.create(DataType.INT, 0);
        DynamicCustomOp op = DynamicCustomOp.builder("gather")
                .addInputs(emptyFloat, emptyInt)
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        assertEquals(1, l.size());
        assertTrue(Shape.isEmpty(l.get(0).asLong()));
        assertArrayEquals(new long[]{0,2,3}, Shape.shape(l.get(0).asLong()));

        INDArray out = Nd4j.empty(DataType.FLOAT);
        op.addOutputArgument(out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSplitEmpty(Nd4jBackend backend) {
    /*
    tf.reset_default_graph()
    # Hack to create empty array
    input = tf.constant([False], dtype=tf.bool)
    empty = tf.where(condition=input)
    empty = tf.reshape(empty, [0,4])
    emptyFloat = tf.cast(empty, tf.float32)
    const1 = tf.constant(1, dtype=tf.int32)
    split = tf.split(value=emptyFloat, num_or_size_splits=4, axis=1)
    sess = tf.Session()
    out = sess.run([split])
    # print(out[0].shape);
    print(out[0]);
     */

        INDArray emptyIn = Nd4j.create(DataType.FLOAT, 0, 4);
        INDArray axis = Nd4j.scalar(1);

        DynamicCustomOp op = DynamicCustomOp.builder("split")
                .addInputs(axis, emptyIn)
                .addIntegerArguments(4) //num_splits = 4
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        assertEquals(4, l.size());
        for( int i=0; i<4; i++ ){
            val desc = l.get(i);
            assertArrayEquals(new long[]{0, 1}, Shape.shape(desc.asLong()));
            assertTrue(Shape.isEmpty(desc.asLong()));
            op.addOutputArgument(Nd4j.create(DataType.FLOAT, Shape.shape(desc.asLong())));
        }

        Nd4j.exec(op);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcatEmpty(Nd4jBackend backend) {
    /*
    TF behaviour with concatenation of empty arrays:
    concat(empty,empty,empty) -> empty
    cotcat(empty,nonEmpty) -> nonEmpty, etc (i.e., empty arrays are ignored)

    tf.reset_default_graph()
    input = tf.constant([False], dtype=tf.bool)
    emptyFloat = tf.constant([], shape=[0,1], dtype=tf.float32)
    var11 = tf.constant([1], dtype=tf.float32, shape=[1,1])

    concat = tf.concat(values=[emptyFloat, emptyFloat, var11, emptyFloat], axis=0)

    sess = tf.Session()
    out = sess.run([concat])
    print(out[0].shape)
    print(out[0]);
     */

        INDArray one1 = Nd4j.create(DataType.FLOAT, 1, 1);
        INDArray empty01 = Nd4j.create(DataType.FLOAT, 0, 1);

        DynamicCustomOp op = DynamicCustomOp.builder("concat")
                .addInputs(empty01, empty01, empty01)
                .addIntegerArguments(0) //axis = 0
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        assertEquals(1, l.size());
        long[] shapeInfo = l.get(0).asLong();
        assertTrue(Shape.isEmpty(shapeInfo));
        assertArrayEquals(new long[]{0, 1}, Shape.shape(shapeInfo));

        op.addOutputArgument(Nd4j.create(DataType.FLOAT, 0, 1));
        Nd4j.exec(op);


        op = DynamicCustomOp.builder("concat")
                .addInputs(empty01, empty01, one1, empty01)
                .addIntegerArguments(0) //axis = 0
                .build();
        l = op.calculateOutputShape();
        assertEquals(1, l.size());
        assertFalse(Shape.isEmpty(l.get(0).asLong()));
        assertArrayEquals(new long[]{1, 1}, Shape.shape(l.get(0).asLong()));
        op.addOutputArgument(Nd4j.create(DataType.FLOAT, 1, 1));
        Nd4j.exec(op);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcatEmpty2(Nd4jBackend backend) {
        INDArray empty10a = Nd4j.create(DataType.INT, 1, 0);
        INDArray empty10b = Nd4j.create(DataType.INT, 1, 0);

        DynamicCustomOp op = DynamicCustomOp.builder("concat")
                .addInputs(empty10a, empty10b)
                .addIntegerArguments(0) //axis = 0
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        assertEquals(1, l.size());
        assertTrue(Shape.isEmpty(l.get(0).asLong()));
        assertArrayEquals(new long[]{2, 0}, Shape.shape(l.get(0).asLong()));
        // Shape info buffers store 64-bit values, so dataType() returns LONG
        // Use Shape.dataType() to extract the encoded array data type
        assertEquals(DataType.INT, Shape.dataType(l.get(0).asLong()));

        op.addOutputArgument(Nd4j.create(DataType.INT, 2, 0));
        Nd4j.exec(op);


        op = DynamicCustomOp.builder("concat")
                .addInputs(empty10a, empty10b)
                .addIntegerArguments(1) //axis = 1
                .build();
        l = op.calculateOutputShape();
        assertEquals(1, l.size());
        assertTrue(Shape.isEmpty(l.get(0).asLong()));
        assertArrayEquals(new long[]{1, 0}, Shape.shape(l.get(0).asLong()));
        op.addOutputArgument(Nd4j.create(DataType.INT, 1, 0));
        Nd4j.exec(op);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyGather(Nd4jBackend backend) {
    /*
    tf.reset_default_graph()
    inputFloat = tf.constant([], shape=[0,2,3], dtype=tf.float32)
    emptyInt = tf.constant([], shape=[0], dtype=tf.int32)

    gather = tf.gather(params=inputFloat, indices=emptyInt)

    sess = tf.Session()
    out = sess.run([gather])
    print(out[0].shape)
    print(out[0]);

    > (0, 2, 3)
    > []
     */
        INDArray emptyFloat = Nd4j.create(DataType.FLOAT, 0, 2, 3);
        INDArray emptyInt = Nd4j.create(DataType.INT, 0);
        DynamicCustomOp op = DynamicCustomOp.builder("gather")
                .addInputs(emptyFloat, emptyInt)
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        assertEquals(1, l.size());
        assertTrue(Shape.isEmpty(l.get(0).asLong()));
        assertArrayEquals(new long[]{0,2,3}, Shape.shape(l.get(0).asLong()));

        INDArray out = Nd4j.empty(DataType.FLOAT);
        op.addOutputArgument(out);
    }
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastDynamicShape1(Nd4jBackend backend) {

        //Test case: [2,1] and [4]: expect [2,4]
        INDArray out = Nd4j.create(DataType.INT, 2);
        DynamicCustomOp op = DynamicCustomOp.builder("broadcast_dynamic_shape")
                .addInputs(Nd4j.createFromArray(new int[]{2,1}), Nd4j.createFromArray(new int[]{4}))
                .addOutputs(out)
                .build();
        Nd4j.getExecutioner().exec(op);
        assertEquals(Nd4j.createFromArray(new int[]{2,4}), out);

        //Same thing, reversed input order (expect same output)
        op = DynamicCustomOp.builder("broadcast_dynamic_shape")
                .addInputs(Nd4j.createFromArray(new int[]{4}), Nd4j.createFromArray(new int[]{2,1}))
                .addOutputs(out)
                .build();
        Nd4j.getExecutioner().exec(op);
        assertEquals(Nd4j.createFromArray(new int[]{2,4}), out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastDynamicShape2(Nd4jBackend backend) {

        //Test case: [2,1,4] and [2,2,4]: expect [2,2,4]
        INDArray out = Nd4j.create(DataType.INT, 3);
        DynamicCustomOp op = DynamicCustomOp.builder("broadcast_dynamic_shape")
                .addInputs(Nd4j.createFromArray(new int[]{2,1,4}), Nd4j.createFromArray(new int[]{2,2,4}))
                .addOutputs(out)
                .build();
        Nd4j.getExecutioner().exec(op);
        assertEquals(Nd4j.createFromArray(new int[]{2,2,4}), out);

        //Test case: [1,1,3] and [2,4,1]: expect [2,4,3]
        out = Nd4j.create(DataType.INT, 3);
        op = DynamicCustomOp.builder("broadcast_dynamic_shape")
                .addInputs(Nd4j.createFromArray(new int[]{1,1,3}), Nd4j.createFromArray(new int[]{2,4,1}))
                .addOutputs(out)
                .build();
        Nd4j.getExecutioner().exec(op);
        assertEquals(Nd4j.createFromArray(new int[]{2,4,3}), out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSliceShrinkAxis(Nd4jBackend backend) {
        INDArray in = Nd4j.create(DataType.DOUBLE, 3,2,2);
        INDArray begin = Nd4j.createFromArray(2);
        INDArray end = Nd4j.createFromArray(3);         //Should be ignored due to shrink_axis_mask
        INDArray stride = Nd4j.createFromArray(1);      //Should be ignored due to shrink_axis_mask

        DynamicCustomOp op = DynamicCustomOp.builder("strided_slice")
                .addInputs(in, begin, end, stride)
                .addIntegerArguments(
                        0,  //begin mask
                        0,  //ellipsis mask
                        0,  //end mask
                        0,  //new axis mask
                        1   //shrink axis mask
                )
                .build();

        List<DataBuffer> lsd = op.calculateOutputShape();
        assertEquals(1, lsd.size());
        long[] shape = Shape.shape(lsd.get(0).asLong());
        long[] exp = new long[]{2,2};
        assertArrayEquals(exp, shape);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSliceEmpty(Nd4jBackend backend) {

        INDArray in = Nd4j.createFromArray(10); //Integer, Length 1, rank 1, value 10   - Not used due to begin mask!
        INDArray from = Nd4j.createFromArray(0);
        INDArray to = Nd4j.createFromArray(0);
        INDArray stride = Nd4j.createFromArray(1);

        DynamicCustomOp op = DynamicCustomOp.builder("stridedslice")
                .addInputs(in, from, to, stride)
                .addIntegerArguments(1,0,0,0,0) //Begin mask, ellipsis, end, new axis, shrink
                .build();

        List<DataBuffer> s = Nd4j.getExecutioner().calculateOutputShape(op);
        assertEquals(1, s.size());

        //Is returning shape [0], should be empty
        boolean isEmpty = Shape.isEmpty(s.get(0).asLong());
        assertTrue(isEmpty);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSliceEdgeCase(Nd4jBackend backend) {
        INDArray in = Nd4j.scalar(10).reshape(1);   //Int [1]
        INDArray begin = Nd4j.ones(DataType.INT, 1);
        INDArray end = Nd4j.zeros(DataType.INT, 1);
        INDArray stride = Nd4j.ones(DataType.INT, 1);

        DynamicCustomOp op = DynamicCustomOp.builder("strided_slice")
                .addInputs(in, begin, end, stride)
                .addIntegerArguments(0, //Begin mask
                        0,  //Ellipsis mask
                        1,  //End mask
                        0,  //New axis mask
                        0)  //Shrink axis mask
                .addOutputs(Nd4j.create(DataType.INT, 0))  // Empty 1D array with shape [0]
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        assertEquals(1, l.size());
        // Shape info buffers store 64-bit values, so dataType() returns LONG
        // Use Shape.dataType() to extract the encoded array data type
        assertEquals(DataType.INT, Shape.dataType(l.get(0).asLong()));
        assertTrue(Shape.isEmpty(l.get(0).asLong())); //Should be empty 1D array with shape [0]

        Nd4j.exec(op);  //Execution is OK
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptySlice1(Nd4jBackend backend) {
        INDArray in = Nd4j.createFromArray(38);
        INDArray begin = Nd4j.createFromArray(1);
        INDArray size = Nd4j.createFromArray(-1);

        DynamicCustomOp op = DynamicCustomOp.builder("slice")
                .addInputs(in, begin, size)
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        assertTrue(Shape.isEmpty(l.get(0).asLong()));

        INDArray out = Nd4j.create(DataType.INT, 0);
        op.setOutputArgument(0, out);

        Nd4j.exec(op);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptySlice2(Nd4jBackend backend) {
        INDArray in = Nd4j.createFromArray(38);
        INDArray begin = Nd4j.createFromArray(0);
        INDArray size = Nd4j.createFromArray(0);

        DynamicCustomOp op = DynamicCustomOp.builder("slice")
                .addInputs(in, begin, size)
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        assertTrue(Shape.isEmpty(l.get(0).asLong()));

        INDArray out = Nd4j.create(DataType.INT, 0);
        op.setOutputArgument(0, out);

        Nd4j.exec(op);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFill(Nd4jBackend backend) {

        INDArray shape = Nd4j.createFromArray(0,4);
        INDArray value = Nd4j.scalar(1.0f);

        DynamicCustomOp op = DynamicCustomOp.builder("fill")
                .addInputs(shape, value)
                .build();

        List<DataBuffer> l = op.calculateOutputShape();
        assertEquals(1, l.size());
        assertArrayEquals(new long[]{0,4}, Shape.shape(l.get(0).asLong()));
        assertTrue(Shape.isEmpty(l.get(0).asLong()));

        op.setOutputArgument(0, Nd4j.create(DataType.FLOAT, 0, 4));
        Nd4j.exec(op);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFill2(Nd4jBackend backend) {

        INDArray shape = Nd4j.createFromArray(0,4);
        INDArray value = Nd4j.scalar(1.0f);

        DynamicCustomOp op = new Fill(shape, value, null);

        List<DataBuffer> l = op.calculateOutputShape();
        assertEquals(1, l.size());
        assertTrue(Shape.isEmpty(l.get(0).asLong()));
        assertArrayEquals(new long[]{0,4}, Shape.shape(l.get(0).asLong()));

        op.setOutputArgument(0, Nd4j.create(DataType.FLOAT, 0, 4));
        Nd4j.exec(op);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermuteShapeDynamicAxis(Nd4jBackend backend) {

        DynamicCustomOp op = DynamicCustomOp.builder("permute")
                .addInputs(Nd4j.rand(DataType.FLOAT, 3, 4),
                        Nd4j.createFromArray(1, 0))
                .build();
        List<DataBuffer> l = op.calculateOutputShape();
        assertArrayEquals(new long[]{4, 3}, Shape.shape(l.get(0).asLong()));

        op = DynamicCustomOp.builder("permute")
                .addInputs(Nd4j.rand(DataType.FLOAT, 3, 4))
                .addIntegerArguments(1, 0)
                .build();
        l = op.calculateOutputShape();
        assertArrayEquals(new long[]{4, 3}, Shape.shape(l.get(0).asLong()));


        op = DynamicCustomOp.builder("permute")
                .addInputs(Nd4j.rand(DataType.FLOAT, 3, 4, 5),
                        Nd4j.createFromArray(1, 2, 0))
                .build();
        l = op.calculateOutputShape();
        assertArrayEquals(new long[]{4, 5, 3}, Shape.shape(l.get(0).asLong()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGather2(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.var("in", Nd4j.arange(6).castTo(DataType.DOUBLE).reshape(2,3));
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(0));

        SDVariable gathered = sd.gather(input, indices, 0);
        SDVariable loss = gathered.std(true);

        Map<String, INDArray> output = sd.output((Map<String, INDArray>) null, gathered.name());
        sd.setLossVariables(loss.name());

        String err = OpValidation.validate(new TestCase(sd)
                .gradCheckEpsilon(1e-3)
                .gradCheckMaxRelativeError(1e-4));

        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermute3(Nd4jBackend backend) {
        INDArray in = Nd4j.linspace(DataType.FLOAT, 1, 6, 1).reshape(3,2);
        INDArray permute = Nd4j.createFromArray(1,0);

//        System.out.println(in);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var(in);
        SDVariable v2 = sd.constant(permute);

        SDVariable out = v.permute(v2);

        INDArray exp = in.transpose();
        INDArray outArr = out.eval();
        assertEquals(exp, outArr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermute4(Nd4jBackend backend) {
        INDArray in = Nd4j.linspace(DataType.FLOAT, 1, 6, 1).reshape(3,2);
        INDArray permute = Nd4j.createFromArray(1,0);

        INDArray exp = in.transpose();

        for( boolean iargs : new boolean[]{true, false}) {


            DynamicCustomOp.DynamicCustomOpsBuilder b = DynamicCustomOp.builder("permute")
                    .addInputs(in)
                    .addOutputs(Nd4j.create(DataType.FLOAT, 2, 3));

            if(iargs){
                b.addIntegerArguments(1, 0);
            } else {
                b.addInputs(permute);
            }

            DynamicCustomOp op = b.build();
            Nd4j.exec(op);

//            System.out.println(in);
//            System.out.println(op.outputArguments().get(0));

            assertEquals(exp, op.getOutputArgument(0));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInvertPermutation(Nd4jBackend backend) {
        DynamicCustomOp op = DynamicCustomOp.builder("invert_permutation")
                .addInputs(Nd4j.createFromArray(1, 0))
                .build();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastInt1(Nd4jBackend backend) {

        INDArray out = Nd4j.create(DataType.INT, 1);
        DynamicCustomOp op = DynamicCustomOp.builder("broadcast_dynamic_shape")
                .addInputs(Nd4j.createFromArray(1), Nd4j.createFromArray(4))
                .addOutputs(out)
                .build();
        Nd4j.getExecutioner().exec(op);
        assertEquals(Nd4j.createFromArray(4), out);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastInt2(Nd4jBackend backend) {
        INDArray out = Nd4j.create(DataType.INT, 2);
        DynamicCustomOp op = DynamicCustomOp.builder("broadcast_dynamic_shape")
                .addInputs(Nd4j.createFromArray(2, 2), Nd4j.createFromArray(1))
                .addOutputs(out)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertEquals(Nd4j.createFromArray(2, 2), out);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeZeros(Nd4jBackend backend) {
        int[][] shapes = new int[][]{{2,0}, {10,0},    {10, 0},  {2,0,0,10}, {10, 0},   {0, 0, 10},  {0,2,10}, {1,2,0}};
        int[][] reshape = new int[][]{{2,-1}, {2,0,-1}, {5,2,-1}, {2,0,-1},   {-1, 2, 0}, {2, -1, 0}, {2, 0, 0, 0, -1}, {2,0,-1}};
        int[][] expected = new int[][]{{2,0}, {2,0,5}, {5,2,0}, {2,0,10}, {5,2,0}, {2,5,0}, {2,0,0,0,10}, {2,0,1}};

        for( int i=0; i<shapes.length; i++ ){
            System.out.println(i);
            long[] orig = ArrayUtil.toLongArray(shapes[i]);
            int[] r = reshape[i];
            long[] exp = ArrayUtil.toLongArray(expected[i]);

            SameDiff sd = SameDiff.create();
            SDVariable v = sd.placeHolder("orig", DataType.FLOAT, orig);
            SDVariable rs = v.reshape(r);
            SDVariable rs2 = v.reshape(sd.constant(Nd4j.createFromArray(r)));

            INDArray out = rs.eval(Collections.singletonMap("orig", Nd4j.create(DataType.FLOAT, orig)));
            assertArrayEquals(exp, out.shape());

            out = rs2.eval(Collections.singletonMap("orig", Nd4j.create(DataType.FLOAT, orig)));
            assertArrayEquals(exp, out.shape());
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMergeMaxIndex(Nd4jBackend backend) {

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable inputX = sd.var(Nd4j.createFromArray(new float[] {1, 0, 0}));
        SDVariable inputY = sd.var(Nd4j.createFromArray(new float[] {0, 1, 0}));
        SDVariable inputZ = sd.var(Nd4j.createFromArray(new float[] {0, 0, 1}));
        SDVariable out = new MergeMaxIndex(sd, new SDVariable[]{inputX, inputY, inputZ},DataType.INT32).outputVariable();
        INDArray expected = Nd4j.createFromArray(0,1,2);
        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("mergemaxindex", expected)
                .gradientCheck(false));
        assertNull(err);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTriOp(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable out = new Tri(sd, DataType.INT32, 3, 5, 2).outputVariable();
        INDArray expected = Nd4j.createFromArray(new int[][]{{1, 1, 1, 0, 0}, {1, 1, 1, 1, 0}, {1, 1, 1, 1, 1}});
        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("tri", expected)
                .gradientCheck(false));
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTriuOp(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.var(Nd4j.createFromArray(new double[][]{{1,2,3}, {4,5,6}, {7,8,9},{10,11,12}}));
        SDVariable out = new Triu(sd, input,-1).outputVariable();
        out.markAsLoss();
        INDArray expected = Nd4j.createFromArray(new double[][]{{1,2,3}, {4,5,6}, {0,8,9},{0,0,12}});
        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("triu", expected)
                .gradientCheck(true));
        assertNull(err);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRepeat(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        // Test 1: Simple repeat along axis 0 with uniform repeats
        INDArray in1 = Nd4j.createFromArray(new double[][]{{1, 2}, {3, 4}});
        // Repeat each row 2 times along axis 0
        // Expected: [[1,2], [1,2], [3,4], [3,4]]
        INDArray exp1 = Nd4j.createFromArray(new double[][]{{1, 2}, {1, 2}, {3, 4}, {3, 4}});

        DynamicCustomOp op1 = DynamicCustomOp.builder("repeat")
                .addInputs(in1)
                .addIntegerArguments(2, 0)  // repeats=2, axis=0
                .build();
        List<DataBuffer> shapes1 = op1.calculateOutputShape();
        INDArray out1 = Nd4j.createFromDescriptor(shapes1.get(0));
        op1.addOutputArgument(out1);
        Nd4j.exec(op1);
        assertEquals(exp1, out1);

        // Test 2: Repeat along axis 1 with uniform repeats
        INDArray in2 = Nd4j.createFromArray(new double[][]{{1, 2, 3}, {4, 5, 6}});
        // Repeat each column 3 times along axis 1
        // Expected: [[1,1,1,2,2,2,3,3,3], [4,4,4,5,5,5,6,6,6]]
        INDArray exp2 = Nd4j.createFromArray(new double[][]{
                {1, 1, 1, 2, 2, 2, 3, 3, 3},
                {4, 4, 4, 5, 5, 5, 6, 6, 6}
        });

        DynamicCustomOp op2 = DynamicCustomOp.builder("repeat")
                .addInputs(in2)
                .addIntegerArguments(3, 1)  // repeats=3, axis=1
                .build();
        List<DataBuffer> shapes2 = op2.calculateOutputShape();
        INDArray out2 = Nd4j.createFromDescriptor(shapes2.get(0));
        op2.addOutputArgument(out2);
        Nd4j.exec(op2);
        assertEquals(exp2, out2);

        // Test 3: Non-uniform repeats (different repeat count per element)
        INDArray in3 = Nd4j.createFromArray(new double[]{1, 2, 3});
        // Repeat with [1, 2, 3] along axis 0 means:
        // 1 repeated 1 time, 2 repeated 2 times, 3 repeated 3 times
        // Expected: [1, 2, 2, 3, 3, 3]
        INDArray exp3 = Nd4j.createFromArray(new double[]{1, 2, 2, 3, 3, 3});

        DynamicCustomOp op3 = DynamicCustomOp.builder("repeat")
                .addInputs(in3)
                .addIntegerArguments(1, 2, 3, 0)  // repeats=[1,2,3], axis=0
                .build();
        List<DataBuffer> shapes3 = op3.calculateOutputShape();
        INDArray out3 = Nd4j.createFromDescriptor(shapes3.get(0));
        op3.addOutputArgument(out3);
        Nd4j.exec(op3);
        assertEquals(exp3, out3);

        // Test 4: 3D array repeat along axis 2
        INDArray in4 = Nd4j.createFromArray(new double[][][]{
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}}
        });
        // Repeat each element 2 times along axis 2
        INDArray exp4 = Nd4j.createFromArray(new double[][][]{
                {{1, 1, 2, 2}, {3, 3, 4, 4}},
                {{5, 5, 6, 6}, {7, 7, 8, 8}}
        });

        DynamicCustomOp op4 = DynamicCustomOp.builder("repeat")
                .addInputs(in4)
                .addIntegerArguments(2, 2)  // repeats=2, axis=2
                .build();
        List<DataBuffer> shapes4 = op4.calculateOutputShape();
        INDArray out4 = Nd4j.createFromDescriptor(shapes4.get(0));
        op4.addOutputArgument(out4);
        Nd4j.exec(op4);
        assertEquals(exp4, out4);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRollAxis(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        // Test 1: Roll axis 2 to position 0 for a 3D array
        // Shape [2,3,4] with rollAxis(2) -> axis 2 moves to front -> shape [4,2,3]
        INDArray in1 = Nd4j.linspace(1, 24, 24).reshape(2, 3, 4);
        INDArray result1 = Nd4j.rollAxis(in1, 2);  // default start=0
        assertArrayEquals(new long[]{4, 2, 3}, result1.shape());
        // rollAxis(a, 2, 0) is equivalent to permute(2, 0, 1)
        INDArray exp1 = in1.permute(2, 0, 1);
        assertEquals(exp1, result1);

        // Test 2: Roll axis 0 to position 2
        INDArray in2 = Nd4j.linspace(1, 24, 24).reshape(2, 3, 4);
        INDArray result2 = Nd4j.rollAxis(in2, 0, 2);
        // rollAxis(a, 0, 2) moves axis 0 to before position 2
        // For rank 3: axis=0, start=2 -> permute(1, 0, 2)
        assertArrayEquals(new long[]{3, 2, 4}, result2.shape());
        INDArray exp2 = in2.permute(1, 0, 2);
        assertEquals(exp2, result2);

        // Test 3: Roll axis 1 to position 0
        INDArray in3 = Nd4j.linspace(1, 60, 60).reshape(3, 4, 5);
        INDArray result3 = Nd4j.rollAxis(in3, 1, 0);
        // rollAxis(a, 1, 0) moves axis 1 to position 0 -> permute(1, 0, 2)
        assertArrayEquals(new long[]{4, 3, 5}, result3.shape());
        INDArray exp3 = in3.permute(1, 0, 2);
        assertEquals(exp3, result3);

        // Test 4: Roll axis with negative index
        INDArray in4 = Nd4j.linspace(1, 24, 24).reshape(2, 3, 4);
        INDArray result4 = Nd4j.rollAxis(in4, -1);  // -1 means last axis (axis 2)
        assertArrayEquals(new long[]{4, 2, 3}, result4.shape());
        INDArray exp4 = in4.permute(2, 0, 1);
        assertEquals(exp4, result4);

        // Test 5: Roll when axis equals start (should return unchanged)
        INDArray in5 = Nd4j.linspace(1, 24, 24).reshape(2, 3, 4);
        INDArray result5 = Nd4j.rollAxis(in5, 1, 1);
        // When axis == start, array is returned unchanged
        assertArrayEquals(new long[]{2, 3, 4}, result5.shape());
        assertEquals(in5, result5);

        // Test 6: 4D array roll axis
        INDArray in6 = Nd4j.linspace(1, 120, 120).reshape(2, 3, 4, 5);
        INDArray result6 = Nd4j.rollAxis(in6, 3, 1);
        // rollAxis(a, 3, 1) moves axis 3 to before position 1 -> permute(0, 3, 1, 2)
        assertArrayEquals(new long[]{2, 5, 3, 4}, result6.shape());
        INDArray exp6 = in6.permute(0, 3, 1, 2);
        assertEquals(exp6, result6);
    }

    /**
     * Test for stacking scalar constants with sizeAt results.
     * This reproduces an issue where scalar constants were being confused
     * with other values when stacked together.
     *
     * IMPORTANT: The bug was that Nd4j.createFromArray(-1L) creates shape [1] (1D)
     * while sizeAt returns a true scalar with shape [] (rank 0).
     * Stack requires all inputs to have the same shape, so we must use
     * Nd4j.scalar() to create true scalars.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStackScalarConstantWithSizeAt(Nd4jBackend backend) {
        // Test case: mimics the MatMul reshape scenario in ONNX import
        // Input shape is [196, 1, 768] (like ViT patch embeddings)
        SameDiff sd = SameDiff.create();

        // Create input with known shape
        INDArray inputArr = Nd4j.rand(DataType.FLOAT, 196, 1, 768);
        SDVariable input = sd.var("input", inputArr);

        // IMPORTANT: Use Nd4j.scalar() to create a true scalar with shape []
        // NOT Nd4j.createFromArray(-1L) which creates shape [1]
        SDVariable minusOne = sd.constant("minus_one", Nd4j.scalar(DataType.INT64, -1L));

        // Get size at last dimension (should be 768) - this returns shape []
        SDVariable lastDim = sd.sizeAt("last_dim", input, -1);

        // Stack them: should produce [2] with values [-1, 768]
        SDVariable stacked = sd.stack("stacked", 0, minusOne, lastDim);

        // Evaluate
        Map<String, INDArray> results = sd.output(Collections.emptyMap(), "minus_one", "last_dim", "stacked");

        INDArray minusOneResult = results.get("minus_one");
        INDArray lastDimResult = results.get("last_dim");
        INDArray stackedResult = results.get("stacked");

        // Verify shape of inputs (both should be scalars with shape [])
        assertArrayEquals(new long[]{}, minusOneResult.shape(), "Constant should have scalar shape []");
        assertArrayEquals(new long[]{}, lastDimResult.shape(), "sizeAt result should have scalar shape []");

        // Verify individual values
        assertEquals(-1L, minusOneResult.getLong(0), "Constant -1 should be -1");
        assertEquals(768L, lastDimResult.getLong(0), "sizeAt(-1) should be 768");

        // Verify stacked result
        assertArrayEquals(new long[]{2}, stackedResult.shape(), "Stacked shape should be [2]");
        assertEquals(-1L, stackedResult.getLong(0), "First element of stacked should be -1");
        assertEquals(768L, stackedResult.getLong(1), "Second element of stacked should be 768");

        log.info("Stack scalar test passed: minusOne={}, lastDim={}, stacked={}",
                minusOneResult, lastDimResult, stackedResult);
    }

    /**
     * Test for stacking multiple sizeAt results with constants.
     * This tests a more complex scenario with multiple dynamic shape values.
     *
     * IMPORTANT: Use Nd4j.scalar() not Nd4j.createFromArray() for scalar constants
     * when stacking with sizeAt results, because sizeAt returns true scalars [].
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStackMultipleSizeAtWithConstants(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        // Create inputs with known shapes
        INDArray aArr = Nd4j.rand(DataType.FLOAT, 64, 196, 768);  // [batch, seq, hidden]
        INDArray bArr = Nd4j.rand(DataType.FLOAT, 768, 256);      // [hidden, out]

        SDVariable a = sd.var("a", aArr);
        SDVariable b = sd.var("b", bArr);

        // Get various dimensions - all return true scalars with shape []
        SDVariable aLastDim = sd.sizeAt("a_last_dim", a, -1);      // Should be 768
        SDVariable bLastDim = sd.sizeAt("b_last_dim", b, -1);      // Should be 256
        SDVariable aSecondDim = sd.sizeAt("a_second_dim", a, 1);   // Should be 196

        // Create scalar constants (use Nd4j.scalar for shape [] to match sizeAt output)
        SDVariable minusOne = sd.constant("const_minus_one", Nd4j.scalar(DataType.INT64, -1L));
        SDVariable zero = sd.constant("const_zero", Nd4j.scalar(DataType.INT64, 0L));

        // Test stacking different combinations
        SDVariable stack1 = sd.stack("stack1", 0, minusOne, aLastDim);           // [-1, 768]
        SDVariable stack2 = sd.stack("stack2", 0, aSecondDim, bLastDim);         // [196, 256]
        SDVariable stack3 = sd.stack("stack3", 0, zero, minusOne, aLastDim);     // [0, -1, 768]

        // Evaluate all
        Map<String, INDArray> results = sd.output(Collections.emptyMap(),
                "a_last_dim", "b_last_dim", "a_second_dim",
                "const_minus_one", "const_zero",
                "stack1", "stack2", "stack3");

        // Verify individual sizeAt results
        assertEquals(768L, results.get("a_last_dim").getLong(0), "a sizeAt(-1) should be 768");
        assertEquals(256L, results.get("b_last_dim").getLong(0), "b sizeAt(-1) should be 256");
        assertEquals(196L, results.get("a_second_dim").getLong(0), "a sizeAt(1) should be 196");

        // Verify constants
        assertEquals(-1L, results.get("const_minus_one").getLong(0), "const -1 should be -1");
        assertEquals(0L, results.get("const_zero").getLong(0), "const 0 should be 0");

        // Verify stacked results
        INDArray s1 = results.get("stack1");
        assertArrayEquals(new long[]{2}, s1.shape());
        assertEquals(-1L, s1.getLong(0), "stack1[0] should be -1");
        assertEquals(768L, s1.getLong(1), "stack1[1] should be 768");

        INDArray s2 = results.get("stack2");
        assertArrayEquals(new long[]{2}, s2.shape());
        assertEquals(196L, s2.getLong(0), "stack2[0] should be 196");
        assertEquals(256L, s2.getLong(1), "stack2[1] should be 256");

        INDArray s3 = results.get("stack3");
        assertArrayEquals(new long[]{3}, s3.shape());
        assertEquals(0L, s3.getLong(0), "stack3[0] should be 0");
        assertEquals(-1L, s3.getLong(1), "stack3[1] should be -1");
        assertEquals(768L, s3.getLong(2), "stack3[2] should be 768");

        log.info("Multiple stack test passed");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStridedSliceEmptyDimension(Nd4jBackend backend) {
        // Test 1: strided_slice where begin==end on one dimension produces empty output
        // This previously caused out-of-bounds reads because _preprocess_strided_slice
        // skipped pushing index triplets for empty dimensions, but calcSubArrShapeInfoAndOffset
        // always reads rank*3 entries.
        INDArray in = Nd4j.rand(DataType.FLOAT, 3, 4, 5);
        INDArray begin = Nd4j.createFromArray(0, 2, 0);
        INDArray end = Nd4j.createFromArray(3, 2, 5);
        INDArray stride = Nd4j.createFromArray(1, 1, 1);

        DynamicCustomOp op = DynamicCustomOp.builder("strided_slice")
                .addInputs(in, begin, end, stride)
                .addIntegerArguments(0, 0, 0, 0, 0) // no masks
                .build();

        List<DataBuffer> shapes = Nd4j.getExecutioner().calculateOutputShape(op);
        assertEquals(1, shapes.size());
        long[] shape = Shape.shape(shapes.get(0).asLong());
        // dim 1 has begin==end so output should have 0 elements in that dim
        assertArrayEquals(new long[]{3, 0, 5}, shape,
                "strided_slice with begin==end on dim 1 should produce shape [3,0,5]");

        // Test 2: strided_slice on a 2D array with begin==end on dim 0
        INDArray in2 = Nd4j.rand(DataType.FLOAT, 5, 3);
        INDArray begin2 = Nd4j.createFromArray(2, 0);
        INDArray end2 = Nd4j.createFromArray(2, 3);
        INDArray stride2 = Nd4j.createFromArray(1, 1);

        DynamicCustomOp op2 = DynamicCustomOp.builder("strided_slice")
                .addInputs(in2, begin2, end2, stride2)
                .addIntegerArguments(0, 0, 0, 0, 0)
                .build();

        List<DataBuffer> shapes2 = Nd4j.getExecutioner().calculateOutputShape(op2);
        assertEquals(1, shapes2.size());
        long[] shape2 = Shape.shape(shapes2.get(0).asLong());
        assertArrayEquals(new long[]{0, 3}, shape2,
                "strided_slice with begin==end on dim 0 should produce shape [0,3]");
    }
}
