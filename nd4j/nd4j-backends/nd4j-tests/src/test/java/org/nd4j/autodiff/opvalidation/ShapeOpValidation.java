/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.autodiff.opvalidation;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.math3.linear.LUDecomposition;
import org.junit.Test;
import org.nd4j.OpValidationSuite;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.shape.*;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

@Slf4j
public class ShapeOpValidation extends BaseOpValidation {
    public ShapeOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    /*
    To test:
    tile
    reshape
    permute
    expandDims
    repeat
    rollAxis
    doRepeat
     */

    @Test
    public void testConcat() {
//        int[] concatDim = new int[]{0,0,0,1,1,1,2,2,2};
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
                orig[j] = Nd4j.rand(shapes.get(j));
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
                failed.add(name);
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testReshapeGradient() {
        int[] origShape = new int[]{3, 4, 5};

        List<String> failed = new ArrayList<>();

        for (int[] toShape : new int[][]{{3, 4 * 5}, {3 * 4, 5}, {1, 3 * 4 * 5}, {3 * 4 * 5, 1}}) {
            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, origShape)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable reshape = sd.reshape(in, toShape);
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", reshape, true);

                INDArray out = sd.execAndEndResult();
                INDArray expOut = in.getArr().std(true, Integer.MAX_VALUE);

                String msg = "toShape=" + Arrays.toString(toShape) + ", source=" + p.getSecond();
                TestCase tc = new TestCase(sd);
                tc.testName(msg)
                        .expectedOutput("out", expOut);

                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testPermuteGradient() {
        int[] origShape = new int[]{3, 4, 5};

        List<String> failed = new ArrayList<>();

        for (int[] perm : new int[][]{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}) {
            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, origShape)) {
                String msg = "permute=" + Arrays.toString(perm) + ", source=" + p.getSecond();
                System.out.println(msg);

                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable permute = sd.f().permute(in, perm);
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", permute, true);

                INDArray exp = inArr.permute(perm);
                INDArray expOut = in.getArr().std(true, Integer.MAX_VALUE);


                TestCase tc = new TestCase(sd);
                tc.testName(msg)
                        .expected("out", expOut)
                        .expected(permute, exp);

                String error = OpValidation.validate(tc, true);
                if(error != null){
                    failed.add(msg);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testRank(){

        List<long[]> inShape = Arrays.asList(null, new long[]{1}, new long[]{6}, new long[]{3,4}, new long[]{3,4,5});

        for( long[] shape : inShape){

            SameDiff sd = SameDiff.create();
            SDVariable var;
            if(shape == null){
                var = sd.var("in", Nd4j.trueScalar(1));
            } else {
                var = sd.var("in", Nd4j.create(shape));
            }

            SDVariable rank = sd.rank(var);

            INDArray expRank = Nd4j.trueScalar(shape == null ? 0 : shape.length);
            String msg = "Rank " + (shape == null ? 0 : shape.length);
            String err = OpValidation.validate(new TestCase(sd)
                    .expected(rank, expRank));

            assertNull(err);
        }
    }

    @Test
    public void testExpandDimsGradient() {
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

            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAllTestMatricesWithShape(origShape[0], origShape[1], 12345)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable expand = sd.f().expandDims(in, i);
                //Using stdev here: mean/sum would backprop the same gradient for each input...
                SDVariable stdev = sd.standardDeviation("out", expand, true);

                INDArray out = sd.execAndEndResult();
                INDArray expOut = in.getArr().std(true);

                assertArrayEquals(expExpandShape, expand.getArr().shape());
                INDArray expExpand = inArr.dup('c').reshape(expExpandShape);

                String msg = "expandDim=" + i + ", source=" + p.getSecond();
                log.info("Starting: " + msg);

                TestCase tc = new TestCase(sd);
                tc.testName(msg)
                        .expectedOutput("out", expOut)
                        .expectedOutput(expand.getVarName(), expExpand);

                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(error);
                }
            }
        }
        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testSqueezeGradient() {
        val origShape = new long[]{3, 4, 5};

        List<String> failed = new ArrayList<>();

        for (int i = 0; i < 3; i++) {

            val shape = origShape.clone();
            shape[i] = 1;

            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, shape)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable squeeze = sd.f().squeeze(in, i);
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

                sd.execAndEndResult();

                INDArray squeezed = squeeze.getArr();
//                assertArrayEquals(expShapePostSqueeze, squeezed.shape());

                INDArray out = sd.execAndEndResult();
                INDArray expOut = in.getArr().std(true, Integer.MAX_VALUE);
                assertEquals(expOut, out);

                String msg = "squeezeDim=" + i + ", source=" + p.getSecond();
                TestCase tc = new TestCase(sd)
                        .testName(msg)
                        .expected(squeeze.getVarName(), exp)
                        .expectedOutput("out", expOut);


                String error = OpValidation.validate(tc, true);
                if(error != null){
                    failed.add(name);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testSliceGradient() {
        Nd4j.getRandom().setSeed(12345);

        //Order here: original shape, begin, size
        List<Triple<int[], int[], int[]>> testCases = new ArrayList<>();
        testCases.add(new Triple<>(new int[]{3, 4}, new int[]{0, 0}, new int[]{3, 4}));
        testCases.add(new Triple<>(new int[]{3, 4}, new int[]{1, 1}, new int[]{2, 2}));
        testCases.add(new Triple<>(new int[]{3, 4}, new int[]{1, 2}, new int[]{2, 2}));
        testCases.add(new Triple<>(new int[]{3, 4, 5}, new int[]{0, 0, 0}, new int[]{3, 4, 5}));
        testCases.add(new Triple<>(new int[]{3, 4, 5}, new int[]{1, 1, 1}, new int[]{2, 3, 4}));
        testCases.add(new Triple<>(new int[]{3, 4, 5}, new int[]{1, 0, 2}, new int[]{2, 3, 2}));

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
            INDArray arr = Nd4j.rand(os);

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

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Builder(builderClassName = "Builder")
    @Data
    private static class SSCase {
        private int[] shape;
        private int[] begin;
        private int[] end;
        private int[] strides;
        private int beginMask;
        private int endMask;
        private int ellipsisMask;
        private int newAxisMask;
        private int shrinkAxisMask;

        public static class Builder {

            public Builder shape(int... shape) {
                this.shape = shape;
                return this;
            }

            public Builder begin(int... begin) {
                this.begin = begin;
                return this;
            }

            public Builder end(int... end) {
                this.end = end;
                return this;
            }

            public Builder strides(int... strides) {
                this.strides = strides;
                return this;
            }
        }
    }

    @Test
    public void testStridedSliceGradient() {
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
        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testMerge() {
        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int t = 0; t < 3; t++) {
            for (int numArrays : new int[]{3, 1}) {
                for (long[] shape : new long[][]{{1}, {3, 4}, {3, 4, 5}}) {


                    SameDiff sd = SameDiff.create();
                    SDVariable[] arr = new SDVariable[numArrays];

                    for (int i = 0; i < numArrays; i++) {
                        arr[i] = sd.var(String.valueOf(i), Nd4j.rand(shape));
                    }

                    INDArray exp = arr[0].getArr().dup();
                    SDVariable merge;
                    String name;
                    switch (t) {
                        case 0:
                            name = "mergeAdd";
                            merge = sd.mergeAdd(arr);
                            for( int i=1; i<numArrays; i++ ){
                                exp.addi(arr[i].getArr().dup());
                            }
                            break;
                        case 1:
                            name = "mergeMax";
                            merge = sd.mergeMax(arr);
                            for( int i=1; i<numArrays; i++ ){
                                exp = Transforms.max(exp, arr[i].getArr(), true);
                            }
                            break;
                        case 2:
                            name = "mergeAvg";
                            merge = sd.mergeAvg(arr);
                            for( int i=1; i<numArrays; i++ ){
                                exp.addi(arr[i].getArr().dup());
                            }
                            exp.divi(numArrays);
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
                        failed.add(msg);
                    }
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testStack() {
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
                    if(Arrays.equals(new long[]{3,4}, shape)){
                        if(axis == 0){
                            INDArray out = Nd4j.create(numInputs, 3, 4);
                            for( int i=0; i<numInputs; i++ ){
                                out.get(point(i), all(), all()).assign(inArr[i]);
                            }
                            expStack = out;
                        } else if(axis == 1) {
                            INDArray out = Nd4j.create(3, numInputs, 4);
                            for( int i=0; i<numInputs; i++ ){
                                out.get(all(), point(i), all()).assign(inArr[i]);
                            }
                            expStack = out;
                        } else {
                            INDArray out = Nd4j.create(3, 4, numInputs);
                            for( int i=0; i<numInputs; i++ ){
                                out.get(all(), all(), point(i)).assign(inArr[i]);
                            }
                            expStack = out;
                        }
                    }

                    SDVariable stack = sd.stack(axis, in);

                    INDArray out = sd.execAndEndResult();
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
                        failed.add(name);
                    }
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testUnStack() {
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
                    SDVariable merged = sd.mergeAvg(unstacked);

                    if (ArrayUtil.prodLong(stackedShape) == 1 || ArrayUtil.prodLong(shape) == 1) {
                        SDVariable loss = sd.sum("loss", merged);
                    } else {
                        SDVariable loss = sd.standardDeviation("loss", merged, true);
                    }

                    String msg = "Unstacked shape = " + Arrays.toString(shape) + ", stacked shape = " + Arrays.toString(stackedShape)
                            + ", axis=" + axis + ", numInputs=" + numInputs;

                    sd.execAndEndResult();
                    for (SDVariable v : unstacked) {
                        assertArrayEquals(msg, shape, v.getArr().shape());
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

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testTile() {
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
        exp[0] = Nd4j.trueVector(new double[]{3});
        exp[1] = Nd4j.trueVector(new double[]{3,3,3,3,3});
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

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testTileBp(){
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

    @Test
    public void testTileBp2(){
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


    @Test
    public void testReshape() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(-5, 6, 12)).reshape(3, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result1 = sameDiff.reshape(x, 4, 3);
        SDVariable loss = sameDiff.standardDeviation(result1, true);

        INDArray exp = arr.dup('c').reshape('c', 4,3);

        String err = OpValidation.validate(new TestCase(sameDiff)
                .expectedOutput(result1.getVarName(), exp));

        assertNull(err);
    }

    @Test
    public void testReshape2() {
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


    @Test
    public void testTranspose() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.transpose(x);
        SDVariable loss = sameDiff.standardDeviation(result, true);

        String err = OpValidation.validate(new TestCase(sameDiff).expectedOutput(result.getVarName(), arr.transpose()));
        assertNull(err);
    }

    @Test
    public void testTransposeOp(){

        INDArray arr = Nd4j.linspace(1,15, 15).reshape(5,3);
        INDArray out = Nd4j.create(3,5);

        OpTestCase op = new OpTestCase(new Transpose(arr, out));
        INDArray exp = arr.transpose();
        op.expectedOutput(0, exp);
        String err = OpValidation.validate(op);
        assertNull(err);
    }

    @Test
    public void testShape() {
        SameDiff sameDiff = SameDiff.create();
        val shape = new long[]{2, 3};
        SDVariable x = sameDiff.var("x", shape);
        SDVariable result = sameDiff.shape(x);
        SDVariable loss = sameDiff.standardDeviation(result, true);

        System.out.println(sameDiff.summary());

        String err = OpValidation.validate(new TestCase(sameDiff)
                .expected(result, Nd4j.create(new double[]{2,3}, new long[]{2})));

        assertNull(err);
    }

    @Test
    public void testSize() {
        SameDiff sameDiff = SameDiff.create();
        val shape = new long[]{2, 3};
        SDVariable x = sameDiff.var("x", shape);
        SDVariable result = sameDiff.size(x);

        String err = OpValidation.validate(new TestCase(sameDiff)
                .expected(result, Nd4j.trueScalar(6)));

        assertNull(err);
    }

    @Test
    public void testDiagShapeFn() {
        INDArray i = Nd4j.linspace(1, 16, 16).reshape(4,4);

        OpTestCase op = new OpTestCase(new DiagPart(i, null));

        INDArray exp = Nd4j.create(new double[]{1,6,11,16}, new long[]{4});
        op.expectedOutput(0, exp);

        String err = OpValidation.validate(op);
        assertNull(err);
    }


    @Test
    public void testPermute(){
        INDArray in = Nd4j.linspace(1, 60, 60).reshape(3,4,5);
        INDArray exp = in.permute(0,1,2);   //No op

        assertEquals(in, exp);

        INDArray out = Nd4j.create(3,4,5);
        OpTestCase op = new OpTestCase(new Permute(in,out,0,1,2));
        op.expectedOutput(0, exp);

        assertNull(OpValidation.validate(op));
    }

    @Test
    public void testPermute2(){
        for (int[] perm : new int[][]{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}) {
            INDArray in = Nd4j.linspace(1, 60, 60).reshape(3,4,5);
            INDArray exp = in.permute(perm);

            int[] outShape = new int[3];
            for( int i=0; i<3; i++ ){
                outShape[i] = (int)in.size(perm[i]);
            }

            //System.out.println(Arrays.toString(outShape) + " - permute " + Arrays.toString(perm));
            INDArray out = Nd4j.create(outShape);
            OpTestCase op = new OpTestCase(new Permute(in, out, perm));
            op.expectedOutput(0, exp);

            assertNull(OpValidation.validate(op));
        }
    }

    @Test
    public void testConstant(){
        OpValidationSuite.ignoreFailing();

        //Case 0: no shape
        SameDiff sd = SameDiff.create();
        INDArray ia = Nd4j.trueVector(new double[]{1,2,3});
        SDVariable in = sd.var(ia);
        SDVariable constant = sd.constant(in, 3);
        SDVariable loss = constant.std(true);

        assertNull(OpValidation.validate(new TestCase(sd).expected(constant, ia)));

        //Case 1: shape is provided + scalar

        sd = SameDiff.create();
        ia = Nd4j.trueScalar(3.0);
        in = sd.var(ia);
        constant = sd.constant(in, 3,4,5);
        INDArray exp = Nd4j.valueArrayOf(new long[]{3,4,5}, 3.0);
        loss = constant.std(true);

        assertNull(OpValidation.validate(new TestCase(sd).expected(constant, ia)));
    }


    @Test
    public void testUnstackEdgeCase2(){
        for( int i=0; i<3; i++ ) {

            INDArray arr = Nd4j.rand(new long[]{1, 1, 1});

            List<long[]> shapes = Nd4j.getExecutioner().calculateOutputShape(
                    new Unstack(arr, null, i));

            assertEquals(1, shapes.size());
            assertArrayEquals(new long[]{1, 1}, shapes.get(0));
        }
    }

    @Test
    public void invertPermutation() {
        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new float[] {3, 4, 0, 2, 1});
        INDArray expOut = Nd4j.create(new float[] {2, 4, 3, 0, 1});

        SDVariable input = sd.var("in", new int[] {1, 5});
        sd.associateArrayWithVariable(ia, input);
        SDVariable out = sd.invertPermutation(input);

        assertNull(OpValidation.validate(new TestCase(sd)
                .gradientCheck(false)   //Integer indices in/out
                .expected(out, expOut)));
    }


    @Test
    public void testGatherNd(){

        List<INDArray> indices = new ArrayList<>();
        List<INDArray> params = new ArrayList<>();
        List<INDArray> expected = new ArrayList<>();


        indices.add(Nd4j.create(new double[][]{{0,0},{1,1}}));
        params.add(Nd4j.create(new double[][]{{1,2},{3,4}}));
        expected.add(Nd4j.trueVector(new double[]{1,4}));

        indices.add(Nd4j.create(new double[][]{{1},{0}}));
        params.add(Nd4j.create(new double[][]{{1,2},{3,4}}));
        expected.add(Nd4j.create(new double[][]{{3,4},{1,2}}));

        indices.add(Nd4j.create(new double[][]{{0,1},{1,0}}));
        params.add(Nd4j.create(new double[][][]{{{10,20},{30,40}},
                {{11, 21}, {31,41}}}));
        expected.add(Nd4j.create(new double[][]{{30,40},{11,21}}));

        for( int i=0; i<indices.size(); i++ ){
            SameDiff sd = SameDiff.create();
            SDVariable p = sd.var("p", params.get(i));
            SDVariable ind = sd.var("i", indices.get(i));
            SDVariable g = sd.gatherNd(p, ind);

            INDArray exp = expected.get(i);
            //INDArray act = sd.execAndEndResult();

            String err = OpValidation.validate(new TestCase(sd)
                    .expected(g, exp)
                    .gradientCheck(false)); //Grad not implemented
            assertNull(err);
        }
    }


    @Test
    public void testReverseSequence() {
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
        INDArray arr1 = Nd4j.create(input_data, new long[]{2, 5, 3});
        INDArray arr2 = Nd4j.create(new float[]{3, 2}).reshape(2);
        SDVariable x = sameDiff.var("x", arr1);
        SDVariable seq_lengths = sameDiff.var("seq_lengths", arr2);
        SDVariable result = sameDiff.reverseSequence(x, seq_lengths, 1, 0);
        INDArray expected = Nd4j.create(expected_output, new long[]{2, 5, 3});
        assertArrayEquals(arr1.shape(), result.eval().shape());
        assertEquals(expected, result.eval());

        SDVariable loss = sameDiff.standardDeviation(result, true);
        String err = OpValidation.validate(new TestCase(sameDiff)
                .expected(result.getVarName(), expected)
                .gradCheckSkipVariables(seq_lengths.getVarName()));
        assertNull(err);
    }


    @Test
    public void testMatrixDeterminant(){
        OpValidationSuite.ignoreFailing();  //Gradient check failing

        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(3,3);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", in);
        SDVariable md = sd.f().matrixDeterminant(var);

        double d = new LUDecomposition(CheckUtil.convertToApacheMatrix(in)).getDeterminant();


        INDArray outExp = Nd4j.trueScalar(d);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(md.getVarName(), outExp));
        assertNull(err);
    }

    @Test
    public void testDeterminant22(){
        OpValidationSuite.ignoreFailing();  //Gradient check failing

        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.create(new double[][]{{1, 2.5}, {3.5, 4.5}});


        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", in);
        SDVariable md = sd.f().matrixDeterminant(var);

        double d = new LUDecomposition(CheckUtil.convertToApacheMatrix(in)).getDeterminant();
        double d2 = in.getDouble(0,0) * in.getDouble(1,1) - in.getDouble(1,0) * in.getDouble(0,1);
        assertEquals(d, d2, 1e-5);


        INDArray outExp = Nd4j.trueScalar(d);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(md.getVarName(), outExp));
        assertNull(err);
    }

    @Test
    public void testMatrixDeterminant3(){
        OpValidationSuite.ignoreFailing();  //Gradient checks failing
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(3,3);
        //System.out.println(in.shapeInfoToString());   //Rank: 2,Offset: 0 Order: c Shape: [3,3],  stride: [3,1]
        //System.out.println(Arrays.toString(in.data().asFloat())); //[0.27620894, 0.21801452, 0.062078513, 7.348895E-4, 0.24149609, 0.4948205, 0.93483436, 0.52035654, 0.30292067]

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", in);
        SDVariable md = sd.f().matrixDeterminant(var);

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

        INDArray outExp = Nd4j.trueScalar(d);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(md.getVarName(), outExp));
        assertNull(err);
    }

    @Test
    public void testMatrixDeterminant4(){
        OpValidationSuite.ignoreFailing();  //Gradient checks failing
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(4,4);
        //System.out.println(in.shapeInfoToString());   //Rank: 2,Offset: 0 Order: c Shape: [4,4],  stride: [4,1]
        System.out.println(Arrays.toString(in.data().asFloat())); //[0.27620894, 0.21801452, 0.062078513, 7.348895E-4, 0.24149609, 0.4948205, 0.93483436, 0.52035654, 0.30292067, 0.3289706, 0.7977864, 0.03180518, 0.1455722, 0.90352905, 0.9405744, 0.0048329555]

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", in);
        SDVariable md = sd.f().matrixDeterminant(var);

        double d = new LUDecomposition(CheckUtil.convertToApacheMatrix(in)).getDeterminant();   //-0.06713878100086641
        //System.out.println(d);

        String err = OpValidation.validate(new TestCase(sd)
                .expected(md.getVarName(), Nd4j.trueScalar(d)));
        assertNull(err);
    }

    @Test
    public void testSegmentOps(){
        INDArray s = Nd4j.create(new double[]{0,0,0,1,2,2,3,3}, new long[]{8});
        INDArray d = Nd4j.create(new double[]{5,1,7,2,3,4,1,3}, new long[]{8});
        int numSegments = 4;

        List<String> failed = new ArrayList<>();

        for(String op : new String[]{"max", "min", "mean", "prod", "sum",
                "umax", "umin", "umean", "uprod", "usum", "usqrtn"}) {
            log.info("Starting test: {}", op);

            if(op.startsWith("u")){
                //Unsorted segment cases
                s = Nd4j.create(new double[]{3,1,0,0,2,0,3,2}, new long[]{8});
                d = Nd4j.create(new double[]{1,2,5,7,3,1,3,4}, new long[]{8});
            }

            SameDiff sd = SameDiff.create();
            SDVariable data = sd.var("data", d);
            SDVariable segments = sd.var("segments", s);

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
                    exp = Nd4j.trueVector(new double[]{(5+7+1)/Math.sqrt(3), 2, (3+4)/Math.sqrt(2), (1+3)/Math.sqrt(2)});
                    break;
                default:
                    throw new RuntimeException();
            }

            SDVariable loss = sm.std(false);

            TestCase tc = new TestCase(sd)
                    .testName(op)
                    .expected(sm, exp)
                    .gradientCheck(true)
                    .gradCheckSkipVariables(segments.getVarName());

            String err = OpValidation.validate(tc);
            if(err != null)
                failed.add(err);
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testSequenceMask() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.create(new float[] {1, 3, 2}).reshape(3);
        SDVariable lengths = sameDiff.var("lengths", arr);

        // Test with static max len
        int maxlen = 5;
        INDArray expected = Nd4j.create(new float[] {1, 0, 0, 0, 0,
                        1, 1, 1, 0, 0,
                        1, 1, 0, 0, 0},
                new long[]{3, 5});
        SDVariable result1 = sameDiff.sequenceMask(lengths, maxlen);
        assertArrayEquals(expected.shape(), result1.eval().shape());
        assertEquals(expected, result1.eval());

        SDVariable loss = sameDiff.standardDeviation(result1, true);

        String err = OpValidation.validate(new TestCase(sameDiff)
                .expected(result1, expected)
                .gradCheckSkipVariables(lengths.getVarName()));
        assertNull(err);

        // Test with dynamic maxlen
        lengths = sameDiff.var("lengths2", arr); // required because of an internal samediff bug
        SDVariable maxLen = sameDiff.var("maxLen", Nd4j.create(new float[]{5}).reshape(1));
        SDVariable result2 = sameDiff.sequenceMask(lengths, maxLen);
        assertArrayEquals(expected.shape(), result2.eval().shape());
        assertEquals(expected, result2.eval());
    }

    @Test
    public void testMeshGrid(){
        List<String> failed = new ArrayList<>();

        for( int rank=2; rank<=4; rank++ ){
            SameDiff sd = SameDiff.create();

            SDVariable[] arr = new SDVariable[rank];
            List<String> names = new ArrayList<>();
            for( int i=0; i<rank; i++ ){
                INDArray in = Nd4j.linspace(1,3+i, 3+i).reshape(3+i);
                arr[i] = sd.var("in"+i, in);
                names.add("meshgrid-" + i);
            }

            SDVariable[] meshgrid = sd.meshgrid(names, false, arr);

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
                exp[i] = Nd4j.create(shape);
                long nTensors = exp[i].tensorssAlongDimension(i);
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

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testGather(){
        List<INDArray> inArrs = new ArrayList<>();
        List<Integer> axis = new ArrayList<>();
        List<INDArray> indices = new ArrayList<>();

        inArrs.add(Nd4j.linspace(1,48,48).reshape(2,4,3,2));
        indices.add(Nd4j.trueVector(new double[]{1,0}));
        axis.add(-2);

        for( int i=0; i<inArrs.size(); i++ ){

            INDArray in = inArrs.get(i);
            INDArray idx = indices.get(i);
            int a = axis.get(i);
            int aNorm = (a >= 0 ? a : a + in.rank());

            INDArray expOut;
            if(idx.rank() == 0){
                INDArrayIndex[] get = new INDArrayIndex[in.rank()];
                for( int j=0; j<aNorm; j++ ){
                    get[j] = NDArrayIndex.all();
                }
                get[aNorm] = NDArrayIndex.point(idx.getInt(0));
                for( int j=aNorm+1; j<in.rank(); j++ ){
                    get[j] = NDArrayIndex.all();
                }
                expOut = in.get(get);
            } else if (idx.rank() == 1){
                long[] shape = in.shape().clone();
                shape[aNorm] = idx.length();
                expOut = Nd4j.create(shape);

                INDArrayIndex[] get = new INDArrayIndex[in.rank()];
                INDArrayIndex[] put = new INDArrayIndex[in.rank()];
                for( int j=0; j<aNorm; j++ ){
                    get[j] = NDArrayIndex.all();
                    put[j] = NDArrayIndex.all();
                }
                for( int j=aNorm+1; j<in.rank(); j++ ){
                    get[j] = NDArrayIndex.all();
                    put[j] = NDArrayIndex.all();
                }

                for(int j=0; j<idx.length(); j++ ){
                    get[aNorm] = NDArrayIndex.point(idx.getInt(j));
                    put[aNorm] = NDArrayIndex.point(j);
                    expOut.put(put, in.get(get));
                }
            } else {
                throw new RuntimeException("Rank 2+ tests not yet implemented");
            }


            SameDiff sd = SameDiff.create();
            SDVariable sdIn = sd.var("in", in);
            SDVariable sdIdx = sd.var("idx", idx);
            SDVariable gather = sd.gather(sdIn, sdIdx, a);

            SDVariable loss = gather.std(true);

            String err = OpValidation.validate(new TestCase(sd)
                    .expected(gather, expOut)
                    .gradCheckSkipVariables("idx"));

            assertNull(err);
        }
    }

    //TODO UPDATE TO OPVALIDATION
    @Test
    public void testGatherSimple() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.create(new float[]{1, 2, 3, 4}, new long[]{2, 2});
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.gather(x, new int[]{1, 0}, 1);
        INDArray expected = Nd4j.create(new float[]{2, 1, 4, 3}, new long[]{2, 2});
        assertEquals(expected, result.eval());
    }



    //TODO UPDATE TO OPVALIDATION
    @Test
    public void testGatherNdSingle() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 24, 24)).reshape(2, 3, 4);
        INDArray arr2 = Nd4j.create(new float[]{1, 2, 3, 0, 1, 3, 1, 0, 2}, new long[]{3, 3});
        SDVariable x = sameDiff.var("x", arr1);
        SDVariable idxs = sameDiff.var("idxs", arr2);
        SDVariable result = sameDiff.gatherNd(x, idxs);
        // build expected output array
        INDArray expected  = Nd4j.zeros(3);
        for (int i=0; i<3; i++){
            INDArray idx = arr2.get(point(i));
            expected.get(point(i)).assign(
                    arr1.get(point(idx.getInt(0)),
                            point(idx.getInt(1)),
                            point(idx.getInt(2))));
        }
        assertEquals(expected, result.eval());
    }

    //TODO UPDATE TO OPVALIDATION
    @Test
    public void testStack2() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 6, 6)).reshape(3, 2);
        INDArray arr2 = Transforms.sigmoid(Nd4j.linspace(7, 12, 6)).reshape(3, 2);
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable result = sameDiff.stack(1, x1, x2);
        assertArrayEquals(new long[]{3, 2, 2}, result.eval().shape());
    }

    //TODO UPDATE TO OPVALIDATION
    @Test
    public void testParallelStack() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 6, 6)).reshape(3, 2);
        INDArray arr2 = Transforms.sigmoid(Nd4j.linspace(7, 12, 6)).reshape(3, 2);
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable result = sameDiff.parallel_stack(new SDVariable[]{x1, x2});
        assertArrayEquals(new long[]{2, 3, 2}, result.eval().shape());
        assertEquals(Nd4j.concat(0, arr1, arr2).reshape(2, 3, 2), result.eval());
    }

    //TODO UPDATE TO OPVALIDATION
    @Test
    public void testUnStack2() {
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

    //TODO UPDATE TO OPVALIDATION
    @Test
    public void testPermuteSimple() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 6, 6).reshape(2, 3));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.permute(x, 1, 0);
        assertArrayEquals(new long[]{3, 2}, result.getShape());

    }

    //TODO UPDATE TO OPVALIDATION
    @Test
    public void testConcat2() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        INDArray arr2 = Transforms.sigmoid(Nd4j.linspace(4, 8, 4));
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable result = sameDiff.concat(0, x1, x2);
        assertArrayEquals(new long[]{2, 4}, result.eval().shape());

    }

    //TODO UPDATE TO OPVALIDATION
    @Test
    public void testTile2() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.tile(x, new int[]{2, 2});
        assertArrayEquals(new long[]{2, 8}, result.eval().shape());
        INDArray arr2 = Nd4j.concat(0, arr, arr);  // (1, 4), (1, 4) -> (2, 4)
        INDArray expected = Nd4j.concat(1, arr2, arr2);  // (2, 4), (2, 4) -> (2, 8)
        assertEquals(expected, result.eval());
    }

    @Test
    public void testBroadcast() {
        OpValidationSuite.ignoreFailing();
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.rand(3, 4));
        SDVariable broadcast = sd.f().broadcast(in, 3, 4, 5);

        INDArray out = sd.execAndEndResult();
        assertArrayEquals(new long[]{3, 4, 5}, out.shape());

        for (int i = 0; i < 5; i++) {
            assertEquals(in.getArr(), out.get(all(), all(), point(i)));
        }
    }


    @Test
    public void testSlice2d() {
        INDArray inArr = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice_full = sd.slice(in, new int[]{0, 0}, new int[]{3, 4});
        SDVariable subPart = sd.slice(in, new int[]{1, 2}, new int[]{2, 2});

        sd.execAndEndResult();

        assertEquals(inArr, slice_full.getArr());
        assertEquals(inArr.get(interval(1, 3), interval(2, 4)), subPart.getArr());
    }


    @Test
    public void testSlice3d() {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice_full = sd.slice(in, new int[]{0, 0, 0}, new int[]{3, 4, 5});
        SDVariable subPart = sd.slice(in, new int[]{1, 2, 3}, new int[]{2, 2, 1});

        sd.execAndEndResult();

        assertEquals(inArr, slice_full.getArr());
        assertEquals(inArr.get(interval(1, 3), interval(2, 4), interval(3, 4)), subPart.getArr());
    }

    @Test
    public void testStridedSlice2dBasic() {
        INDArray inArr = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice_full = sd.stridedSlice(in, new int[]{0, 0}, new int[]{3, 4}, new int[]{1, 1});
        SDVariable subPart = sd.stridedSlice(in, new int[]{1, 2}, new int[]{3, 4}, new int[]{1, 1});
        // SDVariable subPart2 = sd.stridedSlice(in, new int[]{0, 0}, new int[]{4, 5}, new int[]{2, 2});

        sd.execAndEndResult();

        assertEquals(inArr, slice_full.getArr());
        assertEquals(inArr.get(interval(1, 3), interval(2, 4)), subPart.getArr());
        // assertEquals(inArr.get(interval(0, 2, 4), interval(0, 2, 5)), subPart2.getArr());
    }


    @Test
    public void testStridedSliceBeginEndMask() {
        INDArray inArr = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice1 = sd.stridedSlice(in, new int[]{-999, 0}, new int[]{2, 4}, new int[]{1, 1}, 1 << 1, 0, 0, 0, 0);
        SDVariable slice2 = sd.stridedSlice(in, new int[]{1, 0}, new int[]{-999, 4}, new int[]{1, 1}, 0, 1, 0, 0, 0);

        sd.execAndEndResult();

        assertEquals(inArr.get(NDArrayIndex.interval(0, 2), NDArrayIndex.all()), slice1.getArr());
        assertEquals(inArr.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all()), slice2.getArr());
    }

    @Test
    public void testStridedSliceEllipsisMask() {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);

        //[1:3,...] -> [1:3,:,:]
        SDVariable slice = sd.stridedSlice(in, new int[]{1}, new int[]{3}, new int[]{1}, 0, 0, 1 << 1, 0, 0);
        //[1:3,...,1:4] -> [1:3,:,1:4]
        SDVariable slice2 = sd.stridedSlice(in, new int[]{1, 1}, new int[]{3, 4}, new int[]{1, 1}, 0, 0, 1 << 1, 0, 0);

        sd.execAndEndResult();

        assertEquals(inArr.get(interval(1, 3), all(), all()), slice.getArr());
        assertEquals(inArr.get(interval(1, 3), all(), all()), slice2.getArr());
    }

    @Test
    public void testStridedSliceNewAxisMask() {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice = sd.stridedSlice(in, new int[]{-999, 0, 0, 0}, new int[]{-999, 3, 4, 5}, new int[]{-999, 1, 1, 1}, 0, 0, 0, 1, 0);

        INDArray out = sd.execAndEndResult();

        assertArrayEquals(new long[]{1, 3, 4, 5}, out.shape());
        assertEquals(inArr, out.get(point(0), all(), all(), all()));
    }

    @Test
    public void testStridedSliceNewAxisMask2() {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice = sd.stridedSlice(in, new int[]{1, 1, -999, 1}, new int[]{3, 3, -999, 4}, new int[]{1, 1, -999, 1}, 0, 0, 0, 1 << 2, 0);
        INDArray out = sd.execAndEndResult();

        assertArrayEquals(new long[]{2, 2, 1, 3}, slice.getArr().shape());
    }

    @Test
    public void testStridedSliceShrinkAxisMask() {

        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice = sd.stridedSlice(in, new int[]{0, 0, 0}, new int[]{-999, 4, 5}, new int[]{1, 1, 1}, 0, 0, 0, 0, 1);
        SDVariable slice2 = sd.stridedSlice(in, new int[]{2, 0, 0}, new int[]{-999, 4, 5}, new int[]{1, 1, 1}, 0, 0, 0, 0, 1);
        SDVariable slice3 = sd.stridedSlice(in, new int[]{1, 2, 1}, new int[]{-999, -999, 5}, new int[]{1, 1, 1}, 0, 0, 0, 0, 1 | 1 << 1);

        sd.execAndEndResult();

        assertEquals(inArr.get(point(0), all(), all()), slice.getArr());
        assertEquals(inArr.get(point(2), all(), all()), slice2.getArr());
        assertEquals(inArr.get(point(1), point(2), interval(1, 5)), slice3.getArr());
    }

    @Test
    public void testSizeAt_1() throws Exception {
        val array = Nd4j.create(10, 20, 30);
        val exp = Nd4j.trueScalar(20);

        val op = new SizeAt(array, 1);

        Nd4j.getExecutioner().exec(op);

        val output = op.outputArguments()[0];

        assertEquals(exp, output);
    }

    @Test
    public void testEye(){
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

    @Test
    public void testSplit1(){
        INDArray in = Nd4j.linspace(1,10,10).reshape(10);
        INDArray axis = Nd4j.trueScalar(-1);

        INDArray out1 = Nd4j.create(new long[]{5});
        INDArray out2 = Nd4j.create(new long[]{5});

        INDArray exp1 = in.get(NDArrayIndex.interval(0,5)).reshape(5);
        INDArray exp2 = in.get(NDArrayIndex.interval(5,10)).reshape(5);

        assertNull(OpValidation.validate(new OpTestCase(DynamicCustomOp.builder("split")
                .addInputs(axis, in)
                .addOutputs(out1, out2)
                .addIntegerArguments(2)
                .build()).expectedOutput(0, exp1).expectedOutput(1,exp2)));
    }

    @Test
    public void testSplit2(){
        INDArray in = Nd4j.linspace(1,24,24).reshape(3,8);
        INDArray axis = Nd4j.trueScalar(-1);

        INDArray out1 = Nd4j.create(new long[]{3,4});
        INDArray out2 = Nd4j.create(new long[]{3,4});

        INDArray exp1 = in.get(NDArrayIndex.all(), NDArrayIndex.interval(0,4));
        INDArray exp2 = in.get(NDArrayIndex.all(), NDArrayIndex.interval(4,8));

        assertNull(OpValidation.validate(new OpTestCase(DynamicCustomOp.builder("split")
                .addInputs(axis, in)
                .addOutputs(out1, out2)
                .addIntegerArguments(2)
                .build()).expectedOutput(0, exp1).expectedOutput(1,exp2)));
    }
}
