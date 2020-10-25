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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.OpValidationSuite;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.enums.DataFormat;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.enums.PadMode;
import org.nd4j.enums.ImageResizeMethod;
import org.nd4j.enums.PartitionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.image.ImageResize;
import org.nd4j.linalg.api.ops.impl.layers.convolution.DepthToSpace;
import org.nd4j.linalg.api.ops.impl.layers.convolution.SpaceToDepth;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling3d;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarFMod;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication;
import org.nd4j.linalg.api.ops.impl.shape.Cross;
import org.nd4j.linalg.api.ops.impl.shape.MergeAvg;
import org.nd4j.linalg.api.ops.impl.shape.MergeMax;
import org.nd4j.linalg.api.ops.impl.shape.MergeMaxIndex;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.EmbeddingLookup;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByAvgNorm;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CReLU;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LessThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Max;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Min;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Reverse;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Standardize;
import org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MergeAddOp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ACosh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ASinh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Erf;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Erfc;
import org.nd4j.linalg.api.ops.impl.transforms.strict.HardSigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.strict.LogSigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.strict.RationalTanh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.RectifiedTanh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Swish;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.function.Function;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.*;

import static org.junit.Assert.*;

@Slf4j
public class TransformOpValidation extends BaseOpValidation {

    private DataType initialType;

    public TransformOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Before
    public void before() {
        Nd4j.create(1);
        initialType = Nd4j.dataType();

        Nd4j.setDataType(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @After
    public void after() {
        Nd4j.setDataType(initialType);
    }


    @After
    public void tearDown() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    @Test
    public void testScalarOps() {
        int d0 = 2;
        int d1 = 3;
        int d2 = 4;

        int n = d0 * d1 * d2;

        List<String> failed = new ArrayList<>();

        for (int i = 0; i < 11; i++) {
            for (char inOrder : new char[]{'c', 'f'}) {
                SameDiff sd = SameDiff.create();

                INDArray inArr = Nd4j.linspace(1, n, n, DataType.DOUBLE).reshape('c', d0, d1, d2).dup(inOrder);
                SDVariable in = sd.var("in", inArr);
                TestCase tc = new TestCase(sd).gradientCheck(true);

                SDVariable out;
                String msg;
                switch (i) {
                    case 0:
                        out = in.mul(2);
                        tc.expectedOutput(out.name(), inArr.mul(2));
                        msg = "mul - " + inOrder;
                        break;
                    case 1:
                        out = in.div(2);
                        tc.expectedOutput(out.name(), inArr.div(2));
                        msg = "div - " + inOrder;
                        break;
                    case 2:
                        out = in.add(2);
                        tc.expectedOutput(out.name(), inArr.add(2));
                        msg = "add - " + inOrder;
                        break;
                    case 3:
                        out = in.sub(2);
                        tc.expectedOutput(out.name(), inArr.sub(2));
                        msg = "sub - " + inOrder;
                        break;
                    case 4:
                        out = in.rdiv(2);
                        tc.expectedOutput(out.name(), inArr.rdiv(2));
                        msg = "rdiv - " + inOrder;
                        break;
                    case 5:
                        out = in.rsub(2);
                        tc.expectedOutput(out.name(), inArr.rsub(2));
                        msg = "rsub - " + inOrder;
                        break;
                    case 6:
                        out = sd.math().pow(in, 2);
                        tc.expectedOutput(out.name(), Transforms.pow(inArr, 2));
                        msg = "pow - " + inOrder;
                        break;
                    case 7:
                        inArr.assign(Nd4j.rand(inArr.dataType(), inArr.shape()).muli(5).subi(2.5));
                        out = sd.math().floorMod(in, 2.0);
                        tc.expected(out, Nd4j.getExecutioner().exec(new ScalarFMod(inArr.dup(), 2.0)));
                        msg = "scalarFloorMod - " + inOrder;
                        break;
                    case 8:
                        inArr.assign(Nd4j.rand(inArr.shape()));
                        out = sd.scalarMax(in, 0.5);
                        tc.expected(out, Transforms.max(inArr.dup(), 0.5));
                        msg = "scalarMax - " + inOrder;
                        break;
                    case 9:
                        inArr.assign(Nd4j.rand(inArr.shape()));
                        out = sd.scalarMin(in, 0.5);
                        tc.expected(out, Transforms.min(inArr.dup(), 0.5));
                        msg = "scalarMin - " + inOrder;
                        break;
                    case 10:
                        out = in.assign(0.5);
                        tc.expected(out, Nd4j.valueArrayOf(inArr.shape(), 0.5));
                        msg = "scalarSet - " + inOrder;
                        break;
                    default:
                        throw new RuntimeException();
                }

                tc.testName(msg);

                SDVariable loss = sd.standardDeviation(out, true);

                log.info("Starting test: " + msg);
                String err = OpValidation.validate(tc, true);
                if (err != null) {
                    failed.add(err);
                }
            }
        }
        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testScalarMulCF() {

        INDArray in = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape('c', 3, 4);
        INDArray outC = Nd4j.createUninitialized(3, 4);
        INDArray outF = Nd4j.createUninitialized(3, 4);

        Nd4j.getExecutioner().exec(new ScalarMultiplication(in, null, outC, 2.0));
        Nd4j.getExecutioner().exec(new ScalarMultiplication(in, null, outF, 2.0));

        assertEquals(outC, outF);
    }


    @Test
    public void testScalarMulCF2() {

        INDArray in = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape('c', 3, 4);

        INDArray outC = Nd4j.getExecutioner().exec(new ScalarMultiplication(in.dup('c'), 2.0));
        INDArray outF = Nd4j.getExecutioner().exec(new ScalarMultiplication(in.dup('f'), 2.0));

        assertEquals(outC, outF);
    }

    @Test
    public void testCross() {
        INDArray a = Nd4j.create(new double[]{4, 2, 1}, new int[]{1, 3});
        INDArray b = Nd4j.create(new double[]{1, 3, 4}, new int[]{1, 3});

        INDArray expOut = Nd4j.create(DataType.DOUBLE, 1, 3);

        val op = new Cross(a, b, expOut);
        Nd4j.getExecutioner().exec(op);

        SameDiff sd = SameDiff.create();

        SDVariable sdA = sd.var("a", expOut.shape());
        SDVariable sdB = sd.var("b", expOut.shape());


        sd.associateArrayWithVariable(a, sdA);
        sd.associateArrayWithVariable(b, sdB);

        SDVariable t = sd.math().cross("cross", sdA, sdB);
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("cross", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testSpaceToDepth() {
        Nd4j.getRandom().setSeed(1337);

        int miniBatch = 128;
        int blockSize = 4;
        int[] inputShape = new int[]{miniBatch, 2 * blockSize, 2 * blockSize, 1};

        INDArray input = Nd4j.randn(inputShape);
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.var("in", inputShape);

        INDArray expOut = Nd4j.create(miniBatch, 2, 2, blockSize * blockSize);
        DynamicCustomOp op = new SpaceToDepth(input, expOut, blockSize, DataFormat.NHWC);
        Nd4j.getExecutioner().exec(op);

        sd.associateArrayWithVariable(input, sdInput);

        SDVariable t = sd.cnn().spaceToDepth("std", sdInput, blockSize, DataFormat.NHWC);
        //new SpaceToDepth(sd, sdInput, blockSize, dataFormat).outputVariable();
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("std", expOut)
                .gradientCheck(true));
        assertNull(err);
    }

    @Test
    public void testDepthToSpace() {
        Nd4j.getRandom().setSeed(1337);

        int miniBatch = 128;
        int blockSize = 4;
        int[] inputShape = new int[]{miniBatch, 2, 2, blockSize * blockSize};

        INDArray input = Nd4j.randn(inputShape);
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.var("in", inputShape);

        INDArray expOut = Nd4j.create(miniBatch, 2 * blockSize, 2 * blockSize, 1);
        DynamicCustomOp op = new DepthToSpace(input, expOut, blockSize, DataFormat.NHWC);
        Nd4j.getExecutioner().exec(op);

        sd.associateArrayWithVariable(input, sdInput);

        SDVariable t = sd.cnn().depthToSpace("dts", sdInput, blockSize, DataFormat.NHWC);
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("dts", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testBatchToSpace() {
        //OpValidationSuite.ignoreFailing();          //TODO: https://github.com/deeplearning4j/deeplearning4j/issues/6863
        Nd4j.getRandom().setSeed(1337);

        int miniBatch = 4;
        int[] inputShape = new int[]{miniBatch, 1, 1, 1};

        int M = 2;
        int[] blockShape = new int[]{M, 1};
        int[] cropShape = new int[]{M, 2};

        INDArray input = Nd4j.randn(inputShape).castTo(DataType.DOUBLE);
        INDArray crops = Nd4j.create(new float[]{0, 0, 0, 0}, cropShape).castTo(DataType.INT);

        SameDiff sd = SameDiff.create();

        SDVariable sdInput = sd.var("in", inputShape);

        INDArray expOut = Nd4j.create(DataType.DOUBLE, 1, 2, 2, 1);
        DynamicCustomOp op = DynamicCustomOp.builder("batch_to_space")
                .addInputs(input, crops)
                .addIntegerArguments(2)
                .addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(op);

        sd.associateArrayWithVariable(input, sdInput);

        SDVariable t = sd.cnn().batchToSpace("bts", sdInput, new int[]{2, 2}, new int[]{0, 0}, new int[]{0, 0});
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("bts", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testSpaceToBatch() {
        //OpValidationSuite.ignoreFailing();          //TODO: https://github.com/deeplearning4j/deeplearning4j/issues/6863

        Nd4j.getRandom().setSeed(7331);

        int miniBatch = 4;
        int[] inputShape = new int[]{1, 2, 2, 1};

        int M = 2;
        int[] blockShape = new int[]{M, 1};
        int[] paddingShape = new int[]{M, 2};

        INDArray input = Nd4j.randn(inputShape).castTo(DataType.DOUBLE);
        INDArray padding = Nd4j.create(new float[]{0, 0, 0, 0}, paddingShape).castTo(DataType.INT);

        SameDiff sd = SameDiff.create();

        SDVariable sdInput = sd.var("in", inputShape);

        INDArray expOut = Nd4j.create(DataType.DOUBLE, miniBatch, 1, 1, 1);
        DynamicCustomOp op = DynamicCustomOp.builder("space_to_batch")
                .addIntegerArguments(2)
                .addInputs(input, padding)
                .addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(op);

        sd.associateArrayWithVariable(input, sdInput);

        SDVariable t = sd.cnn().spaceToBatch("stb", sdInput, new int[]{2, 2}, new int[]{0, 0}, new int[]{0, 0});
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("stb", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testDynamicPartition() {
        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new double[]{4, 3, 5, 7, 8, 0});
        INDArray partitions = Nd4j.create(new double[]{1, 0, 1, 0, 0, 1}).castTo(DataType.INT);
        int numPartitions = 2;

        SDVariable in = sd.var("in", DataType.DOUBLE, new long[]{6});
        SDVariable sdPartitions = sd.var("partitions", DataType.INT, new long[]{6});

        INDArray expOut1 = Nd4j.create(DataType.DOUBLE, 3L);
        INDArray expOut2 = Nd4j.create(DataType.DOUBLE, 3L);
        INDArray[] expOut = new INDArray[]{expOut1, expOut2};

        DynamicCustomOp dynamicPartition = DynamicCustomOp.builder("dynamic_partition")
                .addInputs(ia, partitions)
                .addIntegerArguments(numPartitions)
                .addOutputs(expOut1, expOut2).build();
        Nd4j.getExecutioner().exec(dynamicPartition);

        SDVariable[] parts = sd.dynamicPartition(new String[]{"dp0", "dp1"}, in, sdPartitions, numPartitions);

        // merge the output partitions together again, to retrieve a single
        // tensor and finally a scalar.
        SDVariable t = sd.math().mergeAdd(parts);
        SDVariable loss = sd.mean("loss", t);

        sd.associateArrayWithVariable(ia, in);
        sd.associateArrayWithVariable(partitions, sdPartitions);

        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true)
                .gradCheckSkipVariables("partitions")
                .expectedOutput("dp0", expOut[0])
                .expectedOutput("dp1", expOut[1])
                .gradientCheck(true));
        assertNull(err);
    }

    @Test
    public void testDynamicPartition2() {
        INDArray data = Nd4j.createFromArray(2, 1, 2, 0);
        INDArray partitions = Nd4j.createFromArray(0, 2, 1, 0);
        INDArray[] out = Nd4j.exec(DynamicCustomOp.builder("dynamic_partition")
                .addOutputs(Nd4j.createUninitialized(DataType.INT, 2), Nd4j.createUninitialized(DataType.INT, 1), Nd4j.createUninitialized(DataType.INT, 1))
                .addIntegerArguments(3) //3 partitions
                .addInputs(data, partitions).build());

        INDArray exp0 = Nd4j.createFromArray(2, 0);
        INDArray exp1 = Nd4j.createFromArray(2);
        INDArray exp2 = Nd4j.createFromArray(1);

        assertEquals(exp0, out[0]);     //Usually just gives [0,0]
        assertEquals(exp1, out[1]);
        assertEquals(exp2, out[2]);
    }

    @Test
    public void testDynamicStitch() {
        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new double[]{5, 1, 3}, new long[]{3});
        INDArray ib = Nd4j.create(new double[]{7, 2, 4}, new long[]{3});
        INDArray indexA = Nd4j.create(new double[]{0, 1, 4}, new long[]{3}).castTo(DataType.INT);
        INDArray indexB = Nd4j.create(new double[]{2, 3, 5}, new long[]{3}).castTo(DataType.INT);

        INDArray expOut = Nd4j.create(DataType.DOUBLE, 6);

        DynamicCustomOp dynamicStitch = DynamicCustomOp.builder("dynamic_stitch")
                .addInputs(indexA, indexB, ia, ib)
                .addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(dynamicStitch);

        INDArray expOut2 = Nd4j.create(new double[]{5, 1, 7, 2, 3, 4});
        assertEquals(expOut2, expOut);

        SDVariable in1 = sd.var("in1", ia);
        SDVariable in2 = sd.var("in2", ib);

        SDVariable index1 = sd.constant("index1", indexA);
        SDVariable index2 = sd.constant("index2", indexB);

        SDVariable t = sd.dynamicStitch("ds", new SDVariable[]{index1, index2}, new SDVariable[]{in1, in2});
        SDVariable loss = sd.standardDeviation("loss", t, true);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("ds", expOut)
                .gradientCheck(true)
                .gradCheckSkipVariables("index1", "index2")

        );
        assertNull(err);
    }

    @Test
    public void testDiag() {
        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new double[]{1, 2}, new int[]{2});
        SDVariable in = sd.var("in", DataType.DOUBLE, new long[]{2});
        INDArray expOut = Nd4j.create(new double[][]{{1, 0}, {0, 2}});

        INDArray expOut2 = Nd4j.create(DataType.DOUBLE, 2, 2);
        DynamicCustomOp diag = DynamicCustomOp.builder("diag").addInputs(ia).addOutputs(expOut2).build();
        Nd4j.getExecutioner().exec(diag);

        assertEquals(expOut, expOut2);

        SDVariable t = sd.math().diag("diag", in);

        SDVariable loss = sd.standardDeviation("loss", t, false, 0, 1);

        sd.associateArrayWithVariable(ia, in);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("diag", expOut)
                .gradientCheck(true));
        assertNull(err);
    }

    @Test
    public void testDiagPart() {
        SameDiff sd = SameDiff.create();

        INDArray input = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(4, 4);
        INDArray expOut = Nd4j.create(new float[]{1, 6, 11, 16}).castTo(DataType.DOUBLE);

        SDVariable in = sd.var("in", input);
        SDVariable t = sd.math().diagPart("dp", in);

        // dimension is 0 here, because output of diagPart is vector, not matrix
        SDVariable loss = sd.standardDeviation("loss", t, true, 0);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("dp", expOut)
                .gradientCheck(true));
        assertNull(err);
    }

    @Test
    public void testEye() {
        int[] rows = new int[]{3, 3, 3, 3};
        int[] cols = new int[]{3, 2, 2, 2};
        int[][] batch = new int[][]{{}, {}, {4}, {3, 3}};
        INDArray[] expOut = new INDArray[4];

        expOut[0] = Nd4j.eye(3).castTo(DataType.DOUBLE);
        expOut[1] = Nd4j.create(new double[][]{{1, 0}, {0, 1}, {0, 0}});
        expOut[2] = Nd4j.create(DataType.DOUBLE, 4, 3, 2);
        for (int i = 0; i < 4; i++) {
            expOut[2].get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).assign(expOut[1]);
        }
        expOut[3] = Nd4j.create(DataType.DOUBLE, 3, 3, 3, 2);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                expOut[3].get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all()).assign(expOut[1]);
            }
        }

        for (int i = 0; i < 3; i++) {
            SameDiff sd = SameDiff.create();
            SDVariable eye = sd.math().eye("e", rows[i], cols[i], DataType.DOUBLE, batch[i]);

            SDVariable loss = sd.standardDeviation("loss", eye, true);

            String err = OpValidation.validate(new TestCase(sd)
                    .expectedOutput("e", expOut[i])
                    .gradCheckSkipVariables("e")
                    .gradientCheck(false));
            assertNull(err);
        }
    }

    @Test
    public void testEyeShape() {
        DynamicCustomOp dco = DynamicCustomOp.builder("eye")
                .addIntegerArguments(3, 3)
                //.addIntegerArguments(-99,3,3) //Also fails
                .build();

        val list = Nd4j.getExecutioner().calculateOutputShape(dco);
        assertEquals(1, list.size());   //Fails here - empty list
        assertArrayEquals(new long[]{3, 3}, list.get(0).getShape());
    }

    @Test
    public void testTransforms() {
        //Test transforms (non-pairwise)
        Nd4j.getRandom().setSeed(12345);

        List<String> allFailed = new ArrayList<>();
        for (int i = 0; i < 82; i++) {
            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;
            SDVariable in = sd.var("in", minibatch, nOut);

            INDArray ia = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

            int dim;
            SDVariable t;
            TestCase tc = new TestCase(sd);
            boolean stdevLoss = false;
            String opName = null;
            switch (i) {
                case 0:
                    t = in.add(5.0);
                    tc.expectedOutput(t.name(), ia.add(5.0));
                    break;
                case 1:
                    t = in.sub(5.0);
                    tc.expectedOutput(t.name(), ia.sub(5.0));
                    break;
                case 2:
                    t = in.mul(2.5);
                    tc.expectedOutput(t.name(), ia.mul(2.5));
                    break;
                case 3:
                    t = in.div(4.0);
                    tc.expectedOutput(t.name(), ia.div(4.0));
                    break;
                case 4:
                    t = in.rsub(5.0);
                    tc.expectedOutput(t.name(), ia.rsub(5.0));
                    break;
                case 5:
                    t = in.rdiv(1.0);
                    tc.expectedOutput(t.name(), ia.rdiv(1.0));
                    break;
                case 6:
                    t = sd.math().pow(in, 2.5);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut);
                    tc.expectedOutput(t.name(), Transforms.pow(ia, 2.5, true));
                    break;
                case 7:
                    t = sd.nn().sigmoid(in);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut).muli(2).subi(1.0);
                    tc.expectedOutput(t.name(), Transforms.sigmoid(ia, true));
                    break;
                case 8:
                    t = sd.math().tanh(in);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut).muli(2).subi(1.0);
                    tc.expectedOutput(t.name(), Transforms.tanh(ia, true));
                    break;
                case 9:
                    ia.assign(Nd4j.rand(DataType.DOUBLE, ia.shape()));
                    t = sd.math().tan(in);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut);
                    tc.expectedOutput(t.name(), Transforms.tan(ia));
                    break;
                case 10:
                    t = sd.math().cos(in);
                    tc.expectedOutput(t.name(), Transforms.cos(ia, true));
                    break;
                case 11:
                    t = sd.math().sin(in);
                    tc.expectedOutput(t.name(), Transforms.sin(ia, true));
                    break;
                case 12:
                    t = sd.nn().softplus(in);
                    tc.expectedOutput(t.name(), Transforms.softPlus(ia, true));
                    break;
                case 13:
                    t = sd.math().log(in);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut);
                    tc.expectedOutput(t.name(), Transforms.log(ia, true));
                    break;
                case 14:
                    t = sd.math().neg(in);
                    INDArray exp14 = ia.neg();
                    tc.expectedOutput(t.name(), exp14);
                    break;
                case 15:
                    t = sd.math().acos(in);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut).muli(1.8).subi(0.9);
                    tc.expectedOutput(t.name(), Transforms.acos(ia, true));
                    break;
                case 16:
                    t = sd.math().acosh(in);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut).addi(1.01); //Only defined for x >= 1
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new ACosh(ia.dup())));
                    break;
                case 17:
                    t = sd.math().asin(in);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut).muli(1.8).subi(0.9);
                    tc.expectedOutput(t.name(), Transforms.asin(ia, true));
                    break;
                case 18:
                    t = sd.math().atan(in);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut).muli(4).subi(2);
                    tc.expectedOutput(t.name(), Transforms.atan(ia, true));
                    break;
                case 19:
                    t = sd.math().atanh(in);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut).muli(1.8).subi(0.9);
                    tc.expectedOutput(t.name(), Transforms.atanh(ia, true));
                    break;
                case 20:
                    t = sd.math().cosh(in);
                    tc.expectedOutput(t.name(), Transforms.cosh(ia, true));
                    break;
                case 21:
                    t = sd.math().cube(in);
                    tc.expectedOutput(t.name(), Transforms.pow(ia, 3.0, true));
                    break;
                case 22:
                    t = sd.nn().elu(in);
                    tc.expectedOutput(t.name(), Transforms.elu(ia, true));
                    break;
                case 23:
                    //TODO SHOULDN'T THIS HAVE A DIMENSION ARG???
                    t = sd.nn().softmax(in, -1);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new SoftMax(ia.dup()))[0]);
                    break;
                case 24:
                    t = sd.math().sqrt(in);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut);
                    tc.expectedOutput(t.name(), Transforms.sqrt(ia, true));
                    break;
                case 25:
                    t = sd.math().square(in);
                    tc.expectedOutput(t.name(), Transforms.pow(ia, 2.0, true));
                    break;
                case 26:
                    t = sd.transpose(in);
                    tc.expectedOutput(t.name(), ia.transpose().dup());
                    break;
                case 27:
                    t = sd.math().abs(in);
                    tc.expectedOutput(t.name(), Transforms.abs(ia, true));
                    break;
                case 28:
                    t = sd.math().sinh(in);
                    tc.expectedOutput(t.name(), Transforms.sinh(ia, true));
                    break;
                case 29:
                    t = sd.math().asinh(in);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new ASinh(ia.dup())));
                    break;
                case 30:
                    t = sd.math().exp(in);
                    tc.expectedOutput(t.name(), Transforms.exp(ia, true));
                    break;
                case 31:
                    t = sd.math().floor(in);
                    tc.expectedOutput(t.name(), Transforms.floor(ia, true));
                    break;
                case 32:
                    t = sd.nn().relu(in, 0.0);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.name(), Transforms.relu(ia, true));
                    break;
                case 33:
                    t = sd.nn().hardTanh(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(2).subi(1.0);
                    tc.expectedOutput(t.name(), Transforms.hardTanh(ia, true));
                    break;
                case 34:
                    t = sd.nn().logSigmoid(in);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new LogSigmoid(ia.dup())));
                    break;
                case 35:
                    t = sd.nn().swish(in);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new Swish(ia.dup())));
                    break;
                case 36:
                    t = sd.math().sign(in);
                    tc.expectedOutput(t.name(), Transforms.sign(ia, true));
                    break;
                case 37:
                    t = sd.nn().softsign(in);
                    tc.expectedOutput(t.name(), Transforms.softsign(ia, true));
                    break;
                case 38:
                    t = sd.nn().leakyRelu(in, 0.0);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.name(), Transforms.leakyRelu(ia, true));
                    break;
                case 39:
                    if (OpValidationSuite.IGNORE_FAILING)
                        continue;
                    t = sd.nn().logSoftmax(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(10).subi(5);
                    tc.expectedOutput(t.name(), Transforms.log(Transforms.softmax(ia, true)));
                    stdevLoss = true;
                    break;
                case 40:
                    t = sd.nn().selu(in);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new SELU(ia.dup())));
                    break;
                case 41:
                    t = sd.gt(in, 1.0).castTo(DataType.DOUBLE);
                    tc.expectedOutput(t.name(), ia.gt(1.0).castTo(DataType.DOUBLE)).gradientCheck(false);
                    break;
                case 42:
                    t = sd.gte(in, 1.0).castTo(DataType.DOUBLE);
                    tc.expectedOutput(t.name(), ia.gte(1.0).castTo(DataType.DOUBLE)).gradientCheck(false);
                    break;
                case 43:
                    t = sd.lt(in, 1.0).castTo(DataType.DOUBLE);
                    tc.expectedOutput(t.name(), ia.lt(1.0).castTo(DataType.DOUBLE)).gradientCheck(false);
                    break;
                case 44:
                    t = sd.lte(in, 1.0).castTo(DataType.DOUBLE);
                    tc.expectedOutput(t.name(), ia.lte(1.0).castTo(DataType.DOUBLE)).gradientCheck(false);
                    break;
                case 45:
                    t = sd.eq(in, 2.0).castTo(DataType.DOUBLE);
                    ia = Nd4j.linspace(1, minibatch * nOut, minibatch * nOut, DataType.DOUBLE).reshape('c', minibatch, nOut);
                    tc.expectedOutput(t.name(), ia.eq(2.0).castTo(DataType.DOUBLE)).gradientCheck(false);
                    break;
                case 46:
                    t = sd.neq(in, 2.0).castTo(DataType.DOUBLE);
                    ia = Nd4j.linspace(1, minibatch * nOut, minibatch * nOut, DataType.DOUBLE).reshape('c', minibatch, nOut);
                    tc.expectedOutput(t.name(), ia.neq(2.0).castTo(DataType.DOUBLE)).gradientCheck(false);
                    break;
                case 47:
                    t = sd.math().ceil(in);
                    tc.expectedOutput(t.name(), Transforms.ceil(ia, true));
                    break;
                case 48:
                    ia = Nd4j.randn(DataType.DOUBLE, ia.shape()).muli(2);
                    t = sd.math().clipByValue(in, -3, 2);
                    INDArray expOut48 = ia.dup();
                    BooleanIndexing.replaceWhere(expOut48, -3, Conditions.lessThan(-3));
                    BooleanIndexing.replaceWhere(expOut48, 2, Conditions.greaterThan(2));
                    tc.expectedOutput(t.name(), expOut48);
                    break;
                case 49:
                    //Clip by norm, dimension 0, some below threshold, some above
                    double clip = 2.0;
                    t = sd.math().clipByNorm(in, clip, 0);
                    ia = Nd4j.rand(DataType.DOUBLE, ia.shape());
                    ia.diviRowVector(ia.norm2(0)).muli(clip);  //Norm2 is now 'clip' (i.e., exactly at threshold
                    //System.out.println(ia.norm2(0));
                    ia.muliColumnVector(Nd4j.linspace(0.9, 1.1, ia.size(0), DataType.DOUBLE).reshape(ia.size(0), 1));
                    //System.out.println(ia.norm2(0));

                    INDArray expOut49 = Nd4j.create(DataType.DOUBLE, ia.shape());
                    for (int j = 0; j < ia.columns(); j++) {
                        INDArray origCol = ia.getColumn(j);
                        if (origCol.norm2Number().doubleValue() < clip) {
                            expOut49.putColumn(j, origCol);
                        } else {
                            expOut49.putColumn(j, origCol.mul(clip / origCol.norm2Number().doubleValue()));
                        }
                    }
                    tc.expectedOutput(t.name(), expOut49);
                    //System.out.println(expOut.norm2(0));
                    break;
                //TODO clip by norm along other dimensions
                case 50:
                    dim = 1;
                    t = sd.reverse(in, dim);
                    INDArray expOut50 = Nd4j.create(DataType.DOUBLE, ia.shape());
                    DynamicCustomOp reverse = DynamicCustomOp.builder("reverse")
                            .addIntegerArguments(dim)
                            .addInputs(ia).addOutputs(expOut50).build();
                    Nd4j.getExecutioner().exec(reverse);
                    tc.expectedOutput(t.name(), expOut50);
                    break;
                case 51:
                    dim = 0;
                    boolean exclusive = false;
                    boolean reverseBool = false;

                    t = sd.cumsum(in, exclusive, reverseBool, dim);
                    INDArray expOut51 = Nd4j.create(DataType.DOUBLE, ia.shape());
                    DynamicCustomOp cumsum = DynamicCustomOp.builder("cumsum")
                            .addIntegerArguments((exclusive) ? 1 : 0, (reverseBool) ? 1 : 0, dim)
                            .addInputs(ia).addOutputs(expOut51).build();
                    Nd4j.getExecutioner().exec(cumsum);
                    tc.expectedOutput(t.name(), expOut51);
                    break;
                case 52:
                    if (OpValidationSuite.IGNORE_FAILING) {
                        continue;
                    }
                    boolean ex = false;
                    boolean revBool = false;
                    t = sd.cumprod(in, ex, revBool, 0);
                    INDArray expOut52 = Nd4j.create(DataType.DOUBLE, ia.shape());
                    for (int s0 = 0; s0 < ia.size(0); s0++) {
                        for (int s1 = 0; s1 < ia.size(1); s1++) {
                            double prod = 1.0;
                            for (int x = 0; x <= s0; x++) {
                                prod *= ia.getDouble(x, s1);
                            }
                            expOut52.putScalar(s0, s1, prod);
                        }
                    }
                    tc.expectedOutput(t.name(), expOut52);
                    break;
                case 53:
                    if (OpValidationSuite.IGNORE_FAILING) {
                        continue;
                    }
                    t = sd.math().diag(in);
                    ia = Nd4j.create(new float[]{4, 2});
                    in = sd.var("in", 1, 2);
                    INDArray expOut53 = Nd4j.create(DataType.DOUBLE, 2, 2);
                    DynamicCustomOp op = DynamicCustomOp.builder("diag").addInputs(ia).addOutputs(expOut53).build();
                    Nd4j.getExecutioner().exec(op);
                    tc.expectedOutput(t.name(), expOut53);
                    break;
                case 54:
                    t = sd.math().erf(in);
                    INDArray expOut54 = Nd4j.createUninitialized(DataType.DOUBLE, ia.shape(), ia.ordering());
                    Nd4j.getExecutioner().exec(new Erf(ia, expOut54));
                    tc.expectedOutput(t.name(), expOut54);
                    break;
                case 55:
                    t = sd.math().erfc(in);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new Erfc(ia, Nd4j.createUninitialized(ia.shape(), ia.ordering()))));
                    break;
                case 56:
                    t = sd.math().expm1(in);
                    tc.expectedOutput(t.name(), Transforms.expm1(ia, true));
                    break;
                case 57:
                    t = sd.math().log1p(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.name(), Transforms.log1p(ia, true));
                    break;
                case 58:
                    t = sd.math().round(in);
                    tc.expectedOutput(t.name(), Transforms.round(ia, true));
                    break;
                case 59:
                    ia = Nd4j.create(new float[]{4, 2}).castTo(DataType.DOUBLE);
//                    in = sd.var("in", new int[]{1, 2});
                    t = sd.math().rsqrt(in);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new RSqrt(ia, Nd4j.create(ia.shape(), ia.ordering()))));
                    break;
                case 60:
                    t = sd.nn().relu6(in, 0);
                    ia = Nd4j.rand(DataType.DOUBLE, minibatch, nOut);
                    tc.expectedOutput(t.name(), Transforms.relu6(ia, true));
                    break;
                case 61:
                    ia = Nd4j.create(new float[]{2, 2}).castTo(DataType.DOUBLE);
                    sd.associateArrayWithVariable(ia, in);
                    double value = 42;
                    t = sd.fill(in.castTo(DataType.INT), DataType.DOUBLE, value);
                    tc.expectedOutput(t.name(), Nd4j.valueArrayOf(new int[]{2, 2}, 42)).gradientCheck(false);
                    opName = "fill";
                    break;
                case 62:
                    t = sd.nn().hardSigmoid(in);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new HardSigmoid(ia, ia.dup())));
                    break;
                case 63:
                    t = sd.scalarMax(in, 0.5);
                    tc.expectedOutput(t.name(), Transforms.max(ia, 0.5, true));
                    break;
                case 64:
                    t = sd.scalarMin(in, 0.5);
                    tc.expectedOutput(t.name(), Transforms.min(ia, 0.5, true));
                    break;
                case 65:
                    continue; // assign op was removed.
                case 66:
                    t = sd.math().floorMod(in, 0.5);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new ScalarFMod(ia.dup(), 0.5)));
                    break;
                case 67:
                    t = sd.math().reciprocal(in);
                    tc.expectedOutput(t.name(), ia.rdiv(1.0));
                    break;
                case 68:
                    t = sd.shape(in).castTo(DataType.DOUBLE);
                    tc.expectedOutput(t.name(), Nd4j.create(ArrayUtil.toDouble(ia.shape()))).gradientCheck(false);
                    break;
                case 69:
                    t = sd.rank(in).castTo(DataType.DOUBLE);
                    tc.expectedOutput(t.name(), Nd4j.scalar((double) ia.rank())).gradientCheck(false);
                    break;
                case 70:
                    t = sd.onesLike(in);
                    tc.expectedOutput(t.name(), Nd4j.ones(ia.shape()));
                    break;
                case 71:
                    ia = Nd4j.randn(DataType.DOUBLE, nOut, nOut);
                    t = sd.math().diagPart(in);
                    tc.expectedOutput(t.name(), Nd4j.create(new double[]{ia.getDouble(0, 0), ia.getDouble(1, 1), ia.getDouble(2, 2), ia.getDouble(3, 3)}).castTo(DataType.DOUBLE));
                    break;
                case 72:
                    t = sd.identity(in);
                    tc.expected(t, ia.dup());
                    break;
                case 73:
                    t = sd.math().step(in, 1.0);
                    tc.expected(t, ia.gte(1.0).castTo(DataType.DOUBLE));
                    break;
                case 74:
                    continue;
                case 75:
                    ia = Nd4j.rand(DataType.DOUBLE, ia.shape());
                    t = sd.math().log(in, 2);
                    tc.expected(t, Transforms.log(ia, 2, true));
                    break;
                case 76:
                    ia = Nd4j.rand(DataType.DOUBLE, ia.shape());
                    t = sd.math().log(in, 10);
                    tc.expected(t, Transforms.log(ia, 10, true));
                    break;
                case 77:
                    ia = Nd4j.rand(DataType.DOUBLE, ia.shape());
                    t = sd.matchCondition(in, Conditions.lessThan(0.5)).castTo(DataType.DOUBLE);
                    INDArray exp = ia.dup().lt(0.5).castTo(DataType.DOUBLE);
                    tc.expected(t, exp).gradientCheck(false);
                    break;
                case 78:
                    ia = Nd4j.rand(DataType.DOUBLE, ia.shape()).muli(2).subi(1);
                    t = sd.math().rationalTanh(in);
                    tc.expected(t, Nd4j.getExecutioner().exec(new RationalTanh(ia.dup())));
                    break;
                case 79:
                    ia = Nd4j.rand(DataType.DOUBLE, ia.shape()).muli(2).subi(1);
                    t = sd.math().rectifiedTanh(in);
                    tc.expected(t, Nd4j.getExecutioner().exec(new RectifiedTanh(ia.dup())));
                    break;
                case 80:
                    t = sd.nn().gelu(in);
                    INDArray gelu = Transforms.sigmoid(ia.mul(1.702)).mul(ia);
                    tc.expected(t, gelu);
                    break;
                case 81:
                    ia = Nd4j.rand(DataType.DOUBLE, ia.shape()).muli(0.5);
                    t = sd.nn().preciseGelu(in);
                    INDArray x3 = Transforms.pow(ia.mul(0.044715), 3, true);
                    INDArray inner1 = ia.add(x3).mul(Math.sqrt(2.0 / Math.PI));
                    INDArray inner2 = Transforms.tanh(inner1, true).addi(1.0);
                    INDArray geluPrecise = inner2.mul(ia).mul(0.5);
                    tc.expected(t, geluPrecise);
                    break;
                default:
                    throw new RuntimeException();
            }


            DifferentialFunction[] funcs = sd.ops();
            String name = opName == null ? funcs[0].opName() : opName;


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            SDVariable loss;
            if (stdevLoss) {
                loss = sd.standardDeviation("loss", t, false, Integer.MAX_VALUE);   //.standardDeviation("loss", t, true, Integer.MAX_VALUE);
            } else {
                loss = sd.mean("loss", t);
            }

            sd.associateArrayWithVariable(ia, in);

            tc.testName(name);
            String error = OpValidation.validate(tc, true);
            if (error != null) {
                allFailed.add(name + " - " + error);
            }
        }

        if (allFailed.size() > 0) {
            log.error("All failed transforms: " + allFailed);
            fail(allFailed.size() + " transforms failed");
        }
    }


    @Test
    public void testPairwiseTransforms() {
        /*
        add, sub, mul, div, rsub, rdiv
        eq, neq, gt, lt, gte, lte, or, and, xor
        min, max
        mmul
        tensormmul
         */
        //Test transforms (pairwise)
        Nd4j.getRandom().setSeed(12345);

        List<String> allFailed = new ArrayList<>();
        for (int i = 0; i < 23; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;
            SDVariable in1 = sd.var("in1", DataType.DOUBLE, minibatch, nOut);
            SDVariable in2 = sd.var("in2", DataType.DOUBLE, minibatch, nOut);

            INDArray ia = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
            INDArray ib = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

            SDVariable t;
            TestCase tc = new TestCase(sd);
            String opName = null;
            switch (i) {
                case 0:
                    t = in1.add(in2);
                    tc.expectedOutput(t.name(), ia.add(ib));
                    break;
                case 1:
                    t = in1.sub(in2);
                    tc.expectedOutput(t.name(), ia.sub(ib));
                    break;
                case 2:
                    t = in1.mul(in2);
                    tc.expectedOutput(t.name(), ia.mul(ib));
                    break;
                case 3:
                    t = in1.div(in2);
                    tc.expectedOutput(t.name(), ia.div(ib));
                    break;
                case 4:
                    t = in1.rsub(in2);
                    tc.expectedOutput(t.name(), ia.rsub(ib));
                    break;
                case 5:
                    ia.assign(Nd4j.rand(ia.shape())).addi(0.5);
                    ib.assign(Nd4j.rand(ib.shape())).addi(0.5);
                    t = in1.rdiv(in2);
                    tc.expectedOutput(t.name(), ia.rdiv(ib));
                    break;
                case 6:
                    t = sd.eq(in1, in2);
                    opName = "eq";
                    tc.expectedOutput(t.name(), ia.eq(ib)).gradientCheck(false);
                    break;
                case 7:
                    t = sd.neq(in1, in2);
                    opName = "neq";
                    tc.expectedOutput(t.name(), ia.neq(ib)).gradientCheck(false);
                    ;
                    break;
                case 8:
                    t = sd.gt(in1, in2);
                    opName = "gt";
                    tc.expectedOutput(t.name(), ia.gt(ib)).gradientCheck(false);
                    break;
                case 9:
                    t = sd.lt(in1, in2);
                    opName = "lt";
                    tc.expectedOutput(t.name(), ia.lt(ib)).gradientCheck(false);
                    break;
                case 10:
                    t = sd.gte(in1, in2);
                    opName = "gte";
                    INDArray expOut10 = Nd4j.create(DataType.BOOL, ia.shape());
                    Nd4j.getExecutioner().exec(new GreaterThanOrEqual(new INDArray[]{ia, ib}, new INDArray[]{expOut10}));
                    tc.expectedOutput(t.name(), expOut10).gradientCheck(false);
                    break;
                case 11:
                    t = sd.lte(in1, in2);
                    opName = "lte";
                    INDArray expOut11 = Nd4j.create(DataType.BOOL, ia.shape());
                    Nd4j.getExecutioner().exec(new LessThanOrEqual(new INDArray[]{ia, ib}, new INDArray[]{expOut11}));
                    tc.expectedOutput(t.name(), expOut11).gradientCheck(false);
                    break;
                case 12:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.math().or(in1.castTo(DataType.BOOL), in2.castTo(DataType.BOOL));
                    opName = "or";
                    tc.expectedOutput(t.name(), Transforms.or(ia.castTo(DataType.BOOL), ib.castTo(DataType.BOOL))).gradientCheck(false);
                    break;
                case 13:
                    ib = Nd4j.randn(DataType.DOUBLE, nOut, nOut);
                    t = sd.mmul(in1, in2);
                    tc.expectedOutput(t.name(), ia.mmul(ib));
                    break;
                case 14:
                    t = sd.max(in1, in2);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new Max(ia, ib, ia.dup()))[0]);
                    break;
                case 15:
                    t = sd.min(in1, in2);
                    tc.expectedOutput(t.name(), Nd4j.getExecutioner().exec(new Min(ia, ib, ia.dup()))[0]);
                    break;
                case 16:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.math().and(in1.castTo(DataType.BOOL), in2.castTo(DataType.BOOL));
                    opName = "and";
                    tc.expectedOutput(t.name(), Transforms.and(ia.castTo(DataType.BOOL), ib.castTo(DataType.BOOL))).gradientCheck(false);
                    break;
                case 17:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.math().xor(in1.castTo(DataType.BOOL), in2.castTo(DataType.BOOL));
                    opName = "xor";
                    tc.expectedOutput(t.name(), Transforms.xor(ia.castTo(DataType.BOOL), ib.castTo(DataType.BOOL))).gradientCheck(false);
                    break;
                case 18:
                    continue; //assign op was removed.
                case 19:
                    t = sd.math().atan2(in1, in2);
                    tc.expectedOutput(t.name(), Transforms.atan2(ib, ia));    //Note: y,x order for samediff; x,y order for transforms
                    break;
                case 20:
                    t = sd.math().mergeAdd(new SDVariable[]{in1, in2, in2});
                    tc.expectedOutput(t.name(), ia.add(ib).add(ib));
                    break;
                case 21:
                    t = in1.squaredDifference(in2);
                    INDArray expOut21 = Nd4j.create(ia.shape(), ia.ordering());
                    DynamicCustomOp squareDiff = DynamicCustomOp.builder("squaredsubtract")
                            .addInputs(ia, ib)
                            .addOutputs(expOut21)
                            .build();
                    Nd4j.getExecutioner().exec(squareDiff);
                    tc.expectedOutput(t.name(), expOut21);
                    break;
                case 22:
                    //set diag
                    ia = Nd4j.randn(DataType.DOUBLE, nOut, nOut);
                    ib = Nd4j.randn(DataType.DOUBLE, 1, nOut).reshape(nOut);
                    INDArray expOut22 = ia.dup();
                    for (int j = 0; j < nOut; j++) {
                        expOut22.putScalar(j, j, ib.getDouble(j));
                    }
                    t = sd.math().setDiag(in1, in2);
                    tc.expectedOutput(t.name(), expOut22);
                    break;
                default:
                    throw new RuntimeException();
            }


            DifferentialFunction[] funcs = sd.ops();
            String name = (opName == null ? funcs[0].opName() : opName);

            String msg = "test: " + i + " - " + name;
            log.info("***** Starting test: {} *****", msg);

            SDVariable loss = sd.mean("loss", t.castTo(DataType.DOUBLE));

            sd.associateArrayWithVariable(ia, in1);
            sd.associateArrayWithVariable(ib, in2);

            tc.testName(name);
            String error = OpValidation.validate(tc, true);
            if (error != null) {
                allFailed.add(name + "(" + error + ")");
            }
        }

        if (allFailed.size() > 0) {
            log.error("All failed transforms: " + allFailed);
            fail(allFailed.size() + " transforms failed: " + allFailed);
        }
    }

    @Test
    public void testIsX() {
        List<String> failed = new ArrayList<>();

        for (int i = 0; i < 4; i++) {

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", 4);

            SDVariable out;
            INDArray exp;
            INDArray inArr;
            switch (i) {
                case 0:
                    inArr = Nd4j.create(new double[]{10, Double.POSITIVE_INFINITY, 0, Double.NEGATIVE_INFINITY});
                    exp = Nd4j.create(new boolean[]{true, false, true, false});
                    out = sd.math().isFinite(in);
                    break;
                case 1:
                    inArr = Nd4j.create(new double[]{10, Double.POSITIVE_INFINITY, 0, Double.NEGATIVE_INFINITY});
                    exp = Nd4j.create(new boolean[]{false, true, false, true});
                    out = sd.math().isInfinite(in);
                    break;
                case 2:
                    //TODO: IsMax supports both bool and float out: https://github.com/deeplearning4j/deeplearning4j/issues/6872
                    inArr = Nd4j.create(new double[]{-3, 5, 0, 2});
                    exp = Nd4j.create(new boolean[]{false, true, false, false});
                    out = sd.math().isMax(in);
                    break;
                case 3:
                    inArr = Nd4j.create(new double[]{0, Double.NaN, 10, Double.NaN});
                    exp = Nd4j.create(new boolean[]{false, true, false, true});
                    out = sd.math().isNaN(in);
                    break;
                default:
                    throw new RuntimeException();
            }

            SDVariable other = sd.var("other", Nd4j.rand(DataType.DOUBLE, 4));

            SDVariable loss = out.castTo(DataType.DOUBLE).add(other).mean();
            TestCase tc = new TestCase(sd)
                    .gradientCheck(false)   //Can't gradient check - in -> boolean -> cast(double)
                    .expected(out, exp);

            in.setArray(inArr);

            String err = OpValidation.validate(tc, true);
            if (err != null) {
                failed.add(err);
            }
        }
        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testReplaceWhereScalar() {
        for (Condition c : new Condition[]{Conditions.lessThan(0.5), Conditions.greaterThan(0.5), Conditions.equals(0.5)}) {

            log.info("Testing condition: " + c.getClass().getSimpleName());
            INDArray inArr = Nd4j.rand(DataType.DOUBLE, 3, 4);
            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", inArr);
            SDVariable where = sd.replaceWhere(in, 10, c);

            INDArray exp = inArr.dup();
            BooleanIndexing.replaceWhere(exp, 10, c);

            SDVariable loss = where.std(true);

            TestCase tc = new TestCase(sd);

            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @Test
    public void testReplaceWhereArray() {
        for (Condition c : new Condition[]{Conditions.lessThan(0.5), Conditions.greaterThan(0.5), Conditions.equals(0.5)}) {

            INDArray inArr = Nd4j.rand(3, 4);
            INDArray inArr2 = Nd4j.valueArrayOf(3, 4, 10);
            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", inArr);
            SDVariable in2 = sd.var("in2", inArr2);
            SDVariable where = sd.replaceWhere(in, in2, c);

            INDArray exp = inArr.dup();
            BooleanIndexing.replaceWhere(exp, inArr2, c);

            SDVariable loss = where.std(true);

            TestCase tc = new TestCase(sd);

            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @Test
    public void testLogGrad() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable input = sameDiff.var("x", Nd4j.linspace(1, 4, 4, DataType.DOUBLE));
        SDVariable log = sameDiff.math().log(input);
        SDVariable sum = sameDiff.sum(log, Integer.MAX_VALUE);
        INDArray result = null;
        sameDiff.calculateGradients(Collections.emptyMap(), sameDiff.getVariables().keySet());
    }


    @Test
    public void testSigmoidBackwards() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        SDVariable input = sameDiff.var("x", inputs.get("x"));
        SDVariable sigmoid = sameDiff.nn().sigmoid(input);
        SDVariable sum = sameDiff.sum(sigmoid, Integer.MAX_VALUE);
        Map<String, INDArray> m = sameDiff.calculateGradients(Collections.emptyMap(), sameDiff.getVariables().keySet());
        INDArray arr = m.get(input.name());
        assertTrue(Nd4j.create(new double[][]{
                {0.1966, 0.1050},
                {0.0452, 0.0177}
        }).equalsWithEps(arr, 1e-2));
    }


/*    @Test
    public void testDepth() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable x = sameDiff.one("one",new long[]{2,2});
        assertEquals(0,x.depth());
        SDVariable sigmoid = sameDiff.sigmoid("sigmoid",x);
        assertEquals(1,sigmoid.depth());
    }*/

    @Test
    public void testRank0EdgeCase() {
        SameDiff sd = SameDiff.create();
        SDVariable v1 = sd.sum(sd.var(Nd4j.create(new double[]{4, 4})));
        double d0 = v1.eval().getDouble(0);
        assertEquals(8, d0, 0);

        SDVariable v2 = sd.sum(sd.var(Nd4j.create(new double[]{4, 4}))).div(2.0);
        Map<String, INDArray> m = sd.outputAll(Collections.emptyMap());
        double d1 = m.get(v2.name()).getDouble(0);
        assertEquals(4, d1, 0);
    }

    @Test
    public void testAtan2BroadcastShape() {
        INDArray arr1 = Nd4j.create(new long[]{3, 1, 4});
        INDArray arr2 = Nd4j.create(new long[]{1, 2, 4});

        DynamicCustomOp op = DynamicCustomOp.builder("tf_atan2")
                .addInputs(arr1, arr2)
                .build();

        val outShapes = Nd4j.getExecutioner().calculateOutputShape(op);
        assertEquals(1, outShapes.size());

        assertArrayEquals(Arrays.toString(outShapes.get(0).getShape()), new long[]{3, 2, 4}, outShapes.get(0).getShape());
    }

    @Test
    public void testBooleanAnd() {
        Nd4j.setDataType(DataType.FLOAT);
        INDArray arr1 = Nd4j.create(new long[]{3, 4});
        INDArray arr2 = Nd4j.create(new long[]{3, 4});
        INDArray out = Nd4j.create(new long[]{3, 4});

        DynamicCustomOp op = DynamicCustomOp.builder("boolean_and")
                .addInputs(arr1, arr2)
                .addOutputs(out)
                .build();
        Nd4j.getExecutioner().exec(op);
    }

    @Test
    public void testScatterOpsScalar() {
        for (String s : new String[]{"add", "sub", "mul", "div"}) {
            INDArray ref = Nd4j.linspace(1, 30, 30, DataType.DOUBLE).reshape(10, 3);
            INDArray indices = Nd4j.scalar(5);
            INDArray upd = Nd4j.create(new double[]{10, 20, 30});

            //The non-scalar case works:
//            INDArray indices = Nd4j.create(new float[]{5});
//            INDArray upd = Nd4j.create(new double[]{10, 20, 30}, new int[]{1, 3});

            INDArray exp = ref.dup();
            switch (s) {
                case "add":
                    exp.getRow(5).addi(upd);
                    break;
                case "sub":
                    exp.getRow(5).subi(upd);
                    break;
                case "mul":
                    exp.getRow(5).muli(upd);
                    break;
                case "div":
                    exp.getRow(5).divi(upd);
                    break;
                default:
                    throw new RuntimeException();
            }


            INDArray out = Nd4j.create(10, 3);

            DynamicCustomOp op = DynamicCustomOp.builder("scatter_" + s)
                    .addInputs(ref, indices, upd)
                    .addOutputs(out)
                    .build();

            Nd4j.getExecutioner().exec(op);

            assertEquals(s, exp, out);
        }
    }


    @Ignore("12/16/2019 https://github.com/eclipse/deeplearning4j/issues/8540")
    @Test
    public void testPad() {
        INDArray in = Nd4j.valueArrayOf(new long[]{5}, 1.0);
        INDArray pad = Nd4j.create(new double[]{1, 1}, new long[]{1, 2}).castTo(DataType.LONG);
        INDArray value = Nd4j.scalar(10.0);

        INDArray out = Nd4j.create(new long[]{7});

        DynamicCustomOp op = DynamicCustomOp.builder("pad")
                .addInputs(in, pad, value)
                //.addInputs(in, pad) //Also doesn't work
                .addOutputs(out)
                .addIntegerArguments(0) //0 = CONSTANT
                .build();

        INDArray exp = Nd4j.create(new double[]{10, 1, 1, 1, 1, 1, 10});
        OpValidation.validate(new OpTestCase(op)
                .expectedOutput(0, exp));

        SameDiff sd = SameDiff.create();
        SDVariable s = sd.var("in", in);
        SDVariable padded = sd.nn().pad(s, sd.constant(pad), 10.0);
        String err2 = OpValidation.validate(new TestCase(sd).expected(padded, exp).gradientCheck(false));
        assertNull(err2);
    }


    @Test
    public void testMirrorPad() {
        INDArray in = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
        INDArray pad = Nd4j.create(new double[][]{{1, 1}, {2, 2}}).castTo(DataType.INT);

        INDArray out = Nd4j.create(DataType.DOUBLE, 4, 7);

        DynamicCustomOp op = DynamicCustomOp.builder("mirror_pad")
                .addInputs(in, pad)
                .addOutputs(out)
                .addIntegerArguments(0) //0=reflect, 1=symmetric
                .build();

        Nd4j.getExecutioner().exec(op);

        INDArray exp = Nd4j.create(new double[][]{
                {6, 5, 4, 5, 6, 5, 4},
                {3, 2, 1, 2, 3, 2, 1},
                {6, 5, 4, 5, 6, 5, 4},
                {3, 2, 1, 2, 3, 2, 1}});
        String err = OpValidation.validate(new OpTestCase(op)
                .expectedOutput(0, exp));

        assertNull(err);


        SameDiff sd = SameDiff.create();
        SDVariable s = sd.var("in", in);
        SDVariable padded = sd.nn().pad(s, sd.constant(Nd4j.createFromArray(new int[][]{{1,1},{2,2}})), PadMode.REFLECT, 0.0);
        String err2 = OpValidation.validate(new TestCase(sd).expected(padded, exp).gradientCheck(false));
        assertNull(err2);
    }

    @Test
    public void testMirrorPad2() {
        INDArray in = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
        INDArray pad = Nd4j.create(new double[][]{{1, 1}, {2, 2}}).castTo(DataType.INT);

        INDArray out = Nd4j.create(DataType.DOUBLE, 4, 7);

        DynamicCustomOp op = DynamicCustomOp.builder("mirror_pad")
                .addInputs(in, pad)
                .addOutputs(out)
                .addIntegerArguments(1) //0=reflect, 1=symmetric
                .build();

        Nd4j.getExecutioner().exec(op);

        INDArray exp = Nd4j.create(new double[][]{
                {2, 1, 1, 2, 3, 3, 2},
                {2, 1, 1, 2, 3, 3, 2},
                {5, 4, 4, 5, 6, 6, 5},
                {5, 4, 4, 5, 6, 6, 5}});
        String err = OpValidation.validate(new OpTestCase(op)
                .expectedOutput(0, exp));

        assertNull(err);
    }

    @Test
    public void testMirrorPadSymmetric() {
        INDArray in = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);
        INDArray pad = Nd4j.create(new double[][]{{1, 1}, {1, 1}}).castTo(DataType.INT);

        INDArray out = Nd4j.create(DataType.DOUBLE, 5, 6);

        DynamicCustomOp op = DynamicCustomOp.builder("mirror_pad")
                .addInputs(in, pad)
                .addOutputs(out)
                .addIntegerArguments(1) //0=reflect, 1=symmetric
                .build();

        Nd4j.getExecutioner().exec(op);

        INDArray exp = Nd4j.create(new double[][]{
                {1, 1, 2, 3, 4, 4},
                {1, 1, 2, 3, 4, 4},
                {5, 5, 6, 7, 8, 8},
                {9, 9, 10, 11, 12, 12},
                {9, 9, 10, 11, 12, 12}});
        String err = OpValidation.validate(new OpTestCase(op)
                .expectedOutput(0, exp));

        assertNull(err);
    }

    @Test
    public void testUnique() {
        INDArray in = Nd4j.create(new double[]{3, 4, 3, 1, 3, 0, 2, 4, 2, 4});

        INDArray expUnique = Nd4j.create(new double[]{3, 4, 1, 0, 2});
        INDArray expUniqueIdxs = Nd4j.create(new double[]{0, 1, 0, 2, 0, 3, 4, 1, 4, 1}).castTo(DataType.LONG);

        INDArray outUnique = Nd4j.create(DataType.DOUBLE, expUnique.shape());
        INDArray outUniqueIdxs = Nd4j.create(DataType.LONG, expUniqueIdxs.shape());

        DynamicCustomOp op = DynamicCustomOp.builder("unique")
                .addInputs(in)
                .addOutputs(outUnique, outUniqueIdxs)
                .build();

        String err = OpValidation.validate(new OpTestCase(op)
                .expectedOutput(0, expUnique)
                .expectedOutput(1, expUniqueIdxs));

        assertNull(err);
    }

    @Test
    public void testTopK() {
        OpValidationSuite.ignoreFailing();  //Can't assume sorted here
        INDArray in = Nd4j.create(new double[]{7, 3, 1, 2, 5, 0, 4, 6, 9, 8});

        INDArray expTopK = Nd4j.create(new double[]{7, 5, 6, 9, 8});
        INDArray expIndices = Nd4j.create(new double[]{0, 4, 7, 8, 9});

        INDArray expTopK_sorted = Nd4j.create(new double[]{9, 8, 7, 6, 5});
        INDArray expIndices_sorted = Nd4j.create(new double[]{8, 9, 0, 7, 4});

        for (boolean sort : new boolean[]{false, true}) {
            INDArray outUnique = Nd4j.create(expTopK.shape());
            INDArray outUniqueIdxs = Nd4j.create(expIndices.shape());

            DynamicCustomOp op = DynamicCustomOp.builder("top_k")
                    .addInputs(in)
                    .addOutputs(outUnique, outUniqueIdxs)
                    .addIntegerArguments(5, sort ? 1 : 0)  //k=5, sort
                    .build();

            String err = OpValidation.validate(new OpTestCase(op)
                    .expectedOutput(0, sort ? expTopK_sorted : expTopK)
                    .expectedOutput(1, sort ? expIndices_sorted : expIndices));

            assertNull(err);
        }
    }

    @Test
    public void testTopK1() {
        INDArray x = Nd4j.createFromArray(0.0, 0.0, 0.0, 10.0, 0.0);
        INDArray k = Nd4j.scalar(1);
        INDArray outValue = Nd4j.create(DataType.DOUBLE, 1);
        INDArray outIdx = Nd4j.create(DataType.INT, 1);

        Nd4j.exec(DynamicCustomOp.builder("top_k")
                .addInputs(x, k)
                .addOutputs(outValue, outIdx)
                .addBooleanArguments(false) //not sorted
                .addIntegerArguments(1)
                .build());

        INDArray expValue = Nd4j.createFromArray(10.0);
        INDArray expIdx = Nd4j.createFromArray(3);

        assertEquals(expValue, outValue);
        assertEquals(expIdx, outIdx);
    }

    @Test
    public void testInTopK() {
        for (int k = 4; k >= 1; k--) {
            log.info("Testing: k=" + k);
            INDArray in = Nd4j.linspace(1, 20, 20, DataType.DOUBLE).reshape(4, 5);
            INDArray idxs = Nd4j.create(new double[]{1, 2, 3, 4}).castTo(DataType.INT);

            INDArray expOut;
            switch (k) {
                case 4:
                    expOut = Nd4j.create(new boolean[]{true, true, true, true});
                    break;
                case 3:
                    expOut = Nd4j.create(new boolean[]{false, true, true, true});
                    break;
                case 2:
                    expOut = Nd4j.create(new boolean[]{false, false, true, true});
                    break;
                case 1:
                    expOut = Nd4j.create(new boolean[]{false, false, false, true});
                    break;
                default:
                    throw new RuntimeException();
            }


            INDArray out = Nd4j.create(DataType.BOOL, expOut.shape());

            DynamicCustomOp op = DynamicCustomOp.builder("in_top_k")
                    .addInputs(in, idxs)
                    .addOutputs(out)
                    .addIntegerArguments(k)  //k=1
                    .build();

            String err = OpValidation.validate(new OpTestCase(op)
                    .expectedOutput(0, expOut));

            assertNull(err);
        }
    }

    @Test
    public void testZeta() {
        OpValidationSuite.ignoreFailing();  //https://github.com/deeplearning4j/deeplearning4j/issues/6182
        INDArray x = Nd4j.rand(3, 4).addi(1.0);
        INDArray q = Nd4j.rand(3, 4);

        INDArray out = Nd4j.create(3, 4);
        DynamicCustomOp op = DynamicCustomOp.builder("zeta")
                .addInputs(x, q)
                .addOutputs(out)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertNotEquals(Nd4j.create(out.shape()), out);
    }

    @Test
    public void testMaxEmptyScalar() {
        INDArray empty = Nd4j.empty(DataType.FLOAT);
        INDArray scalar = Nd4j.scalar(1.0f);

        DynamicCustomOp op = DynamicCustomOp.builder("maximum")
                .addInputs(empty, scalar)
                .build();

        List<LongShapeDescriptor> l = op.calculateOutputShape();
        assertEquals(1, l.size());
        long[] shape = l.get(0).getShape();
        boolean isEmpty = l.get(0).isEmpty();

        assertTrue(isEmpty);
    }

    @Test
    public void testBroadcastEmpty() {
//        Nd4j.getExecutioner().enableVerboseMode(true);
//        Nd4j.getExecutioner().enableDebugMode(true);
        //Check broadcast behaviour with empty arrays. The idea is to match TF import behaviour, for import
        //TF behaviour: broadcastableOp(x,empty) -> empty

        /*
        tf.reset_default_graph()
        # Hack to create empty array
        input = tf.constant([False], dtype=tf.bool)
        empty = tf.where(condition=input)
        emptyFloat = tf.cast(empty, tf.float32)
        emptyFloat = tf.reshape(emptyFloat, [0,1])
        constScalar = tf.constant(1, dtype=tf.float32)
        # out = tf.math.maximum(emptyFloat,constScalar)
        # out = emptyFloat + constScalar
        # out = emptyFloat / constScalar
        out = tf.math.less(emptyFloat, constScalar)
        sess = tf.Session()
        out = sess.run([out])
         */

        for (int i = 0; i < 3; i++) {
            for (boolean scalar : new boolean[]{true, false}) {
                INDArray x = scalar ? Nd4j.scalar(2f) : Nd4j.create(DataType.FLOAT, 3, 4);
                INDArray y = scalar ? Nd4j.scalar(3f) : Nd4j.create(DataType.FLOAT, 3, 4);
                switch (i) {
                    case 0:
                        //x only empty
                        x = Nd4j.empty(DataType.FLOAT);
                        break;
                    case 1:
                        //y only empty
                        y = Nd4j.empty(DataType.FLOAT);
                        break;
                    case 2:
                        //Both empty
                        x = Nd4j.empty(DataType.FLOAT);
                        y = Nd4j.empty(DataType.FLOAT);
                        break;
                    default:
                        throw new RuntimeException();
                }


                for (String opName : new String[]{"maximum", "minimum", "add", "subtract", "multiply", "divide", "assign",
                        "boolean_and", "boolean_or", "boolean_xor", "tf_atan2", "equals", "floordiv", "floormod", "greater",
                        "greater_equal", "less", "less_equal", "mod", "not_equals", "realdiv", "reversedivide", "reversesubtract",
                        "squaredsubtract", "truncatediv"}) {

//                    log.info("Starting op: {}, case {} - x.isScalar()={}, x.isEmpty()={}, y.isScalar()={}, y.isEmpty()={}", opName, i,
//                            x.isScalar(), x.isEmpty(), y.isScalar(), y.isEmpty());

                    DynamicCustomOp op = DynamicCustomOp.builder(opName)
                            .addInputs(x, y)
                            .build();

                    List<LongShapeDescriptor> l = op.calculateOutputShape();
                    assertEquals(1, l.size());
                    long[] shape = l.get(0).getShape();
                    boolean empty = l.get(0).isEmpty();

                    boolean isBool = isBoolBroadcast(opName);
                    if (isBool) {
                        assertEquals(DataType.BOOL, l.get(0).dataType());
                    } else {
                        assertEquals(DataType.FLOAT, l.get(0).dataType());
                    }

                    assertArrayEquals(new long[0], shape);
                    assertTrue(empty);


                    INDArray out = Nd4j.empty(isBool ? DataType.BOOL : DataType.FLOAT);
                    op.addOutputArgument(out);

                    Nd4j.exec(op);
                }
            }
        }
    }

    private static boolean isBoolBroadcast(String opName) {
        if (opName.startsWith("greater") || opName.startsWith("less") || opName.contains("equals"))
            return true;
        //Note that "boolean" ops are inherit
        return false;
    }

    @Test
    public void testStandardize() {
        final INDArray random = Nd4j.rand(new int[]{10, 4});

        final int[] axis = new int[]{1};
        final INDArray means = random.mean(axis);
        final INDArray std = random.std(false, axis);
        final INDArray res = random.subColumnVector(means).divColumnVector(std);
        final INDArray expOut = res.norm1();

        SameDiff sd = SameDiff.create();
        SDVariable sdA = sd.var("a", random);
        SDVariable t = sd.math.standardize(sdA, axis);
        t.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testStandardizeOP() {
        final INDArray random = Nd4j.rand(new int[]{10, 4});

        final int[] axis = new int[]{1};
        final INDArray means = random.mean(axis);
        final INDArray std = random.std(false, axis);
        final INDArray res = random.subColumnVector(means).divColumnVector(std);

        final INDArray output = Nd4j.zerosLike(res);
        Nd4j.getExecutioner().exec(new Standardize(random, output, 1));

        assertEquals(res, output);
    }

    @Test
    public void testStandardizeNoDeviation() {
        final INDArray random = Nd4j.rand(new int[]{10, 4});
        for (int i = 0; i < 4; i++) {
            random.putScalar(1, i, 7);
        }

        final int[] axis = new int[]{1};
        final INDArray means = random.mean(axis);
        final INDArray std = random.std(false, axis);
        std.addi(std.eq(0).castTo(DataType.DOUBLE));

        final INDArray res = random.subColumnVector(means).divColumnVector(std);
        final INDArray expOut = res.norm1();

        SameDiff sd = SameDiff.create();
        SDVariable sdA = sd.var("a", random);
        SDVariable t = sd.math.standardize(sdA, axis);
        t.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testMatMulTensor() {
        final INDArray a = Nd4j.rand(new int[]{1, 2, 3, 4, 5});
        final INDArray b = Nd4j.rand(new int[]{1, 2, 3, 5, 6});

        final INDArray z = Nd4j.matmul(a, b);

        assertArrayEquals(z.shape(), new long[]{1, 2, 3, 4, 6});

        SameDiff sd = SameDiff.create();
        SDVariable sdA = sd.var("a", a);
        SDVariable sdB = sd.var("b", b);
        SDVariable t = sd.mmul(sdA, sdB);
        t.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testMatMulTensorTranspose() {
        for (boolean transposeA : new boolean[]{false, true}) {
            for (boolean transposeB : new boolean[]{false, true}) {
                for (boolean transposeResult : new boolean[]{false, true}) {
                    log.info("Testing with transposeA={}; transposeB={}; transposeResult={};", transposeA, transposeB, transposeResult);
                    int m = 0, n = 0, k = 0, l = 0, i = 0, j = 0;
                    if (!transposeA && !transposeB && !transposeResult) {
                        m = 4;
                        n = 5;
                        k = 5;
                        l = 6;
                        i = 4;
                        j = 6;
                    }
                    if (!transposeA && transposeB && !transposeResult) {
                        m = 4;
                        n = 5;
                        k = 6;
                        l = 5;
                        i = 4;
                        j = 6;
                    }
                    if (!transposeA && !transposeB && transposeResult) {
                        m = 4;
                        n = 5;
                        k = 5;
                        l = 6;
                        i = 6;
                        j = 4;
                    }
                    if (!transposeA && transposeB && transposeResult) {
                        m = 4;
                        n = 5;
                        k = 6;
                        l = 5;
                        i = 6;
                        j = 4;
                    }
                    if (transposeA && !transposeB && !transposeResult) {
                        m = 5;
                        n = 4;
                        k = 5;
                        l = 6;
                        i = 4;
                        j = 6;
                    }
                    if (transposeA && transposeB && !transposeResult) {
                        m = 5;
                        n = 4;
                        k = 6;
                        l = 5;
                        i = 4;
                        j = 6;
                    }
                    if (transposeA && !transposeB && transposeResult) {
                        m = 5;
                        n = 4;
                        k = 5;
                        l = 6;
                        i = 6;
                        j = 4;
                    }
                    if (transposeA && transposeB && transposeResult) {
                        m = 5;
                        n = 4;
                        k = 6;
                        l = 5;
                        i = 6;
                        j = 4;
                    }

                    final INDArray a = Nd4j.rand(new int[]{1, 2, 3, m, n});
                    final INDArray b = Nd4j.rand(new int[]{1, 2, 3, k, l});

                    final INDArray z = Nd4j.matmul(a, b, transposeA, transposeB, transposeResult);

                    assertArrayEquals(z.shape(), new long[]{1, 2, 3, i, j});

                    SameDiff sd = SameDiff.create();
                    SDVariable sdA = sd.var("a", a);
                    SDVariable sdB = sd.var("b", b);
                    SDVariable t = sd.mmul(sdA, sdB, transposeA, transposeB, transposeResult);
                    t.norm1("out");

                    String err = OpValidation.validate(new TestCase(sd)
                            .gradientCheck(true));
                    assertNull(err, err);
                }
            }
        }
    }

    @Test
    public void testSoftmaxCF() {

        INDArray arrC = Nd4j.rand(DataType.FLOAT, 2, 5);
        INDArray arrF = arrC.dup('f');
        INDArray outCC = Nd4j.create(DataType.FLOAT, arrC.shape(), 'c');
        INDArray outCF = Nd4j.create(DataType.FLOAT, arrC.shape(), 'f');
        INDArray outFC = Nd4j.create(DataType.FLOAT, arrC.shape(), 'c');
        INDArray outFF = Nd4j.create(DataType.FLOAT, arrC.shape(), 'f');


        Nd4j.exec(DynamicCustomOp.builder("softmax").addInputs(arrC).addOutputs(outCC).build());
        Nd4j.exec(DynamicCustomOp.builder("softmax").addInputs(arrC).addOutputs(outCF).build());
        Nd4j.exec(DynamicCustomOp.builder("softmax").addInputs(arrF).addOutputs(outFC).build());
        Nd4j.exec(DynamicCustomOp.builder("softmax").addInputs(arrF).addOutputs(outFF).build());

        assertEquals(outCC, outCF);
        assertEquals(outCC, outFC);
        assertEquals(outCC, outFF);
    }

    @Test
    public void testLogSumExp() {
        Nd4j.getRandom().setSeed(12345);
        INDArray inputArr = Nd4j.rand(DataType.FLOAT, 1, 4);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var(inputArr);
        SDVariable lse = sd.math().logSumExp(in);
        INDArray out = lse.eval();

        INDArray exp = Transforms.exp(inputArr, true);
        INDArray sum = exp.sum();
        INDArray log = Transforms.log(sum);
        assertEquals(log, out);
    }

    @Test
    public void testLogSumExp2() {

        for (int dim = 0; dim <= 2; dim++) {
            Nd4j.getRandom().setSeed(12345);
            INDArray inputArr = Nd4j.rand(DataType.DOUBLE, 3, 4, 5);
            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var(inputArr);
            SDVariable lse = sd.math().logSumExp(in, dim);

            INDArray exp = Transforms.exp(inputArr, true);
            INDArray sum = exp.sum(dim);
            INDArray log = Transforms.log(sum);

            OpValidation.validate(new TestCase(sd)
                    .expectedOutput(lse.name(), log)
                    .gradientCheck(true));
        }
    }


    @Test
    public void testCRELU() {

        Nd4j.getRandom().setSeed(12345);
        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, 2, 2);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var(inputArr);

        SDVariable crelu = new CReLU(sd, in).outputVariable();
        INDArray expected = Nd4j.concat(1, Nd4j.nn.relu(inputArr, 0), Nd4j.nn.relu(inputArr.neg(), 0));

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("crelu", expected)
                .gradientCheck(true)
        );

        assertNull(err);
    }

    @Test
    public void testClipByAvgNorm() {

        Nd4j.getRandom().setSeed(12345);
        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, 2, 2, 2);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var(inputArr);
        SDVariable out = new ClipByAvgNorm(sd, in, 1e-2, 0, 1, 2).outputVariable();
        SDVariable expected = sd.math.clipByNorm(in, 1e-2, 0, 1, 2).mul(inputArr.length());

        SDVariable loss = sd.standardDeviation("loss", out, true);
        loss.markAsLoss();

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("clipbyavgnorm", expected.eval())
                .gradientCheck(false)
        );
        assertNull(err);

    }

    @Test
    public void testEmbeddingLookup() {

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.var("in", Nd4j.rand(1024, 10));
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(new long[]{0, 5, 17, 33}));
        SDVariable out = new EmbeddingLookup(sd, input, indices, PartitionMode.MOD).outputVariable();
        // should be matrix of shape [4, 10]
        assertArrayEquals(new long[]{4, 10}, out.eval().shape());

    }

    @Test
    public void testImageResize() {

        //TODO: Methods failed ResizeLanczos5, ResizeMitchelcubic, ResizeArea

        for (ImageResizeMethod method : ImageResizeMethod.values()) {
                if (method==ImageResizeMethod.ResizeLanczos5 || method==ImageResizeMethod.ResizeArea || method==ImageResizeMethod.ResizeMitchellcubic)
                {continue;}

                log.info("Trying {}", method);

                Nd4j.getRandom().setSeed(12345);
                SameDiff sd = SameDiff.create();
                boolean preserveAspectRatio = true;
                boolean antialias = true;
                SDVariable inputImage = sd.var(Nd4j.rand(DataType.FLOAT, 1, 5, 5, 3));
                //  NHWC format
                long[] expectedShape = new long[]{1, 3, 3, 3};
                SDVariable requestedSize = sd.constant(Nd4j.createFromArray( new long[]{3, 3}));

                Function<INDArray, String> checkFunction = in -> {
                    boolean shapeOk = Arrays.equals(expectedShape, in.shape());
                    if (shapeOk) return null;
                    return "Failed: shape differs - expected " + Arrays.toString(expectedShape) + " vs " + Arrays.toString(in.shape()) + " on method " + method;
                };


                SDVariable out = new ImageResize(sd, inputImage, requestedSize, preserveAspectRatio, antialias, method).outputVariable().std(true);

                String err = OpValidation.validate(new TestCase(sd)
                        .gradientCheck(false)
                        .expected("image_resize", checkFunction));

            assertNull(err);


        }
        }




    @Test
    public void testMaximumBp() {

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable inputX = sd.var(Nd4j.rand(2, 3));
        SDVariable inputY = sd.var(Nd4j.rand(2, 3));


        SDVariable out = new org.nd4j.linalg.api.ops.impl.transforms.custom.Max(sd, inputX, inputY).outputVariable().std(true);
        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true));
        assertNull(err);


    }

    @Test
    public void testMergeAddBp() {

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable inputX = sd.var(Nd4j.rand(2, 3));
        SDVariable inputY = sd.var(Nd4j.rand(2, 3));
        SDVariable inputZ = sd.var(Nd4j.rand(2, 3));
        SDVariable out = new MergeAddOp(sd, new SDVariable[]{inputX, inputY, inputZ}).outputVariable().std(true);
        out.markAsLoss();
        String err =  OpValidation.validate(new TestCase(sd)
                .gradientCheck(true));
        assertNull(err);


    }

    @Test
    public void testMergeMaxBp() {

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable inputX = sd.var(Nd4j.rand(2, 3));
        SDVariable inputY = sd.var(Nd4j.rand(2, 3));
        SDVariable inputZ = sd.var(Nd4j.rand(2, 3));
        SDVariable out = new MergeMax(sd, inputX, inputY, inputZ).outputVariable().std(true);
        out.markAsLoss();
        String err =  OpValidation.validate(new TestCase(sd)
                .gradientCheck(true));
        assertNull(err);


    }


    @Test
    public void testMergeAvgBp() {

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable inputX = sd.var(Nd4j.rand(2, 3));
        SDVariable inputY = sd.var(Nd4j.rand(2, 3));
        SDVariable inputZ = sd.var(Nd4j.rand(2, 3));
        SDVariable out = new MergeAvg(sd, inputX, inputY, inputZ).outputVariable().std(true);
        out.markAsLoss();
        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true));
        assertNull(err);


    }

    @Test
    public void testReverseBp() {

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.var(Nd4j.createFromArray(new double[][]{{2,7}, {3,5}, {4,5}}));
        SDVariable out = new Reverse(sd, input,0).outputVariable();
        SDVariable loss = out.std(true);
        loss.markAsLoss();
        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true));
        assertNull(err);
    }

    @Test
    public void testUpsampling3dBp() {

        Nd4j.getRandom().setSeed(12345);
        for (boolean dataformat : new boolean[]{true, false}) {

            SameDiff sd = SameDiff.create();

            // NCDHW input
            SDVariable input = dataformat ? sd.var(Nd4j.rand(DataType.DOUBLE, 2, 1, 5, 5, 5)) : sd.var(Nd4j.rand(DataType.DOUBLE, 2, 5, 5, 5, 1));
            int scaleD = 2;
            int scaleH = 2;
            int scaleW = 2;
            SDVariable out = new Upsampling3d(sd, input, true, scaleD, scaleH, scaleW).outputVariable().std(true);
            out.markAsLoss();
            String err = OpValidation.validate(new TestCase(sd)
                    .gradientCheck(true));
            assertNull(err);
        }
    }
}
