/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.custom;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.custom.AdjustContrast;
import org.nd4j.linalg.api.ops.custom.AdjustHue;
import org.nd4j.linalg.api.ops.custom.AdjustSaturation;
import org.nd4j.linalg.api.ops.custom.BetaInc;
import org.nd4j.linalg.api.ops.custom.BitCast;
import org.nd4j.linalg.api.ops.custom.CompareAndBitpack;
import org.nd4j.linalg.api.ops.custom.DivideNoNan;
import org.nd4j.linalg.api.ops.custom.DrawBoundingBoxes;
import org.nd4j.linalg.api.ops.custom.FakeQuantWithMinMaxVarsPerChannel;
import org.nd4j.linalg.api.ops.custom.Flatten;
import org.nd4j.linalg.api.ops.custom.FusedBatchNorm;
import org.nd4j.linalg.api.ops.custom.HsvToRgb;
import org.nd4j.linalg.api.ops.custom.KnnMinDistance;
import org.nd4j.linalg.api.ops.custom.Lgamma;
import org.nd4j.linalg.api.ops.custom.LinearSolve;
import org.nd4j.linalg.api.ops.custom.Logdet;
import org.nd4j.linalg.api.ops.custom.Lstsq;
import org.nd4j.linalg.api.ops.custom.Lu;
import org.nd4j.linalg.api.ops.custom.MatrixBandPart;
import org.nd4j.linalg.api.ops.custom.Polygamma;
import org.nd4j.linalg.api.ops.custom.RandomCrop;
import org.nd4j.linalg.api.ops.custom.RgbToGrayscale;
import org.nd4j.linalg.api.ops.custom.RgbToHsv;
import org.nd4j.linalg.api.ops.custom.RgbToYiq;
import org.nd4j.linalg.api.ops.custom.RgbToYuv;
import org.nd4j.linalg.api.ops.custom.Roll;
import org.nd4j.linalg.api.ops.custom.ScatterUpdate;
import org.nd4j.linalg.api.ops.custom.ToggleBits;
import org.nd4j.linalg.api.ops.custom.TriangularSolve;
import org.nd4j.linalg.api.ops.custom.YiqToRgb;
import org.nd4j.linalg.api.ops.custom.YuvToRgb;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpStatus;
import org.nd4j.linalg.api.ops.impl.controlflow.Where;
import org.nd4j.linalg.api.ops.impl.image.CropAndResize;
import org.nd4j.linalg.api.ops.impl.image.NonMaxSuppression;
import org.nd4j.linalg.api.ops.impl.image.ResizeArea;
import org.nd4j.linalg.api.ops.impl.image.ResizeBilinear;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.reduce.MmulBp;
import org.nd4j.linalg.api.ops.impl.shape.Create;
import org.nd4j.linalg.api.ops.impl.shape.Linspace;
import org.nd4j.linalg.api.ops.impl.shape.OnesLike;
import org.nd4j.linalg.api.ops.impl.shape.SequenceMask;
import org.nd4j.linalg.api.ops.impl.transforms.Cholesky;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Qr;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.ModOp;
import org.nd4j.linalg.api.ops.random.compat.RandomStandardNormal;
import org.nd4j.linalg.api.ops.random.impl.DropOut;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Float.NaN;
import static org.junit.Assert.*;

/**
 * This class holds various CustomOps tests
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CustomOpsTests extends BaseNd4jTest {

    public CustomOpsTests(Nd4jBackend b){
        super(b);
    }

    @Override
    public char ordering(){
        return 'c';
    }

    @Test
    public void testNonInplaceOp1() {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);
        val arrayZ = Nd4j.create(10, 10);

        arrayX.assign(3.0);
        arrayY.assign(1.0);

        val exp = Nd4j.create(10,10).assign(4.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(arrayX, arrayY)
                .addOutputs(arrayZ)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayZ);
    }

    /**
     * This test works inplace, but without inplace declaration
     */
    @Test
    public void testNonInplaceOp2() {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);

        arrayX.assign(3.0);
        arrayY.assign(1.0);

        val exp = Nd4j.create(10,10).assign(4.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(arrayX, arrayY)
                .addOutputs(arrayX)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
    }

    @Test
    @Ignore // it's noop, we dont care anymore
    public void testNoOp1() {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(5, 3);

        arrayX.assign(3.0);
        arrayY.assign(1.0);

        val expX = Nd4j.create(10,10).assign(3.0);
        val expY = Nd4j.create(5,3).assign(1.0);

        CustomOp op = DynamicCustomOp.builder("noop")
                .addInputs(arrayX, arrayY)
                .addOutputs(arrayX, arrayY)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(expX, arrayX);
        assertEquals(expY, arrayY);
    }

    @Test
    public void testFloor() {
        val arrayX = Nd4j.create(10, 10);

        arrayX.assign(3.0);

        val exp = Nd4j.create(10,10).assign(3.0);

        CustomOp op = DynamicCustomOp.builder("floor")
                .addInputs(arrayX)
                .addOutputs(arrayX)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testInplaceOp1() {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);

        arrayX.assign(4.0);
        arrayY.assign(2.0);

        val exp = Nd4j.create(10,10).assign(6.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(arrayX, arrayY)
                .callInplace(true)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
    }

    @Test
    public void testNoneInplaceOp3() {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);

        arrayX.assign(4.0);
        arrayY.assign(2.0);

        val exp = Nd4j.create(10,10).assign(6.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(arrayX, arrayY)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, op.getOutputArgument(0));
    }

    @Test
    public void testNoneInplaceOp4() {
        val arrayX = Nd4j.create(DataType.INT, 10, 10);
        val arrayY = Nd4j.create(DataType.INT, 10, 10);

        arrayX.assign(4);
        arrayY.assign(2);

        val exp = Nd4j.create(DataType.INT,10, 10).assign(6);

        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(arrayX, arrayY)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        val res = op.getOutputArgument(0);
        assertEquals(DataType.INT, res.dataType());
        assertEquals(exp, res);
    }

    @Test
    public void testNoneInplaceOp5() {
        if (!Nd4j.isExperimentalMode())
            return;

        val arrayX = Nd4j.create(DataType.INT, 10, 10);
        val arrayY = Nd4j.create(DataType.FLOAT, 10, 10);

        arrayX.assign(4);
        arrayY.assign(2.0);

        val exp = Nd4j.create(DataType.FLOAT,10, 10).assign(6);

        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(arrayX, arrayY)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        val res = op.getOutputArgument(0);
        assertEquals(DataType.FLOAT, res.dataType());
        assertEquals(exp, res);
    }

    @Test
    public void testMergeMax1() {
        val array0 = Nd4j.create(new double[] {1, 0, 0, 0, 0});
        val array1 = Nd4j.create(new double[] {0, 2, 0, 0, 0});
        val array2 = Nd4j.create(new double[] {0, 0, 3, 0, 0});
        val array3 = Nd4j.create(new double[] {0, 0, 0, 4, 0});
        val array4 = Nd4j.create(new double[] {0, 0, 0, 0, 5});

        val z = Nd4j.create(DataType.DOUBLE, 5);
        val exp = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        CustomOp op = DynamicCustomOp.builder("mergemax")
                .addInputs(array0, array1, array2, array3, array4)
                .addOutputs(z)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, z);
    }

    @Test
    public void testMergeMaxF() {

        val array0 = Nd4j.rand('f', 5, 2).add(1); //some random array with +ve numbers
        val array1 = array0.dup('f').add(5);
        array1.put(0, 0, 0); //array1 is always bigger than array0 except at 0,0

        //expected value of maxmerge
        val exp = array1.dup('f');
        exp.putScalar(0, 0, array0.getDouble(0, 0));

        val zF = Nd4j.zeros(array0.shape(), 'f');
        CustomOp op = DynamicCustomOp.builder("mergemax")
                .addInputs(array0, array1)
                .addOutputs(zF)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, zF);
    }

    @Test
    public void testMergeMaxMixedOrder_Subtract() {
        val exp = Nd4j.create(new int[] {2, 2}, 'c').assign(5.0);
        Nd4j.getExecutioner().commit();

        val array0 = Nd4j.create(new int[] {2, 2}, 'f'); //some random array with +ve numbers
        val array1 = array0.dup('c').addi(5.0);

        Nd4j.getExecutioner().commit();

        assertEquals(exp, array1);
    }

    @Test
    public void testMergeMaxSameOrder_Subtract() {
        val exp = Nd4j.create(new int[] {2, 2}, 'c').assign(5.0);
        Nd4j.getExecutioner().commit();

        val array0 = Nd4j.create(new int[] {2, 2}, 'c'); //some random array with +ve numbers
        val array1 = array0.dup('c').addi(5);

        assertEquals(exp, array1);
    }

    @Test
    public void testMergeMaxMixedOrder() {
        val array0 = Nd4j.rand('f', 5, 2).addi(1); //some random array with +ve numbers
        val array1 = array0.dup('c').addi(5);
        array1.put(0, 0, 0); //array1 is always bigger than array0 except at 0,0

        //expected value of maxmerge
        val exp = array1.dup();
        exp.putScalar(0, 0, array0.getDouble(0, 0));

        val zF = Nd4j.zeros(array0.shape() ,'f');
        CustomOp op = DynamicCustomOp.builder("mergemax")
                .addInputs(array0, array1)
                .addOutputs(zF)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, zF);
    }


    @Test
    public void testOutputShapes1() {
        val array0 = Nd4j.rand('f', 5, 2).addi(1); //some random array with +ve numbers
        val array1 = array0.dup().addi(5);
        array1.put(0, 0, 0); //array1 is always bigger than array0 except at 0,0

        //expected value of maxmerge
        val exp = array1.dup();
        exp.putScalar(0, 0, array0.getDouble(0, 0));

        CustomOp op = DynamicCustomOp.builder("mergemax")
                .addInputs(array0, array1)
                .build();

        val shapes = Nd4j.getExecutioner().calculateOutputShape(op);

        assertEquals(1, shapes.size());
        assertArrayEquals(new long[]{5, 2}, shapes.get(0).getShape());
    }


    @Test
    public void testScatterUpdate1() {
        val matrix = Nd4j.create(5, 5);
        val updates = Nd4j.create(2, 5).assign(1.0);
        int[] dims = new int[]{1};
        int[] indices = new int[]{1, 3};

        val exp0 = Nd4j.create(5).assign(0);
        val exp1 = Nd4j.create(5).assign(1);

        ScatterUpdate op = new ScatterUpdate(matrix, updates, indices, dims, ScatterUpdate.UpdateOp.ADD);
        Nd4j.getExecutioner().exec(op);

        assertEquals(exp0, matrix.getRow(0));
        assertEquals(exp1, matrix.getRow(1));
        assertEquals(exp0, matrix.getRow(2));
        assertEquals(exp1, matrix.getRow(3));
        assertEquals(exp0, matrix.getRow(4));
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testScatterUpdate2() {
        val matrix = Nd4j.create(5, 5);
        val updates = Nd4j.create(2, 5).assign(1.0);
        int[] dims = new int[]{0};
        int[] indices = new int[]{0, 1};

        val exp0 = Nd4j.create(1, 5).assign(0);
        val exp1 = Nd4j.create(1, 5).assign(1);

        ScatterUpdate op = new ScatterUpdate(matrix, updates, indices, dims, ScatterUpdate.UpdateOp.ADD);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testScatterUpdate3() {
        val matrix = Nd4j.create(5, 5);
        val updates = Nd4j.create(2, 5).assign(1.0);
        int[] dims = new int[]{1};
        int[] indices = new int[]{0, 6};

        val exp0 = Nd4j.create(1, 5).assign(0);
        val exp1 = Nd4j.create(1, 5).assign(1);

        ScatterUpdate op = new ScatterUpdate(matrix, updates, indices, dims, ScatterUpdate.UpdateOp.ADD);
    }

    @Test
    public void testOpStatus1() {
        assertEquals(OpStatus.ND4J_STATUS_OK, OpStatus.byNumber(0));
    }

    @Test
    public void testRandomStandardNormal_1() {
        if (Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA)
            return;

        val shape = Nd4j.create(new float[] {5, 10});
        val op = new RandomStandardNormal(shape);

        Nd4j.getExecutioner().exec(op);

        assertEquals(1, op.outputArguments().size());
        val output = op.getOutputArgument(0);

        assertArrayEquals(new long[]{5, 10}, output.shape());
    }

    @Test
    public void testRandomStandardNormal_2() {
        if (Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA)
            return;

        val shape = new long[]{5, 10};
        val op = new RandomStandardNormal(shape);

        Nd4j.getExecutioner().exec(op);

        assertEquals(1, op.outputArguments().size());
        val output = op.getOutputArgument(0);

        assertArrayEquals(new long[]{5, 10}, output.shape());
    }

    @Test
    public void testOpContextExecution_1() {
        val arrayX = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5});
        val arrayY = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5});
        val arrayZ = Nd4j.create(DataType.FLOAT, 5);

        val exp = Nd4j.createFromArray(new float[]{2, 4, 6, 8, 10});

        val context = Nd4j.getExecutioner().buildContext();
        context.setInputArray(0, arrayX);
        context.setInputArray(1, arrayY);
        context.setOutputArray(0, arrayZ);

        val addOp = new AddOp();
        NativeOpsHolder.getInstance().getDeviceNativeOps().execCustomOp2(null, addOp.opHash(), context.contextPointer());

        assertEquals(exp, arrayZ);
    }

    @Test
    public void testOpContextExecution_2() {
        val arrayX = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5});
        val arrayY = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5});
        val arrayZ = Nd4j.create(DataType.FLOAT, 5);

        val exp = Nd4j.createFromArray(new float[]{2, 4, 6, 8, 10});

        val context = Nd4j.getExecutioner().buildContext();
        context.setInputArray(0, arrayX);
        context.setInputArray(1, arrayY);
        context.setOutputArray(0, arrayZ);

        val addOp = new AddOp();
        val output = Nd4j.exec(addOp, context);

        assertEquals(exp, arrayZ);
        assertTrue(arrayZ == output[0]);
    }

    @Test
    public void testOpContextExecution_3() {
        val arrayX = Nd4j.create(100);
        val arrayY = Nd4j.ones(100);
        val arrayZ = Nd4j.create(100);

        val exp = Nd4j.ones(100);

        val context = Nd4j.getExecutioner().buildContext();
        context.setInputArray(0, arrayX);
        context.setInputArray(1, arrayY);

        context.setOutputArray(0, arrayZ);

        val addOp = new AddOp();
        val output = Nd4j.exec(addOp, context);

        assertEquals(exp, arrayZ);
        assertTrue(arrayZ == output[0]);
    }

    @Test
    public void testFlatten_1() {
        val arrayA = Nd4j.createFromArray(1.f, 2.f, 3.f);
        val arrayB = Nd4j.createFromArray(4.f, 5.f, 6.f);
        val arrayC = Nd4j.createFromArray(7.f, 8.f, 9.f);

        val exp = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f);

        val result = Nd4j.exec(new Flatten('c', arrayA, arrayB, arrayC))[0];

        assertEquals(exp, result);
    }

    @Test
    public void testMatmulBp() {
        val a = Nd4j.create(DataType.DOUBLE, 1,3);
        val b = Nd4j.create(DataType.DOUBLE, 1,4);
        val gI = Nd4j.create(DataType.DOUBLE, 3,4);

        val gA = Nd4j.create(DataType.DOUBLE, 1,3);
        val gB = Nd4j.create(DataType.DOUBLE, 1,4);

        val mt = MMulTranspose.builder()
                .transposeA(true)
                .transposeB(false)
                .transposeResult(false).build();

        val op = new MmulBp(a, b, gI, gA, gB, mt);
        Nd4j.exec(op);
    }

    @Test
    public void testStridedSliceEdgeCase(){
        INDArray in = Nd4j.scalar(10.0).reshape(1);   //Int [1]
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
                //.addOutputs(Nd4j.empty(DataType.INT))
                .build();

        List<LongShapeDescriptor> l = op.calculateOutputShape();
        assertEquals(1, l.size());
        assertEquals(DataType.DOUBLE, l.get(0).dataType());
        assertTrue(l.get(0).isEmpty()); //Should be empty array, is rank 0 scalar

        Nd4j.exec(op);  //Execution is OK
    }



    @Test
    public void testDepthwise(){
        INDArray input = Nd4j.create(DataType.DOUBLE, 1,3,8,8);
        INDArray depthwiseWeight = Nd4j.create(DataType.DOUBLE, 1,1,3,2);
        INDArray bias = Nd4j.create(DataType.DOUBLE, 1, 6);

        INDArray[] inputs = new INDArray[]{input, depthwiseWeight, bias};

        int[] args = {1, 1, 1, 1, 0, 0, 1, 1, 0};

        INDArray output = Nd4j.create(DataType.DOUBLE, 1, 6, 8, 8);

        CustomOp op = DynamicCustomOp.builder("depthwise_conv2d")
                .addInputs(inputs)
                .addIntegerArguments(args)
                .addOutputs(output)
                .callInplace(false)
                .build();

        for( int i=0; i<1000; i++ ) {
//            System.out.println(i);
            Nd4j.getExecutioner().exec(op);
        }
    }

    @Test
    public void testMod_1() {
        val x = Nd4j.createFromArray(5.f, 6.f, 7.f);
        val y = Nd4j.scalar(4.f);
        val e = Nd4j.createFromArray(1.f, 2.f, 3.f);

        val z = Nd4j.exec(new ModOp(new INDArray[]{x, y}, new INDArray[]{}))[0];

        assertEquals(e, z);
    }

    @Test
    public void testScalarVector_edge_1() {
        val x = Nd4j.scalar(2.0f);
        val y = Nd4j.createFromArray(new float[]{2.0f});
        val e = Nd4j.createFromArray(new float[]{4.0f});

        val z = Nd4j.exec(new AddOp(new INDArray[]{x, y}, new INDArray[]{}))[0];

        assertTrue(Shape.shapeEquals(e.shape(), z.shape()));
        assertEquals(e, z);
    }

    @Test
    public void testScalarVector_edge_2() {
        val x = Nd4j.scalar(2.0f);
        val y = Nd4j.createFromArray(new float[]{2.0f});
        val e = Nd4j.createFromArray(new float[]{4.0f});

        val z = Nd4j.exec(new AddOp(new INDArray[]{y, x}, new INDArray[]{}))[0];

        assertTrue(Shape.shapeEquals(e.shape(), z.shape()));
        assertEquals(e, z);
    }

    @Test(expected = RuntimeException.class)
    public void testInputValidationMergeMax(){
        INDArray[] inputs = new INDArray[]{
                Nd4j.createFromArray(0.0f, 1.0f, 2.0f).reshape('c', 1, 3),
                Nd4j.createFromArray(1.0f).reshape('c', 1, 1)};

        INDArray out = Nd4j.create(DataType.FLOAT, 1, 3).assign(Double.NaN);
        CustomOp op = DynamicCustomOp.builder("mergemax")
                .addInputs(inputs)
                .addOutputs(out)
                .callInplace(false)
                .build();

        Nd4j.exec(op);
//        System.out.println(out);
    }

    @Test
    public void testUpsampling2dBackprop(){

        Nd4j.getRandom().setSeed(12345);
        int c = 2;
        int[] sz = {2,2};
        long[] inSize = {1, c, 3, 3};
        INDArray eps = Nd4j.rand(DataType.FLOAT, 1, c, sz[0] * inSize[2], sz[1] * inSize[3]);

        INDArray input = Nd4j.create(inSize);    //Unused, not sure why this is even an arg...
        INDArray exp = Nd4j.create(DataType.FLOAT, inSize);

        for( int ch=0; ch<c; ch++ ) {
            for( int h=0; h<eps.size(2); h++ ){
                for( int w=0; w<eps.size(3); w++ ){
                    int[] from = new int[]{0, ch, h, w};
                    int[] to = new int[]{0, ch, h/sz[0], w/sz[1]};
                    float add = eps.getFloat(from);
                    float current = exp.getFloat(to);
                    exp.putScalar(to, current + add);
                }
            }
        }

//        System.out.println("Eps:");
//        System.out.println(eps.shapeInfoToString());
//        System.out.println(Arrays.toString(eps.data().asFloat()));

//        System.out.println("Expected:");
//        System.out.println(exp.shapeInfoToString());
//        System.out.println(Arrays.toString(exp.data().asFloat()));

        DynamicCustomOp op = DynamicCustomOp.builder("upsampling2d_bp")
                .addInputs(input, eps)
                .addOutputs(exp.ulike())
                .addIntegerArguments(1) //1 = NCHW
                .build();

        Nd4j.exec(op);

        INDArray act = op.getOutputArgument(0);
        assertEquals(exp, act);
    }

    @Test
    public void testIsMaxView(){
        INDArray predictions = Nd4j.rand(DataType.FLOAT, 3, 4, 3, 2);

        INDArray row = predictions.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0));
        row = row.reshape(1, row.length());
        assertArrayEquals(new long[]{1, 4}, row.shape());

        val result1 = row.ulike();
        val result2 = row.ulike();

        Nd4j.exec(new IsMax(row.dup(), result1, 1));        //OK
        Nd4j.exec(new IsMax(row, result2, 1));              //C++ exception

        assertEquals(result1, result2);
    }

    @Test
    public void isMax4d_2dims(){
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(DataType.FLOAT, 3, 3, 4, 4).permute(0, 2, 3, 1);

        INDArray out_permutedIn = in.like();
        INDArray out_dupedIn = in.like();

        Nd4j.exec(new IsMax(in.dup(), out_dupedIn, 2, 3));
        Nd4j.exec(new IsMax(in, out_permutedIn, 2, 3));

        assertEquals(out_dupedIn, out_permutedIn);
    }

    @Test
    public void testSizeTypes(){
        List<DataType> failed = new ArrayList<>();
        for(DataType dt : new DataType[]{DataType.LONG, DataType.INT, DataType.SHORT, DataType.BYTE,
                DataType.UINT64, DataType.UINT32, DataType.UINT16, DataType.UBYTE,
                DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.BFLOAT16}) {

            INDArray in = Nd4j.create(DataType.FLOAT, 100);
            INDArray out = Nd4j.scalar(dt, 0);
            INDArray e = Nd4j.scalar(dt, 100);

            DynamicCustomOp op = DynamicCustomOp.builder("size")
                    .addInputs(in)
                    .addOutputs(out)
                    .build();

            try {
                Nd4j.exec(op);

                assertEquals(e, out);
            } catch (Throwable t){
                failed.add(dt);
            }
        }

        if(!failed.isEmpty()){
            fail("Failed datatypes: " + failed.toString());
        }
    }

    @Test
    public void testListDiff(){
        INDArray x = Nd4j.createFromArray(0, 1, 2, 3);
        INDArray y = Nd4j.createFromArray(3, 1);

        INDArray out = Nd4j.create(DataType.INT, 2);
        INDArray outIdx = Nd4j.create(DataType.INT, 2);

        Nd4j.exec(DynamicCustomOp.builder("listdiff")
                .addInputs(x, y)
                .addOutputs(out, outIdx)
                .build());

        INDArray exp = Nd4j.createFromArray(0, 2);

        assertEquals(exp, out);         //Values in x not in y
        assertEquals(exp, outIdx);      //Indices of the values in x not in y
    }

    @Test
    public void testTopK1(){
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
    public void testMaxPool2Dbp_1() {
        val x = Nd4j.create(DataType.HALF, 2,3,16,16).assign(Double.NaN);
        val y = Nd4j.create(DataType.HALF, 2,3,8,8).assign(Double.NaN);
        val z = Nd4j.create(DataType.HALF, 2,3,16,16);

        val op = DynamicCustomOp.builder("maxpool2d_bp")
                .addInputs(x, y)
                .addOutputs(z)
                .addIntegerArguments(2, 2, 2, 2, 8,8, 1,1,1, 0,0)
                .build();

        Nd4j.exec(op);
        Nd4j.getExecutioner().commit();
    }

    @Test
    public void test() throws Exception {

        INDArray in1 = Nd4j.create(DataType.BFLOAT16, 2, 3, 10, 1);//Nd4j.createFromArray(0.2019043,0.6464844,0.9116211,0.60058594,0.34033203,0.7036133,0.6772461,0.3815918,0.87353516,0.04650879,0.67822266,0.8618164,0.88378906,0.7573242,0.66796875,0.63427734,0.33764648,0.46923828,0.62939453,0.76464844,-0.8618164,-0.94873047,-0.9902344,-0.88916016,-0.86572266,-0.92089844,-0.90722656,-0.96533203,-0.97509766,-0.4975586,-0.84814453,-0.984375,-0.98828125,-0.95458984,-0.9472656,-0.91064453,-0.80859375,-0.83496094,-0.9140625,-0.82470703,0.4802246,0.45361328,0.28125,0.28320312,0.79345703,0.44604492,-0.30273438,0.11730957,0.56396484,0.73583984,0.1418457,-0.44848633,0.6923828,-0.40234375,0.40185547,0.48632812,0.14538574,0.4638672,0.13000488,0.5058594)
                //.castTo(DataType.BFLOAT16).reshape(2,3,10,1);
        INDArray in2 = Nd4j.create(DataType.BFLOAT16, 2, 3, 10, 1); //Nd4j.createFromArray(0.0,-0.13391113,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.1751709,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.51904297,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5107422,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
                //.castTo(DataType.BFLOAT16).reshape(2,3,10,1);

        INDArray out = in1.ulike();

        Nd4j.exec(DynamicCustomOp.builder("maxpool2d_bp")
                .addInputs(in1, in2)
                .addOutputs(out)
                .addIntegerArguments(5,1,1,2,2,0,1,1,1,0,0)
                .build());

        Nd4j.getExecutioner().commit();
    }

    @Test
    public void testAdjustContrast() {
        INDArray in = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 4*4*3).reshape(4,4,3);
        INDArray out = Nd4j.zeros(DataType.DOUBLE,4, 4, 3);

        INDArray expected = Nd4j.createFromArray(new double[]{-21.5, -20.5, -19.5,  -15.5, -14.5, -13.5,  -9.5,  -8.5,  -7.5,  -3.5,  -2.5,  -1.5,
                2.5,   3.5,   4.5,    8.5,   9.5,  10.5,  14.5,  15.5,  16.5,  20.5,  21.5,  22.5,
                26.5,  27.5,  28.5,   32.5,  33.5,  34.5,  38.5,  39.5,  40.5,  44.5,  45.5,  46.5,
                50.5,  51.5,  52.5,   56.5,  57.5,  58.5,  62.5,  63.5,  64.5,  68.5,  69.5,  70.5
        }).reshape(4,4,3);
        Nd4j.exec(new AdjustContrast(in, 2.0, out));

        assertArrayEquals(out.shape(), in.shape());
        assertEquals(expected, out);
    }

    @Test
    public void testAdjustContrastShape(){
        DynamicCustomOp op = DynamicCustomOp.builder("adjust_contrast_v2")
                .addInputs(Nd4j.create(DataType.FLOAT, 256, 256,3), Nd4j.scalar(0.5f))
                .build();
        List<LongShapeDescriptor> lsd = op.calculateOutputShape();
        assertEquals(1, lsd.size());
        assertArrayEquals(new long[]{256, 256, 3}, lsd.get(0).getShape());
    }


    @Test
    public void testBitCastShape(){
        INDArray out = Nd4j.createUninitialized(1,10);
        BitCast op = new BitCast(Nd4j.zeros(1,10), DataType.FLOAT.toInt(), out);
        List<LongShapeDescriptor> lsd = op.calculateOutputShape();
        assertEquals(1, lsd.size());
        assertArrayEquals(new long[]{1,10}, lsd.get(0).getShape());
    }

    @Test
    public void testAdjustSaturation() {
        INDArray in = Nd4j.createFromArray(new double[]{50,100,78, 118.5,220,112.5,190,163.5,230, 255,128.5,134}).reshape(2,2,3);
        INDArray out = Nd4j.create(in.shape());
        INDArray expected = Nd4j.createFromArray(new double[]{0,100,56, 17,220,5, 150,97,230, 255,2,13}).reshape(2,2,3);

        Nd4j.exec(new AdjustSaturation(in, 2.0, out));
        assertEquals(expected, out);
    }

    @Test
    public void testAdjustHue() {
        INDArray in = Nd4j.createFromArray(new double[]{0,100,56, 17,220,5, 150,97,230, 255,2,13}).reshape(2,2,3);
        INDArray out = Nd4j.create(in.shape());
        INDArray expected = Nd4j.createFromArray(new double[]{100,0,44, 208,5,220, 177,230,97, 2,255,244}).reshape(2,2,3);

        Nd4j.exec(new AdjustHue(in, 0.5, out));
        assertEquals(expected, out);
    }

    @Test
    public void testBitCast() {
        INDArray in = Nd4j.linspace(DataType.FLOAT, 1.0f, 1.0f, 8).reshape(2,2,2);
        INDArray out = Nd4j.createUninitialized(2,2);

        Nd4j.exec(new BitCast(in, DataType.DOUBLE.toInt(), out));

        INDArray expected = Nd4j.createFromArray(new double[]{2., 512., 8192., 131072.032 }).reshape(2,2);
        assertArrayEquals(new long[]{2,2}, out.shape());
        assertEquals(expected, out);
    }

    @Test
    public void testDrawBoundingBoxesShape() {
        INDArray images = Nd4j.createFromArray(new float[]{0.7788f, 0.8012f, 0.7244f,  0.2309f, 0.7271f,
                        0.1804f,0.5056f,0.8925f,0.5461f,0.9234f,0.0856f,0.7938f,0.6591f,0.5555f,0.1596f,
                        0.3087f,0.1548f,0.4695f,0.9939f,0.6113f,0.6765f,0.1800f,0.6750f,0.2246f,0.0509f,
                        0.4601f,0.8284f,0.2354f,0.9752f,0.8361f,0.2585f,0.4189f,0.7028f,0.7679f,0.5373f,
                        0.7234f,0.2690f,0.0062f,0.0327f,0.0644f,0.8428f,0.7494f,0.0755f,0.6245f,0.3491f,
                        0.5793f,0.5730f,0.1822f,0.6420f,0.9143f}).reshape(2,5,5,1);
        INDArray boxes = Nd4j.createFromArray(new float[]{0.7717f,    0.9281f,    0.9846f,    0.4838f,
                                                          0.6433f,    0.6041f,    0.6501f,    0.7612f,
                                                          0.7605f,    0.3948f,    0.9493f,    0.8600f,
                                                          0.7876f,    0.8945f,    0.4638f,    0.7157f}).reshape(2,2,4);
        INDArray colors = Nd4j.createFromArray(new float[]{0.9441f, 0.5957f}).reshape(1,2);
        INDArray output = Nd4j.create(DataType.FLOAT, images.shape());
        val op = new DrawBoundingBoxes(images, boxes, colors, output);
        Nd4j.exec(op);
        INDArray expected = Nd4j.createFromArray(new float[]{0.7788f, 0.8012f, 0.7244f, 0.2309f, 0.7271f,
                           0.1804f, 0.5056f, 0.8925f, 0.5461f, 0.9234f, 0.0856f, 0.7938f, 0.9441f,
                           0.9441f, 0.1596f, 0.3087f, 0.1548f, 0.4695f, 0.9939f, 0.6113f, 0.6765f,
                           0.1800f, 0.6750f, 0.2246f, 0.0509f, 0.4601f, 0.8284f, 0.2354f, 0.9752f, 0.8361f,
                0.2585f, 0.4189f,0.7028f,0.7679f,0.5373f,0.7234f,0.2690f,0.0062f,0.0327f,0.0644f,
               0.8428f, 0.9441f,0.9441f,0.9441f,0.3491f,0.5793f,0.5730f,0.1822f,0.6420f,0.9143f});
        assertEquals(expected, output);
    }

    @Test
    public void testFakeQuantAgainstTF_1() {
        INDArray x = Nd4j.createFromArray(new float[]{ 0.7788f,    0.8012f,    0.7244f,    0.2309f,    0.7271f,
     0.1804f,    0.5056f,    0.8925f,    0.5461f,    0.9234f,
     0.0856f,    0.7938f,    0.6591f,    0.5555f,    0.1596f}).reshape(3,5);
        INDArray min = Nd4j.createFromArray(new float[]{ -0.2283f,   -0.0719f,   -0.0154f,   -0.5162f,   -0.3567f});
        INDArray max = Nd4j.createFromArray(new float[]{ 0.9441f,    0.5957f,    0.8669f,    0.3502f,    0.5100f});

        INDArray expected = Nd4j.createFromArray(new float[]{0.7801f,    0.5966f,    0.7260f,    0.2320f,    0.5084f,
             0.1800f,    0.5046f,    0.8684f,    0.3513f,    0.5084f,
             0.0877f,    0.5966f,    0.6600f,    0.3513f,    0.1604f}).reshape(3,5);

        INDArray out = Nd4j.createUninitialized(x.shape());
        val op = new FakeQuantWithMinMaxVarsPerChannel(x,min,max);
        Nd4j.exec(op);
        assertEquals(expected, out);
    }

    @Test
    public void testWhereFail() {
        INDArray in = Nd4j.createFromArray(new float[]{0f,    1.0000f,    1.0000f,    1.0000f,    1.0000f});
        INDArray out = Nd4j.createUninitialized(4,1);
        INDArray expected = Nd4j.createFromArray(4,1);
        val op = new Where(new INDArray[]{in}, new INDArray[]{out});
        Nd4j.exec(op);
        assertArrayEquals(new long[]{4,1} , out.shape());
    }

    @Test
    public void testResizeBilinear1() {

        INDArray x = Nd4j.rand(1, 2,3,4);
        INDArray z = Nd4j.createUninitialized(x.shape());
        boolean align = false;
        val op = new ResizeBilinear(x, z, 10, 10, align, false);
        Nd4j.exec(op);
    }

    @Test
    public void testResizeArea1() {

        INDArray x = Nd4j.rand(DataType.FLOAT, 1, 2,3,4);
        INDArray z = Nd4j.createUninitialized(DataType.FLOAT, 1, 10, 10, 4);
        ResizeArea op = new ResizeArea(x, z, 10, 10, false);
        Nd4j.exec(op);
    }

    @Test
    public void testResizeArea2() {

        INDArray image = Nd4j.linspace(DataType.FLOAT, 1.0f, 1.0f, 9 ).reshape(1,3,3,1);
        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, 1, 6, 6, 1);
        INDArray expected = Nd4j.createFromArray(new float[]{
                1.f, 1.f, 2.f, 2.f, 3.f, 3.f,
                1.f, 1.f, 2.f, 2.f, 3.f, 3.f,
                4.f, 4.f, 5.f, 5.f, 6.f, 6.f,
                4.f, 4.f, 5.f, 5.f, 6.f, 6.f,
                7.f, 7.f, 8.f, 8.f, 9.f, 9.f,
                7.f, 7.f, 8.f, 8.f, 9.f, 9.f
        }).reshape(1,6,6,1);
        ResizeArea op = new ResizeArea(image, output, 6, 6, false);
        Nd4j.exec(op);
        assertEquals(expected, output);
    }

    @Test
    public void testCompareAndBitpack() {
        INDArray in = Nd4j.createFromArray(new double[]{-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f,
                -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f}).reshape( 2,3,4);
        INDArray out = Nd4j.createUninitialized(DataType.UBYTE, 2,3,4);
        INDArray expected = Nd4j.createFromArray(new byte[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}).
                reshape(2,3,4);

        Nd4j.exec(new CompareAndBitpack(in ,2.0, out));
        assertArrayEquals(new long[]{2,3,4}, out.shape());
    }

    @Test
    public void testDivideNoNan() {
        INDArray in1 = Nd4j.rand(DataType.DOUBLE, 2,3,4);
        INDArray in2 = Nd4j.rand(DataType.DOUBLE, 2,3,4);
        INDArray out = Nd4j.createUninitialized(DataType.DOUBLE, 2,3,4);

        Nd4j.exec(new DivideNoNan(in1, in2, out));
        assertArrayEquals(new long[]{2,3,4}, out.shape());
    }

    @Test
    public void testDrawBoundingBoxes() {
        INDArray images = Nd4j.linspace(DataType.FLOAT, 1.0f, 1.0f, 2*4*5*3).reshape(2,4,5,3);
        INDArray boxes = Nd4j.createFromArray(new float[]{ 0.0f , 0.0f , 1.0f , 1.0f,
                0.1f, 0.2f, 0.9f, 0.8f,
                0.3f, 0.3f, 0.7f, 0.7f,
                0.4f, 0.4f, 0.6f, 0.6f}).reshape(2,2,4);
        INDArray colors = Nd4j.createFromArray(new float[]{
                201.0f, 202.0f, 203.0f, 127.0f, 128.0f, 129.0f}).
                reshape(2,3);
        INDArray output = Nd4j.create(DataType.FLOAT, images.shape());
        INDArray expected = Nd4j.createFromArray(new float[]{127.f, 128.f, 129.f,    127.f, 128.f, 129.f,    127.f, 128.f, 129.f,
                127.f, 128.f, 129.f,    201.f, 202.f, 203.f,
                127.f, 128.f,  129.f,    19.f,  20.f,  21.f,     22.f,  23.f,  24.f,    127.f, 128.f, 129.f,    201.f, 202.f, 203.f,
                127.f, 128.f,  129.f,   127.f, 128.f, 129.f,    127.f, 128.f, 129.f,    127.f, 128.f, 129.f,    201.f, 202.f, 203.f,
                201.f, 202.f,  203.f,    201.f ,202.f ,203.f,   201.f, 202.f, 203.f,    201.f, 202.f, 203.f,    201.f, 202.f, 203.f,

                61.f,  62.f,   63.f,    201.f, 202.f, 203.f,    201.f, 202.f, 203.f,     70.f,  71.f,  72.f,     73.f,  74.f,  75.f,
                76.f,  77.f,   78.f,    127.f, 128.f, 129.f,    127.f, 128.f, 129.f,     85.f,  86.f,  87.f,     88.f,  89.f,  90.f,
                91.f,  92.f,   93.f,    201.f, 202.f, 203.f,    201.f, 202.f, 203.f,    100.f, 101.f, 102.f,    103.f, 104.f, 105.f,
                106.f, 107.f,  108.f,    109.f, 110.f, 111.f,    112.f, 113.f, 114.f,    115.f, 116.f, 117.f,    118.f, 119.f, 120.f}).
                reshape(2,4,5,3);

        Nd4j.exec(new DrawBoundingBoxes(images, boxes, colors, output));

        assertArrayEquals(images.shape(), output.shape());
        assertEquals(expected, output);
    }

    @Test
    public void FakeQuantWithMinMaxVarsPerChannel() {

        INDArray x = Nd4j.createFromArray(new float[]{-63.80f, -63.75f, -63.4f, -63.5f, 0.0f, 0.1f}).
                reshape(1,2,3,1);

        INDArray min = Nd4j.createFromArray(new float[]{-63.65f});
        INDArray max = Nd4j.createFromArray(new float[]{0.1f});

        INDArray expected = Nd4j.createFromArray(new float[]{-63.75f, -63.75f, -63.5f, -63.5f, 0.f, 0.f}).
                reshape(1,2,3,1);

        INDArray[] output = Nd4j.exec(new FakeQuantWithMinMaxVarsPerChannel(x,min,max));

        assertEquals(expected, output[0]);
    }

    @Test
    public void testKnnMinDistance() {
        INDArray point = Nd4j.rand(DataType.FLOAT, 1, 20);
        INDArray lowest = Nd4j.rand(DataType.FLOAT, 1, 20);
        INDArray highest = Nd4j.rand(DataType.FLOAT, 1, 20);
        INDArray distance = Nd4j.scalar(0.f);

        Nd4j.exec(new KnnMinDistance(point, lowest, highest, distance));
//        System.out.println(distance);
    }

    @Ignore("2019/11/15 AS - https://github.com/eclipse/deeplearning4j/issues/8399")
    @Test
    public void testCropAndResize() {
        INDArray image = Nd4j.createUninitialized(DataType.FLOAT, 1, 2, 2, 1);
        INDArray boxes = Nd4j.createFromArray(new float[]{1,2,3,4}).reshape(1,4);
        INDArray box_indices = Nd4j.createFromArray(new int[]{1});
        INDArray crop_size = Nd4j.createFromArray(new int[]{1,2}).reshape(1,2);

        //Output shape mismatch - TF [2, 2, 1, 1] vs SD: [1, 2, 1, 1]
        INDArray output = Nd4j.create(DataType.FLOAT, 2,2,1,1);


        Nd4j.exec(new CropAndResize(image, boxes, box_indices, crop_size, CropAndResize.Method.BILINEAR, 0.5,
                 output));
    }

    @Test
    public void testLayersDropoutFail() {
        INDArray input = Nd4j.rand(4, 5);
        INDArray output = Nd4j.createUninitialized(4, 5);
        DropOut op = new DropOut(input, output, 0.1);
        Nd4j.exec(op);
//        System.out.println(output);
    }

    @Test
    public void testRange(){
        DynamicCustomOp op = DynamicCustomOp.builder("range")
                .addFloatingPointArguments(-1.0, 1.0, 0.01)
                .build();

        List<LongShapeDescriptor> lsd = op.calculateOutputShape();
        //System.out.println("Calculated output shape: " + Arrays.toString(lsd.get(0).getShape()));
        op.setOutputArgument(0, Nd4j.create(lsd.get(0)));

        Nd4j.exec(op);
    }

    @Test
    public void testBitCastShape_1(){
        val out = Nd4j.createUninitialized(1,10);
        BitCast op = new BitCast(Nd4j.zeros(DataType.FLOAT,1,10), DataType.INT.toInt(), out);
        List<LongShapeDescriptor> lsd = op.calculateOutputShape();
        assertEquals(1, lsd.size());
        assertArrayEquals(new long[]{1,10}, lsd.get(0).getShape());
    }

    @Test
    public void testBitCastShape_2(){
        val out = Nd4j.createUninitialized(1,10);
        BitCast op = new BitCast(Nd4j.zeros(DataType.DOUBLE,1,10), DataType.INT.toInt(), out);
        List<LongShapeDescriptor> lsd = op.calculateOutputShape();
        assertEquals(1, lsd.size());
        assertArrayEquals(new long[]{1,10, 2}, lsd.get(0).getShape());
    }

    @Test
    public void testFusedBatchNorm() {
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 2*2*3*4).reshape(2,2,3,4);
        INDArray scale = Nd4j.create(DataType.DOUBLE, 4);
        scale.assign(0.5);
        INDArray offset = Nd4j.create(DataType.DOUBLE, 4);
        offset.assign(2.0);

        INDArray y = Nd4j.createUninitialized(DataType.DOUBLE, x.shape());
        INDArray batchMean = Nd4j.create(4);
        INDArray batchVar = Nd4j.create(4);

        FusedBatchNorm op = new FusedBatchNorm(x,scale,offset,0,1,
                                                y, batchMean, batchVar);

        INDArray expectedY = Nd4j.createFromArray(new double[]{1.20337462,  1.20337462,  1.20337462,
                1.20337462, 1.34821558,  1.34821558,  1.34821558,  1.34821558, 1.49305654,  1.49305654,
                1.49305654,  1.49305654, 1.63789749,  1.63789749,  1.63789749,  1.63789749, 1.78273857,
                1.78273857,  1.78273857,  1.78273857, 1.92757952,  1.92757952,  1.92757952,  1.92757952,
                2.0724206 ,  2.0724206 ,  2.0724206 ,  2.0724206 , 2.21726155,  2.21726155,  2.21726155,
                2.21726155, 2.36210251,  2.36210251,  2.36210251,  2.36210251, 2.50694346,  2.50694346,
                2.50694346,  2.50694346, 2.65178442,  2.65178442,  2.65178442,  2.65178442, 2.79662538,
                2.79662538,  2.79662538,  2.79662538}).reshape(x.shape());
        INDArray expectedBatchMean = Nd4j.createFromArray(new double[]{23.,  24.,  25.,  26.});
        INDArray expectedBatchVar = Nd4j.createFromArray(new double[]{208.00001526,  208.00001526,  208.00001526,  208.00001526});
        Nd4j.exec(op);
        assertArrayEquals(expectedY.shape(), y.shape());
        assertArrayEquals(expectedBatchMean.shape(), batchMean.shape());
        assertArrayEquals(expectedBatchVar.shape(), batchVar.shape());
    }

    @Test
    public void testFusedBatchNorm1() {
        INDArray x = Nd4j.createFromArray(new float[]{0.7788f,0.8012f,0.7244f,0.2309f,
            0.7271f, 0.1804f, 0.5056f, 0.8925f,
            0.5461f, 0.9234f, 0.0856f, 0.7938f,
            0.6591f, 0.5555f, 0.1596f, 0.3087f,
            0.1548f, 0.4695f, 0.9939f, 0.6113f,
            0.6765f, 0.1800f, 0.6750f, 0.2246f}).reshape(1,2,3,4);
        INDArray scale = Nd4j.createFromArray(new float[]{ 0.7717f, 0.9281f, 0.9846f, 0.4838f});
        INDArray offset = Nd4j.createFromArray(new float[]{0.9441f, 0.5957f, 0.8669f, 0.3502f});

        INDArray y = Nd4j.createUninitialized(DataType.DOUBLE, x.shape());
        INDArray batchMean = Nd4j.create(4);
        INDArray batchVar = Nd4j.create(4);

        FusedBatchNorm op = new FusedBatchNorm(x,scale,offset,0,1,
                y, batchMean, batchVar);

        INDArray expectedY = Nd4j.createFromArray(new float[]{1.637202024f, 1.521406889f, 1.48303616f, -0.147269756f,
              1.44721508f,  -0.51030159f,  0.810390055f,     1.03076458f,
              0.781284988f, 1.921229601f,  -0.481337309f,    0.854952335f,
              1.196854949f, 0.717398405f,  -0.253610134f,    -0.00865117f,
              -0.658405781f,0.43602103f,   2.311818838f,    0.529999137f,
              1.260738254f, -0.511638165f, 1.331095099f,   -0.158477545f}).reshape(x.shape());
        Nd4j.exec(op);
        assertArrayEquals(expectedY.shape(), y.shape());
    }

    @Test
    public void testFusedBatchNormHalf() {
        INDArray x = Nd4j.create(DataType.HALF, 1,2,3,4);
        //INDArray scale = Nd4j.createFromArray(new float[]{0.7717f, 0.9281f, 0.9846f, 0.4838f});
        //INDArray offset = Nd4j.createFromArray(new float[]{0.9441f, 0.5957f, 0.8669f, 0.3502f});
        INDArray scale = Nd4j.create(DataType.HALF, 4);
        INDArray offset = Nd4j.create(DataType.HALF, 4);

        INDArray y = Nd4j.createUninitialized(DataType.HALF, x.shape());
        INDArray batchMean = Nd4j.create(4);
        INDArray batchVar = Nd4j.create(4);

        FusedBatchNorm op = new FusedBatchNorm(x, scale, offset, 0, 1,
                y, batchMean, batchVar);
        Nd4j.exec(op);
    }

    @Test
    public void testMatrixBandPart() {
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 2*3*3).reshape(2,3,3);
        val op = new MatrixBandPart(x,1,1);
        INDArray expected = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 2*3*3).reshape(2,3,3);
        /*expected.putScalar(0, 0, 2, 0.);
        expected.putScalar(1, 0, 2, 0.);
        expected.putScalar(0, 2, 0, 0.);
        expected.putScalar(1, 2, 0, 0.);*/

        INDArray[] out = Nd4j.exec(op);
        assertEquals(expected, x);
    }

    @Ignore("AS failed 2019/12/04")
    @Test
    public void testPolygamma() {
        INDArray n = Nd4j.linspace(DataType.FLOAT, 1.0, 1.0, 9).reshape(3,3);
        INDArray x = Nd4j.create(DataType.FLOAT, 3,3);
        x.assign(0.5);
        INDArray expected = Nd4j.createFromArray(new float[]{4.934802f, -16.828796f, 97.409088f, -771.474243f,
                7691.113770f, -92203.460938f, 1290440.250000f, -20644900.000000f, 3.71595e+08f}).reshape(3,3);
        INDArray output = Nd4j.create(DataType.FLOAT, expected.shape());
        val op = new Polygamma(x,n,output);
        Nd4j.exec(op);
        assertEquals(expected, output);
    }

    @Test
    public void testLgamma() {
        INDArray x = Nd4j.createFromArray(new double[]{0.1, 0.5, 0.7, 1.5, 1.7, 2.0, 2.5, 2.7, 3.}).reshape(3,3);
        INDArray expected = Nd4j.createFromArray(new double[]{
                2.2527127 ,  0.5723649 ,  0.26086727,
                -0.12078223, -0.09580769,        0.,
                0.28468287,  0.4348206 ,  0.6931472
        }).reshape(3,3);
        INDArray[] ret = Nd4j.exec(new Lgamma(x));
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testRandomCrop() {
        INDArray x = Nd4j.createFromArray(new double[]{1.8, 2.5,  4.,  9., 2.1, 2.4,  3.,  9.,2.1, 2.1, 0.7, 0.1,3., 4.2, 2.2, 1. }).reshape(2,2,4);
        INDArray shape = Nd4j.createFromArray(new int[] {1,2,3});
        val op = new RandomCrop(x, shape);
        INDArray[] res = Nd4j.exec(op);
    }

    @Test
    public void testRoll() {
        INDArray x = Nd4j.createFromArray(new double[]{    11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42,     12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42,
                21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42,     22.11, 22.12, 22.21, 22.22, 22.31, 22.32, 22.41, 22.42}).
                reshape(2,2,4,2);

        INDArray expected = Nd4j.createFromArray(new double[]{    22.21, 22.22, 22.31, 22.32, 22.41, 22.42, 11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42,
                12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42, 21.11, 21.12, 21.21, 21.22, 21.31, 21.32,
                21.41, 21.42, 22.11, 22.12
        }).reshape(x.shape());
        val op = new Roll(x, 6);
        INDArray[] res = Nd4j.exec(op);
        assertEquals(expected, res[0]);
    }

    @Test
    public void testToggleBits() {
        INDArray input = Nd4j.createFromArray(new int[]{2,2});
        INDArray expected = Nd4j.createFromArray(new int[]{-3,-3});
        ToggleBits op = new ToggleBits(input);
        val result = Nd4j.exec(op);
        assertEquals(expected, result[0]);
    }

    @Ignore("AS 11.28.2019 - https://github.com/eclipse/deeplearning4j/issues/8449")
    @Test
    public void testNonMaxSuppression() {
        INDArray boxes = Nd4j.createFromArray(new float[] {0.8115f,    0.4121f,    0.0771f,    0.4863f,
                            0.7412f,    0.7607f,    0.1543f,    0.5479f,
                            0.8223f,    0.2246f,    0.0049f,    0.6465f}).reshape(3,4);
        INDArray scores = Nd4j.createFromArray(new float[]{0.0029f,    0.8135f,    0.4873f});
        val op = new NonMaxSuppression(boxes,scores,2,0.5,0.5);
        val res = Nd4j.exec(op);
        assertEquals(new long[]{1}, res[0].shape());
    }

    @Test
    public void testMatrixBand() {
        INDArray input = Nd4j.createFromArray(new float[]{0.7788f,0.8012f,0.7244f,0.2309f,
                                               0.7271f,0.1804f,0.5056f,0.8925f,
                                               0.5461f,0.9234f,0.0856f,0.7938f}).reshape(3,4);
        MatrixBandPart op = new MatrixBandPart(input,1,-1);
        List<LongShapeDescriptor> lsd = op.calculateOutputShape();
        assertEquals(1, lsd.size());
    }

    @Ignore("Failed AS 11.26.2019 - https://github.com/eclipse/deeplearning4j/issues/8450")
    @Test
    public void testBetaInc1() {
        INDArray a = Nd4j.createFromArray(new float[]{0.7788f,    0.8012f,    0.7244f,    0.2309f});
        INDArray b = Nd4j.createFromArray(new float[]{0.7717f,    0.9281f,    0.9846f,    0.4838f});
        INDArray c = Nd4j.createFromArray(new float[]{0.9441f,    0.5957f,    0.8669f,    0.3502f});
        BetaInc op = new BetaInc(a,b,c);
        INDArray[] ret = Nd4j.exec(op);
        INDArray expected = Nd4j.createFromArray(new float[]{0.9122f,    0.6344f,    0.8983f,    0.6245f});
        assertEquals(expected, ret[0]);
    }

    @Ignore("Failure AS 11.28.2019 - https://github.com/eclipse/deeplearning4j/issues/8452")
    @Test
    public void testPolygamma1() {
        INDArray a = Nd4j.createFromArray(new float[]{0.7788f,    0.8012f,    0.7244f,    0.2309f,
                                        0.7271f,    0.1804f,    0.5056f,    0.8925f,
                                        0.5461f,    0.9234f,    0.0856f,    0.7938f}).reshape(3,4);
        INDArray b = Nd4j.createFromArray(new float[]{0.7717f,    0.9281f,    0.9846f,    0.4838f,
                                        0.6433f,    0.6041f,    0.6501f,    0.7612f,
                                        0.7605f,    0.3948f,    0.9493f,    0.8600f}).reshape(3,4);
        INDArray expected = Nd4j.createFromArray(new float[]{NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN, }).reshape(3,4);
        Polygamma op = new Polygamma(a,b);
        INDArray[] ret = Nd4j.exec(op);
        assertEquals(expected, ret[0]);
    }

    @Ignore("Failure AS 11.28.2019 - https://github.com/eclipse/deeplearning4j/issues/8453")
    @Test
    public void testRoll1() {
        INDArray a = Nd4j.createFromArray(new float[]{0.7788f,    0.8012f,    0.7244f,    0.2309f});
        Roll op = new Roll(a,Nd4j.scalar(2),Nd4j.scalar(0));
        INDArray[] ret = Nd4j.exec(op);
        INDArray expected = Nd4j.createFromArray(new float[]{0.7244f,    0.2309f,    0.7788f,    0.8012f});
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testAdjustHueShape(){
        INDArray image = Nd4j.createFromArray(new float[]{0.7788f,    0.8012f,    0.7244f,
            0.2309f,    0.7271f,    0.1804f, 0.5056f,    0.8925f,    0.5461f,
            0.9234f,    0.0856f,    0.7938f, 0.6591f,    0.5555f,    0.1596f,
            0.3087f,    0.1548f,    0.4695f, 0.9939f,    0.6113f,    0.6765f,
            0.1800f,    0.6750f,    0.2246f, 0.0509f,    0.4601f,    0.8284f,
            0.2354f,    0.9752f,    0.8361f, 0.2585f,    0.4189f,    0.7028f,
            0.7679f,    0.5373f,    0.7234f,  0.2690f,    0.0062f,    0.0327f,
            0.0644f,    0.8428f,    0.7494f,  0.0755f,    0.6245f,    0.3491f,
            0.5793f,    0.5730f,    0.1822f,  0.6420f,    0.9143f,    0.3019f,
            0.3574f,    0.1704f,    0.8395f, 0.5468f,    0.0744f,    0.9011f,
            0.6574f,    0.4124f,    0.2445f, 0.4248f,    0.5219f,    0.6952f,
            0.4900f,    0.2158f,    0.9549f, 0.1386f,    0.1544f,    0.5365f,
            0.0134f,    0.4163f,    0.1456f, 0.4109f,    0.2484f,    0.3330f,
            0.2974f,    0.6636f,    0.3808f, 0.8664f,    0.1896f,    0.7530f,
            0.7215f,    0.6612f,    0.7270f, 0.5704f,    0.2666f,    0.7453f,
            0.0444f,    0.3024f,    0.4850f, 0.7982f,    0.0965f,    0.7843f,
            0.5075f,    0.0844f,    0.8370f, 0.6103f,    0.4604f,    0.6087f,
            0.8594f,    0.4599f,    0.6714f, 0.2744f,    0.1981f,    0.4143f,
            0.7821f,    0.3505f,    0.5040f, 0.1180f,    0.8307f,    0.1817f,
            0.8442f,    0.5074f,    0.4471f, 0.5105f,    0.6666f,    0.2576f,
            0.2341f,    0.6801f,    0.2652f, 0.5394f,    0.4690f,    0.6146f,
            0.1210f,    0.2576f,    0.0769f, 0.4643f,    0.1628f,    0.2026f,
            0.3774f,    0.0506f,    0.3462f, 0.5720f,    0.0838f,    0.4228f,
            0.0588f,    0.5362f,    0.4756f, 0.2530f,    0.1778f,    0.0751f,
            0.8977f,    0.3648f,    0.3065f, 0.4739f,    0.7014f,    0.4473f,
            0.5171f,    0.1744f,    0.3487f, 0.7759f,    0.9491f,    0.2072f,
            0.2182f,    0.6520f,    0.3092f, 0.9545f,    0.1881f,    0.9579f,
            0.1785f,    0.9636f,    0.4830f, 0.6569f,    0.3353f,    0.9997f,
            0.5869f,    0.5747f,    0.0238f, 0.2943f,    0.5248f,    0.5879f,
            0.7266f,    0.1965f,    0.9167f, 0.9726f,    0.9206f,    0.0519f,
            0.2997f,    0.0039f,    0.7652f, 0.5498f,    0.3794f,    0.3791f,
            0.3528f,    0.2873f,    0.8082f,  0.4732f,    0.4399f,    0.6606f,
            0.5991f,    0.0034f,    0.4874f}).reshape(8,8,3);

        AdjustHue op = new AdjustHue(image, 0.2f);
        INDArray[] res = Nd4j.exec(op);
//        System.out.println(res[0]);
        List<LongShapeDescriptor> lsd = op.calculateOutputShape();
        assertEquals(1, lsd.size());
        assertArrayEquals(new long[]{8, 8, 3}, lsd.get(0).getShape());
    }

    @Test
    public void testBitCastShape_3(){
        val x = Nd4j.createFromArray(new int[]{1, 2, 3, 4, 5, 6, 7, 8}).reshape(1, 4, 2);
        val e = Nd4j.createFromArray(new long[]{8589934593L, 17179869187L, 25769803781L, 34359738375L}).reshape(1, 4);
        val z = Nd4j.exec(new BitCast(x, DataType.LONG.toInt()))[0];

        assertEquals(e, z);
    }


    @Test
    public void testMatch_1() {
        INDArray x = Nd4j.ones(DataType.FLOAT, 3,3);
        INDArray y = Nd4j.linspace(DataType.FLOAT, -5, 9, 1).reshape(3, 3);
        val c =  Conditions.equals(0.0);

        INDArray z = x.match(y, c);
        INDArray exp = Nd4j.createFromArray(new boolean[][]{
                {false, false, false},
                {false, false, false},
                {true,  false, false}
        });

        assertEquals(exp, z);
    }

    @Test
    public void testCreateOp_1() {
        val shape = Nd4j.createFromArray(new int[] {3, 4, 5});
        val exp = Nd4j.create(DataType.INT, 3, 4, 5);

        val result = Nd4j.exec(new Create(shape, 'c', true, DataType.INT))[0];

        assertEquals(exp, result);
    }

    // Exact copy of libnd4j test
    @Test
    @Ignore
    public void testRgbToHsv() {
        INDArray expected = Nd4j.createFromArray(new float[]{
                0.545678377f, 0.644941628f, 0.461456001f, 0.588904262f, 0.725874603f,
                0.517642438f, 0.0869259685f, 0.54742825f, 0.413571358f, 0.890151322f,
                0.928968489f, 0.684074104f, 0.52110225f, 0.753103435f, 0.913557053f,
                0.46850124f, 0.761800349f, 0.237176552f, 0.90049392f, 0.965541422f,
                0.486593395f, 0.263826847f, 0.290193319f, 0.148351923f, 0.674094439f,
                0.0361763388f, 0.3721793f, 0.823592246f, 0.524110138f, 0.2204483f,
                0.632020354f, 0.637001634f, 0.216262609f, 0.279114306f, 0.25007084f,
                0.30433768f, 0.0448598303f, 0.586083114f, 0.978048146f, 0.91390729f,
                0.385092884f, 0.218390301f, 0.762684941f, 0.505838513f, 0.366362303f,
                0.931746006f, 0.00208298792f, 0.875348926f, 0.428009957f, 0.270003974f,
                0.313204288f, 0.775881767f, 0.367065936f, 0.164243385f, 0.644775152f,
                0.575452209f, 0.911922634f, 0.0581932105f, 0.437950462f, 0.946475744f
        }).reshape(5,4,3);
        INDArray input = Nd4j.createFromArray(new float[]{
                0.262831867f, 0.723622441f, 0.740797927f, 0.717254877f, 0.430244058f,
                0.418478161f, 0.906427443f, 0.199753001f, 0.725874603f, 0.890151322f,
                0.928968489f, 0.684074104f, 0.312434604f, 0.991390795f, 0.163174023f,
                0.268038541f, 0.361258626f, 0.685067773f, 0.682347894f, 0.84635365f,
                0.761800349f, 0.753103435f, 0.913557053f, 0.965541422f, 0.112067183f,
                0.540247589f, 0.280050347f, 0.106776128f, 0.679180562f, 0.870388806f,
                0.604331017f, 0.630475283f, 0.674094439f, 0.279114306f, 0.632020354f,
                0.823592246f, 0.490824632f, 0.75257351f, 0.129888852f, 0.849081645f,
                0.883509099f, 0.765611768f, 0.997870266f, 0.446510047f, 0.385092884f,
                0.931746006f, 0.978048146f, 0.91390729f, 0.685308874f, 0.0834472676f,
                0.396037966f, 0.756701186f, 0.597481251f, 0.784472764f, 0.514242649f,
                0.392005324f, 0.911922634f, 0.270003974f, 0.644775152f, 0.946475744f
        }).reshape(5,4,3);
        RgbToHsv op = new RgbToHsv(input);
        INDArray[] ret = Nd4j.exec(op);
        assertEquals(ret[0], expected);
    }

    // Exact copy of libnd4j test
    @Test
    public void testHsvToRgb() {
        INDArray input = Nd4j.createFromArray(new float[]{0.705504596f, 0.793608069f, 0.65870738f, 0.848827183f, 0.920532584f,
                0.887555957f, 0.72317636f, 0.563831031f, 0.773604929f, 0.269532293f,
                0.332347751f, 0.111181192f}).reshape(4,3);

        INDArray expected = Nd4j.createFromArray(new float[]{0.257768334f, 0.135951888f, 0.65870738f, 0.887555957f, 0.0705317783f,
                0.811602857f, 0.485313689f, 0.337422464f, 0.773604929f, 0.0883753772f,
                0.111181192f, 0.074230373f}).reshape(4,3);

        HsvToRgb op = new HsvToRgb(input);
        INDArray[] ret = Nd4j.exec(op);
        assertEquals(ret[0], expected);
    }

    @Test
    public void testHsvToRgb_1() {
        /* Emulation of simple TF test:
           image = tf.random_uniform(shape = [1,1,3])
           tf.image.hsv_to_rgb(image)*/
        INDArray image = Nd4j.createFromArray(new float[]{0.778785586f,0.801197767f,0.724374652f}).
                reshape(1,1,3);
        HsvToRgb op = new HsvToRgb(image);
        INDArray[] ret = Nd4j.exec(op);
        System.out.println(ret[0].toStringFull());
        INDArray expected = Nd4j.createFromArray(new float[]{ 0.53442812f,    0.144007325f,    0.724374652f}).reshape(1,1,3);
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testRgbToHsv_1() {
        /* Emulation of simple TF test:
           image = tf.random_uniform(shape = [1,2,3])
           tf.image.rgb_to_hsv(image)*/
        INDArray image = Nd4j.createFromArray(new float[]{0.778785586f,0.801197767f,0.724374652f,
                0.230894327f, 0.727141261f, 0.180390716f }).reshape(2,3);
        RgbToHsv op = new RgbToHsv(image);
        INDArray[] ret = Nd4j.exec(op);
        INDArray expected = Nd4j.createFromArray(new float[]{0.215289578f,0.095885336f,0.801197767f,
                0.317938268f,0.751917899f,0.727141261f}).reshape(2,3);
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testLu() {
        INDArray input = Nd4j.createFromArray(new float[]{1.f, 2.f, 3.f, 0.f, 2.f, 3.f, 0.f, 0.f, 7.f})
                .reshape(3,3);
        Lu op = new Lu(input);
        INDArray[] ret = Nd4j.exec(op);

        INDArray expected = Nd4j.createFromArray(new float[]{1.f, 2.f, 3.f, 0.f, 2.f, 3.f, 0.f, 0.f, 7f}).reshape(3,3);
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testRgbToYiq() {
        INDArray image = Nd4j.createFromArray(new float[]{
                0.48055f   , 0.80757356f, 0.2564435f , 0.94277316f, 0.17006584f,
                0.33366168f, 0.41727918f, 0.54528666f, 0.48942474f, 0.3305715f ,
                0.98633456f, 0.00158441f, 0.97605824f, 0.02462568f, 0.14837205f,
                0.00112842f, 0.99260217f, 0.9585542f , 0.41196227f, 0.3095014f ,
                0.6620493f , 0.30888894f, 0.3122602f , 0.7993488f , 0.86656475f,
                0.5997049f , 0.9776477f , 0.72481847f, 0.7835693f , 0.14649455f,
                0.3573504f , 0.33301765f, 0.7853056f , 0.25830218f, 0.59289205f,
                0.41357264f, 0.5934154f , 0.72647524f, 0.6623308f , 0.96197623f,
                0.0720306f , 0.23853847f, 0.1427159f , 0.19581454f, 0.06766324f,
                0.10614152f, 0.26093867f, 0.9584985f , 0.01258832f, 0.8160156f ,
                0.56506383f, 0.08418505f, 0.86440504f, 0.6807802f , 0.20662387f,
                0.4153733f , 0.76146203f, 0.50057423f, 0.08274968f, 0.9521758f
        }).reshape(5,4,3);

        INDArray expected = Nd4j.createFromArray(new float[]{
                0.64696468f, -0.01777124f, -0.24070648f,  0.41975525f,  0.40788622f,
                0.21433232f,  0.50064416f, -0.05832884f, -0.04447775f,  0.67799989f,
                -0.07432612f, -0.44518381f,  0.32321111f,  0.52719408f,  0.2397369f ,
                0.69227005f, -0.57987869f, -0.22032876f,  0.38032767f, -0.05223263f,
                0.13137188f,  0.3667803f , -0.15853189f,  0.15085728f,  0.72258149f,
                0.03757231f,  0.17403452f,  0.69337627f,  0.16971045f, -0.21071186f,
                0.39185397f, -0.13084008f,  0.145886f  ,  0.47240727f, -0.1417591f ,
                -0.12659159f,  0.67937788f, -0.05867803f, -0.04813048f,  0.35710624f,
                0.47681283f,  0.24003804f,  0.1653288f ,  0.00953913f, -0.05111816f,
                0.29417614f, -0.31640032f,  0.18433114f,  0.54718234f, -0.39812097f,
                -0.24805083f,  0.61018603f, -0.40592682f, -0.22219216f,  0.39241133f,
                -0.23560742f,  0.06353694f,  0.3067938f , -0.0304029f ,  0.35893188f
        }).reshape(5,4,3);

        RgbToYiq op = new RgbToYiq(image);
        INDArray[] ret = Nd4j.exec(op);
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testYiqToRgb() {
        INDArray image = Nd4j.createFromArray(new float[]{
                0.775258899f, -0.288912386f, -0.132725924f, 0.0664454922f, -0.212469354f,
                0.455438733f, 0.418221354f, 0.349350512f, 0.145902053f, 0.947576523f,
                -0.471601307f, 0.263960421f, 0.700227439f, 0.32434237f, -0.278446227f,
                0.130805135f, -0.438441873f, 0.187127829f, 0.0276055578f, -0.179727226f,
                0.305075705f, 0.716282248f, 0.278215706f, -0.44586885f, 0.76971364f,
                0.131288841f, -0.141177326f, 0.900081575f, -0.0788725987f, 0.14756602f,
                0.387832165f, 0.229834676f, 0.47921446f, 0.632930398f, 0.0443540029f,
                -0.268817365f, 0.0977194682f, -0.141669706f, -0.140715122f, 0.946808815f,
                -0.52525419f, -0.106209636f, 0.659476519f, 0.391066104f, 0.426448852f,
                0.496989518f, -0.283434421f, -0.177366048f, 0.715208411f, -0.496444523f,
                0.189553142f, 0.616444945f, 0.345852494f, 0.447739422f, 0.224696323f,
                0.451372236f, 0.298027098f, 0.446561724f, -0.187599331f, -0.448159873f
        }).reshape(5,4,3);

        INDArray expected = Nd4j.createFromArray(new float[]{
                0.416663059f, 0.939747555f, 0.868814286f, 0.146075352f, -0.170521997f,
                1.07776645f, 0.842775284f, 0.228765106f, 0.280231822f, 0.660605291f,
                0.905021825f, 1.91936605f, 0.837427991f, 0.792213732f, -0.133271854f,
                -0.17216571f, 0.128957025f, 0.934955336f, 0.0451873479f, -0.120952621f,
                0.746436225f, 0.705446224f, 0.929172217f, -0.351493549f, 0.807577594f,
                0.825371955f, 0.383812296f, 0.916293093f, 0.82603058f, 1.23885956f,
                0.905059196f, 0.015164554f, 0.950156781f, 0.508443732f, 0.794845279f,
                0.12571529f, -0.125074273f, 0.227326869f, 0.0147000261f, 0.378735409f,
                1.15842402f, 1.34712305f, 1.2980804f, 0.277102016f, 0.953435072f,
                0.115916842f, 0.688879376f, 0.508405162f, 0.35829352f, 0.727568094f,
                1.58768577f, 1.22504294f, 0.232589777f, 0.996727258f, 0.841224629f,
                -0.0909671176f, 0.233051388f, -0.0110094378f, 0.787642119f, -0.109582274f
        }).reshape(5,4,3);

        YiqToRgb op = new YiqToRgb(image);
        INDArray[] ret = Nd4j.exec(op);
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testRgbToGrayscale() {
        INDArray image = Nd4j.createFromArray(new float[]{
                1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,
                9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f,
                -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,
                2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,
                1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,
                1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,
                2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,
                3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,
                5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,
                1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,
                5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,
                -2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f,
                -4.8125e+01f
        }).reshape(5,4,3);

        INDArray expected = Nd4j.createFromArray(new float[]{
                -47.82958221f,  34.46305847f,  21.36137581f, -21.91625023f,2.49686432f,
                -43.59792709f,   9.64180183f,  23.04854202f,40.7946167f,  44.98754883f,
                -25.19047546f,  20.64586449f,-4.97033119f,   30.0226841f,  30.30688286f,
                15.61459541f,43.36166f,  18.22480774f,  13.74833488f,  21.59387016f
        }).reshape(5,4,1);

        RgbToGrayscale op = new RgbToGrayscale(image);
        INDArray[] ret = Nd4j.exec(op);
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testRgbToYuv() {
        INDArray image = Nd4j.createFromArray(new float[]{
                10f,50f,200f
        });

        INDArray expected = Nd4j.createFromArray(new float[]{
                55.14f , 71.2872001f, -39.6005542f
        });

        RgbToYuv op = new RgbToYuv(image);
        INDArray[] ret = Nd4j.exec(op);
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testYuvToRgb() {
        INDArray image = Nd4j.createFromArray(new float[]{
                55.14f , 71.2872001f, -39.6005542f
        });

        INDArray expected = Nd4j.createFromArray(new float[]{
                10f, 50f, 200f
        });
        YuvToRgb op = new YuvToRgb(image);
        INDArray[] ret = Nd4j.exec(op);
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testRgbToYiqEmpty() {
        INDArray image = Nd4j.create(0,4,3);
        RgbToYiq op = new RgbToYiq(image);
        INDArray[] ret = Nd4j.exec(op);
        assertArrayEquals(image.shape(), ret[0].shape());
    }

    @Test
    public void testTriangularSolve() {
        INDArray a = Nd4j.createFromArray(new float[]{
                3.f, 0.f, 0.f, 0.f,
                2.f, 1.f, 0.f, 0.f,
                1.f, 0.f, 1.f, 0.f,
                1.f, 1.f, 1.f, 1.f
        }).reshape(4, 4);

        INDArray b = Nd4j.createFromArray(new float[]{
                4.f, 2.f, 4.f, 2.f
        }).reshape(4, 1);

        INDArray expected = Nd4j.createFromArray(new float[]{
                1.333333f, -0.6666667f, 2.6666667f, -1.3333333f
        }).reshape(4, 1);

        val op = new TriangularSolve(a, b, true, false);
        INDArray[] ret = Nd4j.exec(op);

        assertEquals(expected, ret[0]);
    }

    @Test
    public void testOnesLike_1() {
        val x = Nd4j.create(DataType.FLOAT, 3, 4, 5);
        val e = Nd4j.ones(DataType.INT32, 3, 4, 5);

        val z = Nd4j.exec(new OnesLike(x, DataType.INT32))[0];
        assertEquals(e, z);
    }

    @Test
    public void testLinSpaceEdge_1() {
        val x = Nd4j.linspace(1,10,1, DataType.FLOAT);
        val e = Nd4j.scalar(1.0f);

        assertEquals(e, x);
    }

    @Test
    public void testLinearSolve() {
        INDArray a = Nd4j.createFromArray(new float[]{
                2.f, -1.f, -2.f, -4.f, 6.f, 3.f, -4.f, -2.f, 8.f
        }).reshape(3, 3);

        INDArray b = Nd4j.createFromArray(new float[]{
                2.f, 4.f, 3.f
        }).reshape(3, 1);

        INDArray expected = Nd4j.createFromArray(new float[]{
                7.625f, 3.25f, 5.f
        }).reshape(3, 1);

        val op = new LinearSolve(a, b);
        INDArray[] ret = Nd4j.exec(op);

        assertEquals(expected, ret[0]);
    }

    @Test
    public void testLinearSolveAdjust() {
        INDArray a = Nd4j.createFromArray(new float[]{
                0.7788f,    0.8012f,    0.7244f,
                0.2309f,    0.7271f,    0.1804f,
                0.5056f,    0.8925f,    0.5461f
        }).reshape(3, 3);

        INDArray b = Nd4j.createFromArray(new float[]{
                0.7717f,    0.9281f,    0.9846f,
                0.4838f,    0.6433f,    0.6041f,
                0.6501f,    0.7612f,    0.7605f
        }).reshape(3, 3);

        INDArray expected = Nd4j.createFromArray(new float[]{
                1.5504692f,  1.8953944f,  2.2765768f,
                0.03399149f,  0.2883001f ,  0.5377323f,
                -0.8774802f, -1.2155888f, -1.8049058f
        }).reshape(3, 3);

        val op = new LinearSolve(a, b, true);
        INDArray[] ret = Nd4j.exec(op);

        assertEquals(expected, ret[0]);
    }

    @Test
    public void testLstsq() {
        INDArray a = Nd4j.createFromArray(new float[]{
                1.f,  2.f,  3.f,
                4.f,  5.f,  6.f,
                11.f,  8.f, 21.f
        }).reshape(3,3);

        INDArray b = Nd4j.createFromArray(new float[]{   1.f, 2.f, 3.f   }).reshape(3,1);

        val op = new Lstsq(a,b);
        INDArray[] ret = Nd4j.exec(op);

        DynamicCustomOp matmul = DynamicCustomOp.builder("matmul")
                .addInputs(a, ret[0])
                .build();
        INDArray[] matres = Nd4j.exec(matmul);
        for (int i = 0; i < 3; ++i) {
            assertEquals(b.getFloat(i, 0), matres[0].getFloat(i, 0), 1e-4);
        }
    }

    @Test
    public void testSequenceMask() {
        INDArray arr = Nd4j.createFromArray(new int[]{1, 3, 2});
        // Test with static max len
        int maxlen = 2;
        INDArray expected = Nd4j.createFromArray(new int[]{
                1, 0, 0,
                1, 1, 1,
                1, 1, 0
        }).reshape(3, 3);

        INDArray[] ret = Nd4j.exec(new SequenceMask(arr, maxlen, DataType.INT32));
        assertEquals(expected, ret[0]);
    }

    @Test
    public void testCholesky() {
        INDArray x = Nd4j.createFromArray(new double[] {4,12,-16, 12 ,37,-43, -16, -43, 98}).reshape(3,3);
        INDArray exp = Nd4j.createFromArray(new double[] {2.,  0.,  0., 6., 1.,  0., -8.,  5.,  3.}).reshape(3,3);

        INDArray[] res = Nd4j.exec(new Cholesky(x));
        assertEquals(res[0], exp);
    }

    @Test
    public void testQr() {
        INDArray in = Nd4j.createFromArray(new double[]{
                12.,  -51.,    4.,        6.,   167.,  -68.,       -4.,    24.,  -41.,       -1.,     1.,    0.,        2.,     0.,    3.
        }).reshape(5,3);
        Qr op = new Qr(in);
        INDArray[] ret = Nd4j.exec(op);
        INDArray res = Nd4j.createUninitialized(in.shape());
        DynamicCustomOp matmul = DynamicCustomOp.builder("matmul")
                .addInputs(ret[0], ret[1])
                .build();
        ret = Nd4j.exec(matmul);
        assertEquals(ret[0], in);
    }


    @Test
    public void testLinspaceSignature_1() throws Exception {
        val array1 = Nd4j.exec(new Linspace(DataType.FLOAT, Nd4j.scalar(1.0f), Nd4j.scalar(10.f), Nd4j.scalar(10L)))[0];
        val array2 = Nd4j.exec(new Linspace(DataType.FLOAT, 1.0f, 10.f, 10L))[0];

        assertEquals(array1.dataType(), array2.dataType());
        assertEquals(array1, array2);
    }

    @Test
    public void testLogdet() {
        INDArray x = Nd4j.createFromArray(new double[]{
                4,12,-16,12,37,-43,-16,-43,98, 4,1.2,-1.6,1.2,3.7,-4.3,-1.6,-4.3,9.8
        }).reshape(2,3,3);

        INDArray expected = Nd4j.createFromArray(new double[]{3.5835189, 4.159008});
        INDArray[] ret = Nd4j.exec(new Logdet(x));
        assertEquals(ret[0], expected);

    }

    @Test
    public void testBatchNormBpNHWC(){
        //Nd4j.getEnvironment().allowHelpers(false);        //Passes if helpers/MKLDNN is disabled

        INDArray in = Nd4j.rand(DataType.FLOAT, 2, 4, 4, 3);
        INDArray eps = Nd4j.rand(DataType.FLOAT, in.shape());
        INDArray epsStrided = eps.permute(1,0,2,3).dup().permute(1,0,2,3);
        INDArray mean = Nd4j.rand(DataType.FLOAT, 3);
        INDArray var = Nd4j.rand(DataType.FLOAT, 3);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, 3);
        INDArray beta = Nd4j.rand(DataType.FLOAT, 3);

        assertEquals(eps, epsStrided);

        INDArray out1eps = in.like();
        INDArray out1m = mean.like();
        INDArray out1v = var.like();

        INDArray out2eps = in.like();
        INDArray out2m = mean.like();
        INDArray out2v = var.like();

        DynamicCustomOp op1 = DynamicCustomOp.builder("batchnorm_bp")
                .addInputs(in, mean, var, gamma, beta, eps)
                .addOutputs(out1eps, out1m, out1v)
                .addIntegerArguments(1, 1, 3)
                .addFloatingPointArguments(1e-5)
                .build();

        DynamicCustomOp op2 = DynamicCustomOp.builder("batchnorm_bp")
                .addInputs(in, mean, var, gamma, beta, epsStrided)
                .addOutputs(out2eps, out2m, out2v)
                .addIntegerArguments(1, 1, 3)
                .addFloatingPointArguments(1e-5)
                .build();

        Nd4j.exec(op1);
        Nd4j.exec(op2);

        assertEquals(out1eps, out2eps);        //Fails here
        assertEquals(out1m, out2m);
        assertEquals(out1v, out2v);
    }

    @Test
    public void testSpaceToDepthBadStrides(){
        INDArray in = Nd4j.rand(DataType.FLOAT, 2, 3, 6, 6);
        INDArray inBadStrides = in.permute(1,0,2,3).dup().permute(1,0,2,3);
        assertEquals(in, inBadStrides);

        System.out.println("in: " + in.shapeInfoToString());
        System.out.println("inBadStrides: " + inBadStrides.shapeInfoToString());

        INDArray out = Nd4j.create(DataType.FLOAT, 2, 12, 3, 3);
        INDArray out2 = out.like();

        CustomOp op1 = DynamicCustomOp.builder("space_to_depth")
                .addInputs(in)
                .addIntegerArguments(2, 0)       //nchw = 0, nhwc = 1
                .addOutputs(out)
                .build();
        Nd4j.exec(op1);

        CustomOp op2 = DynamicCustomOp.builder("space_to_depth")
                .addInputs(inBadStrides)
                .addIntegerArguments(2, 0)       //nchw = 0, nhwc = 1
                .addOutputs(out2)
                .build();
        Nd4j.exec(op2);

        assertEquals(out, out2);
    }
}
