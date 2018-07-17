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

package org.nd4j.linalg.jcublas.ops.executioner;

import lombok.val;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.PointerPointer;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridDescriptor;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.accum.Sum;
import org.nd4j.linalg.api.ops.impl.meta.PredicateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.ReduceMetaOp;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarSubtraction;
import org.nd4j.linalg.api.ops.impl.transforms.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.Set;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.AddressRetriever;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class MetaOpTests {
    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration()
                .enableDebug(true);
    }


    @Ignore
    @Test
    public void testLinearMetaOp1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray array = Nd4j.create(new float[]{-11f, -12f, -13f, -14f, -15f, -16f, -17f, -18f, -19f, -20f});
        INDArray exp = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f});
        INDArray exp2 = Nd4j.create(new float[]{11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f});

        ScalarAdd opA = new ScalarAdd(array, 10f);

        Abs opB = new Abs(array);

        PredicateMetaOp metaOp = new PredicateMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        GridDescriptor descriptor = metaOp.getGridDescriptor();

        assertEquals(2, descriptor.getGridDepth());
        assertEquals(2, descriptor.getGridPointers().size());

        assertEquals(Op.Type.SCALAR, descriptor.getGridPointers().get(0).getType());
        assertEquals(Op.Type.TRANSFORM, descriptor.getGridPointers().get(1).getType());

        long time1 = System.nanoTime();
        executioner.exec(metaOp);
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 1));
        assertEquals(exp, array);

        time1 = System.nanoTime();
        Nd4j.getExecutioner().exec(opA);
        Nd4j.getExecutioner().exec(opB);
        time2 = System.nanoTime();

        System.out.println("Execution time Linear: " + ((time2 - time1) / 1));

        assertEquals(exp2, array);
    }

    @Ignore
    @Test
    public void testLinearMetaOp2() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray array = Nd4j.create(new float[]{-11f, -12f, -13f, -14f, -15f, -16f, -17f, -18f, -19f, -20f});
        INDArray exp = Nd4j.create(new float[]{21f, 22f, 23f, 24f, 25f, 26f, 27f, 28f, 29f, 30f});
        INDArray exp2 = Nd4j.create(new float[]{31f, 32f, 33f, 34f, 35f, 36f, 37f, 38f, 39f, 40f});

        Abs opA = new Abs(array);

        ScalarAdd opB = new ScalarAdd(array, 10f);

        PredicateMetaOp metaOp = new PredicateMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        GridDescriptor descriptor = metaOp.getGridDescriptor();

        assertEquals(2, descriptor.getGridDepth());
        assertEquals(2, descriptor.getGridPointers().size());

        assertEquals(Op.Type.TRANSFORM, descriptor.getGridPointers().get(0).getType());
        assertEquals(Op.Type.SCALAR, descriptor.getGridPointers().get(1).getType());

        long time1 = System.nanoTime();
        executioner.exec(metaOp);
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 1));
        assertEquals(exp, array);

        time1 = System.nanoTime();
        Nd4j.getExecutioner().exec(opA);
        Nd4j.getExecutioner().exec(opB);
        time2 = System.nanoTime();

        System.out.println("Execution time Linear: " + ((time2 - time1) / 1));

        assertEquals(exp2, array);
    }

    @Test
    public void testPooling2D() {
        Nd4j.create(1);

        val input = Nd4j.linspace(1, 600, 600).reshape(2, 10, 10, 3);
        val permuted = input.permute(0, 3, 1, 2);

        val nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        val output = Nd4j.create(2, 3, 4, 4);

        val context = AtomicAllocator.getInstance().getFlowController().prepareAction(output, permuted);

        val ptrBIn = (FloatPointer) AtomicAllocator.getInstance().getPointer(permuted, context);
        val ptrBOut = (FloatPointer) AtomicAllocator.getInstance().getPointer(output, context);

        val ptrSIn = (IntPointer) AtomicAllocator.getInstance().getPointer(permuted.shapeInfoDataBuffer());
        val ptrSOut = (IntPointer) AtomicAllocator.getInstance().getPointer(output.shapeInfoDataBuffer());
        //                                                                       kY  kX  sY  sX  pY  pX  dY  dX  N   M   P
        val bufParams = Nd4j.getConstantHandler().getConstantBuffer(new float[] {3,  3,  3,  3,  1,  1,  1,  1,  1,  2,  2});
        val ptrBParams = (FloatPointer) AtomicAllocator.getInstance().getPointer(bufParams, context);

        PointerPointer xShapeInfoHostPointer = new PointerPointer(32).put(AddressRetriever.retrieveHostPointer(permuted.shapeInfoDataBuffer()), // 0
                        context.getOldStream(), // 1
                        AtomicAllocator.getInstance().getDeviceIdPointer(), // 2
                        context.getBufferAllocation(), // 3
                        context.getBufferReduction(), // 4
                        context.getBufferScalar(), // 5
                        context.getBufferSpecial(), // 6
                        null, // 7
                        AddressRetriever.retrieveHostPointer(output.shapeInfoDataBuffer()) // 8
/*                        hostTadShapeInfo, // 9
                        devTadShapeInfo, // 10
                        devTadOffsets, // 11
                        hostMaxTadShapeInfo, // 12
                        devMaxTadShapeInfo, // 13
                        devMaxTadOffsets, // 14
                        dimensionDevPointer, // special pointer for IsMax  // 15
                        dimensionHostPointer, // special pointer for IsMax  // 16
                        retPointer, // special pointer for IsMax // 17
                        new CudaPointer(dimension == null ? 0 : dimension.length));
*/
                        );
        nativeOps.execTransformFloat(xShapeInfoHostPointer, 71, ptrBIn, ptrSIn, ptrBOut, ptrSOut, ptrBParams);


        AtomicAllocator.getInstance().getFlowController().registerAction(context, output, permuted);
        nativeOps.streamSynchronize(context.getOldStream());
        nativeOps.streamSynchronize(context.getOldStream());

        val reverted = output.permute(0, 2, 3, 1);

        System.out.println("Result: " + reverted.toString());

    }
/*
    @Ignore
    @Test
    public void testPredicateScalarPairwise1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(new float[]{0f, 0f, 0f, 0f, 0f, 0f});
        INDArray arrayY = Nd4j.create(new float[]{2f, 2f, 2f, 2f, 2f, 2f});
        INDArray exp = Nd4j.create(new float[]{3f, 3f, 3f, 3f, 3f, 3f});

        ScalarAdd opA = new ScalarAdd(arrayX, 1.0f);

        AddOp opB = new AddOp(arrayX, arrayY, arrayX);

        PredicateMetaOp metaOp = new PredicateMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        long time1 = System.nanoTime();
        executioner.exec(metaOp);
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 1));

        assertEquals(exp, arrayX);
    }

    @Ignore
    @Test
    public void testPredicateScalarPairwise2() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(new float[]{0f, 0f, 0f, 0f, 0f, 0f});
        INDArray arrayY = Nd4j.create(new float[]{2f, 2f, 2f, 2f, 2f, 2f});
        INDArray exp = Nd4j.create(new float[]{1f, 1f, 1f, 1f, 1f, 1f});

        ScalarSubtraction opA = new ScalarSubtraction(arrayX, 1.0f);

        AddOp opB = new AddOp(arrayX, arrayY, arrayX);

        PredicateMetaOp metaOp = new PredicateMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        long time1 = System.nanoTime();
        executioner.exec(metaOp);
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 1));

        assertEquals(exp, arrayX);
    }
*/
    /**
     * This is the MOST crucial test, basically it's test for dup() + following linear op
     *
     * @throws Exception
     */
    @Test
    public void testPredicateScalarPairwise3() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(new float[]{0f, 0f, 0f, 0f, 0f, 0f});
        INDArray arrayY = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        INDArray exp = Nd4j.create(new float[]{2f, 4f, 6f, 8f, 10f, 12f});

        Set opA = new Set(arrayX, arrayY, arrayX, arrayX.length());

        ScalarMultiplication opB = new ScalarMultiplication(arrayX, 2.0f);
        //ScalarAdd opB = new ScalarAdd(arrayX, 3.0f);

        PredicateMetaOp metaOp = new PredicateMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        long time1 = System.nanoTime();
        executioner.exec(metaOp);
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 1));

        assertEquals(exp, arrayX);
    }

    /**
     * Scalar + reduce along dimension
     *
     * @throws Exception
     */
    @Test
    public void testPredicateReduce1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(5, 5);
        INDArray exp = Nd4j.create(new float[]{2f, 2f, 2f, 2f, 2f});

        ScalarAdd opA = new ScalarAdd(arrayX, 2.0f);

        Max opB = new Max(arrayX);

        OpDescriptor a = new OpDescriptor(opA);
        OpDescriptor b = new OpDescriptor(opB, new int[]{1});

        executioner.buildZ(opB, b.getDimensions());

        ReduceMetaOp metaOp = new ReduceMetaOp(a, b);

        executioner.prepareGrid(metaOp);

        executioner.exec(metaOp);

        INDArray result = opB.z();
        assertNotEquals(null, result);
        assertEquals(exp, result);
    }

    /**
     * Predicate test for scalar + reduceScalar
     *
     * @throws Exception
     */
    @Ignore
    @Test
    public void testPredicateReduce2() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(5, 5);

        ScalarAdd opA = new ScalarAdd(arrayX, 1.0f);

        Sum opB = new Sum(arrayX);

        PredicateMetaOp metaOp = new PredicateMetaOp(opA, opB);

        executioner.exec(metaOp);

        INDArray result = opB.z();

        assertNotEquals(null, result);

        assertTrue(result.isScalar());
        assertEquals(25f, result.getFloat(0), 0.1f);
    }


    @Test
    public void testPerformance1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

/*        INDArray array = Nd4j.create(new float[]{-11f, -12f, -13f, -14f, -15f, -16f, -17f, -18f, -19f, -20f});
        INDArray exp = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f});
        INDArray exp2 = Nd4j.create(new float[]{11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f});
        */
        INDArray arrayX = Nd4j.create(65536);
        INDArray arrayY = Nd4j.create(65536);
        INDArray exp2 = Nd4j.create(65536);

        Set opA = new Set(arrayX, arrayY, arrayX, arrayX.length());

        ScalarMultiplication opB = new ScalarMultiplication(arrayX, 2.0f);
        //ScalarAdd opB = new ScalarAdd(arrayX, 3.0f);

        PredicateMetaOp metaOp = new PredicateMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        GridDescriptor descriptor = metaOp.getGridDescriptor();

        assertEquals(2, descriptor.getGridDepth());
        assertEquals(2, descriptor.getGridPointers().size());

        assertEquals(Op.Type.PAIRWISE, descriptor.getGridPointers().get(0).getType());
        assertEquals(Op.Type.SCALAR, descriptor.getGridPointers().get(1).getType());

        long time1 = System.nanoTime();
        for (int x = 0; x < 1000000; x++) {
            executioner.exec(metaOp);
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 1000000));
      //  assertEquals(exp, array);

        time1 = System.nanoTime();
        for (int x = 0; x < 1000000; x++) {
            Nd4j.getExecutioner().exec(opA);
            Nd4j.getExecutioner().exec(opB);
        }
        time2 = System.nanoTime();

        System.out.println("Execution time Linear: " + ((time2 - time1) / 1000000));


      //  assertEquals(exp2, array);

    }

    @Test
    public void testEnqueuePerformance1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(65536);
        INDArray arrayY = Nd4j.create(65536);

        Set opA = new Set(arrayX, arrayY, arrayX, arrayX.length());

        ScalarMultiplication opB = new ScalarMultiplication(arrayX, 2.0f);
        //ScalarAdd opB = new ScalarAdd(arrayX, 3.0f);

        PredicateMetaOp metaOp = new PredicateMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        GridDescriptor descriptor = metaOp.getGridDescriptor();

        assertEquals(2, descriptor.getGridDepth());
        assertEquals(2, descriptor.getGridPointers().size());

        assertEquals(Op.Type.PAIRWISE, descriptor.getGridPointers().get(0).getType());
        assertEquals(Op.Type.SCALAR, descriptor.getGridPointers().get(1).getType());

        long time1 = System.nanoTime();
        for (int x = 0; x < 10000000; x++) {
            executioner.exec(opA);
            executioner.purgeQueue();
        }
        long time2 = System.nanoTime();

        System.out.println("Enqueue time: " + ((time2 - time1) / 10000000));
    }
/*

    @Ignore
    @Test
    public void testPerformance2() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(1024);
        INDArray arrayY = Nd4j.create(1024);
        INDArray exp = Nd4j.create(1024);

        ScalarAdd opA = new ScalarAdd(arrayX, 1.0f);

        AddOp opB = new AddOp(arrayX, arrayY, arrayX);

        PredicateMetaOp metaOp = new PredicateMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        long time1 = System.nanoTime();
        for (int x = 0; x < 100000; x++) {
            executioner.exec(metaOp);
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 100000));

        time1 = System.nanoTime();
        for (int x = 0; x < 100000; x++) {
            Nd4j.getExecutioner().exec(opA);
            Nd4j.getExecutioner().exec(opB);
        }
        time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 100000));
    }
    */
}
