package org.nd4j.linalg.jcublas.buffer;

import org.apache.commons.io.FilenameUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.util.ArrayUtil;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaFloatDataBufferTest {

    @Test
    public void testDoubleDimJava1() throws Exception {
        INDArray sliceZero = Nd4j.create(new double[][]{{1, 7}, {4, 10}});

        System.out.println("Slice: " + sliceZero);
        assertEquals(1f, sliceZero.getFloat(0), 0.01f);
        assertEquals(7f, sliceZero.getFloat(1), 0.01f);
    }

    @Test
    public void getDouble() throws Exception {
        DataBuffer buffer = Nd4j.createBuffer(new float[]{1f,2f,3f,4f});

        assertEquals("CudaFloatDataBuffer", buffer.getClass().getSimpleName());


        System.out.println("Starting check...");

        assertEquals(1.0f, buffer.getFloat(0), 0.001f);
        assertEquals(2.0f, buffer.getFloat(1), 0.001f);
        assertEquals(3.0f, buffer.getFloat(2), 0.001f);
        assertEquals(4.0f, buffer.getFloat(3), 0.001f);

        System.out.println("Data: " + buffer);
    }

    @Test
    public void testPut() throws Exception {
        DataBuffer buffer = Nd4j.createBuffer(new float[]{1f,2f,3f,4f});
        buffer.put(2, 16f);

        assertEquals(16.0f, buffer.getFloat(2), 0.001f);

        System.out.println("Data: " + buffer);
    }

    @Test
    public void testNdArrayView1() throws Exception {
        INDArray array = Nd4j.create(new float[] {1f,2f,3f,4f});

        assertEquals(1.0f, array.getFloat(0), 0.001f);
        assertEquals(2.0f, array.getFloat(1), 0.001f);
        assertEquals(3.0f, array.getFloat(2), 0.001f);
        assertEquals(4.0f, array.getFloat(3), 0.001f);
    }

    @Test
    public void testNdArrayView2() throws Exception {
        INDArray array = Nd4j.create(10, 10);

        System.out.println("X0 --------------------------------");
        long tp1 = array.data().getTrackingPoint();

        array.putScalar(0, 10f);
        assertEquals(10.0f, array.getFloat(0), 0.01f);

        System.out.println("X1 --------------------------------");

        INDArray array2 = array.slice(1);
        long tp2 = array2.data().getTrackingPoint();

        assertEquals(tp1, tp2);

        array2.putScalar(0, 10);
        System.out.println("X2 --------------------------------");
        assertEquals(10.0f, array2.getFloat(0), 0.01f);

        tp2 = array2.data().getTrackingPoint();
        tp1 = array.data().getTrackingPoint();

        assertEquals(tp1, tp2);
    }

    @Test
    public void testINDArrayOffsets2() throws Exception {
        INDArray array = Nd4j.linspace(0, 24, 25).reshape(5, 5);

        assertEquals(6.0f, array.getFloat(6), 0.01f);

        INDArray slice0 = array.slice(0);
        assertEquals(0f, slice0.getFloat(0), 0.01f);

        INDArray slice1 = array.slice(1);
        assertEquals(5f, slice1.getFloat(0), 0.01f);

        INDArray slice2 = array.slice(2);
        assertEquals(10f, slice2.getFloat(0), 0.01f);

        INDArray slice3 = array.slice(3);
        assertEquals(15f, slice3.getFloat(0), 0.01f);

        assertEquals(array.data().getTrackingPoint(), slice0.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice1.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice2.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice3.data().getTrackingPoint());
    }

    @Test
    public void testINDArrayOffsets3() throws Exception {
        INDArray array = Nd4j.linspace(0, 24, 25).reshape(5, 5);
        INDArray array2 = Nd4j.create(5, 5);

        assertEquals(6.0f, array.getFloat(6), 0.01f);

        INDArray slice0 = array.slice(0);
        assertEquals(0f, slice0.getFloat(0), 0.01f);

        array2.putRow(0, slice0);
        assertEquals(slice0, array2.slice(0));

        INDArray slice1 = array.slice(1);
        assertEquals(5f, slice1.getFloat(0), 0.01f);

        System.out.println("---------------------------------------------------------------------");
        array2.putRow(1, slice1);
//        assertFalse(true);
        assertEquals(slice1, array2.slice(1));
        System.out.println("---------------------------------------------------------------------");


        INDArray slice2 = array.slice(2);
        assertEquals(10f, slice2.getFloat(0), 0.01f);

        INDArray slice3 = array.slice(3);
        assertEquals(15f, slice3.getFloat(0), 0.01f);

        assertEquals(array.data().getTrackingPoint(), slice0.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice1.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice2.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice3.data().getTrackingPoint());
    }

    @Test
    public void testDup1() throws Exception {
        INDArray array0 = Nd4j.create(10);
        INDArray array7 = Nd4j.create(10);

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();
        context.syncOldStream();

        long time1 = System.nanoTime();
        INDArray array1 =  Nd4j.linspace(0, 9, 1000);
        long time2 = System.nanoTime();
//        context.syncOldStream();
        long time3 = System.nanoTime();
        INDArray array2 = array1.dup();
        long time4 = System.nanoTime();
        //context.syncOldStream();

        System.out.println("Linspace time: " + (time2 - time1));
        System.out.println("Dup time: " + (time4 - time3));
        System.out.println("Total time: " + ((time2 - time1)+ (time4 - time3)));

        assertEquals(array1, array2);
        assertNotEquals(array1.data().getTrackingPoint(), array2.data().getTrackingPoint());
    }

    @Test
    public void testDup2() throws Exception {
        INDArray array = Nd4j.linspace(0,99, 100).reshape(10, 10);

        INDArray slice1 = array.slice(2);

        assertEquals(10, slice1.length());

        INDArray duplicate = slice1.dup();
        assertEquals(10, duplicate.length());

        assertEquals(10, duplicate.data().length());
        assertEquals(10, duplicate.data().asDouble().length);

        assertEquals(20f, duplicate.getFloat(0), 0.0001f);
        assertEquals(21f, duplicate.getFloat(1), 0.0001f);
        assertEquals(22f, duplicate.getFloat(2), 0.0001f);
        assertEquals(23f, duplicate.getFloat(3), 0.0001f);
    }

    @Test
    public void testDup3() throws Exception {
        INDArray array = Nd4j.linspace(0,99, 100).reshape(10, 10);

        INDArray slice1 = array.slice(2);
        DataBuffer duplicate = slice1.data().dup();

        assertEquals(10, duplicate.length());
        assertEquals(20f, duplicate.getFloat(0), 0.0001f);
        assertEquals(21f, duplicate.getFloat(1), 0.0001f);
        assertEquals(22f, duplicate.getFloat(2), 0.0001f);
        assertEquals(23f, duplicate.getFloat(3), 0.0001f);
        assertEquals(24f, duplicate.getFloat(4), 0.0001f);
        assertEquals(25f, duplicate.getFloat(5), 0.0001f);
        assertEquals(26f, duplicate.getFloat(6), 0.0001f);
        assertEquals(27f, duplicate.getFloat(7), 0.0001f);
        assertEquals(28f, duplicate.getFloat(8), 0.0001f);
        assertEquals(29f, duplicate.getFloat(9), 0.0001f);

    }

    @Test
    public void testShapeInfo1() throws Exception {
        INDArray array1 = Nd4j.ones(1,10);
        System.out.println("X 0: -----------------------------");
        System.out.println(array1.shapeInfoDataBuffer());
        System.out.println(array1);

        System.out.println("X 1: -----------------------------");

        assertEquals(1.0, array1.getFloat(0), 0.0001);

        System.out.println("X 2: -----------------------------");

        assertEquals(1.0, array1.getFloat(1), 0.0001);
        assertEquals(1.0, array1.getFloat(2), 0.0001);

        System.out.println("X 3: -----------------------------");

        float sum = array1.sumNumber().floatValue();

        System.out.println("X 4: -----------------------------");

        System.out.println("Sum: " + sum);
    }

    @Test
    public void testIndexer1() throws Exception {
        INDArray array1 = Nd4j.zeros(15,15);

        System.out.println("-------------------------------------");
        assertEquals(0.0, array1.getFloat(0), 0.0001);
      //  System.out.println(array1);
    }

    @Test
    public void testIndexer2() throws Exception {
        INDArray array1 = Nd4j.create(15);

        System.out.println("-------------------------------------");
//        assertEquals(0.0, array1.getFloat(0), 0.0001);
        System.out.println(array1);
    }

    @Test
    public void testSum2() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        System.out.println("X 0: -------------------------------------");
//        System.out.println("N result: " + n);
        INDArray test = Nd4j.create(new float[]{3, 7, 11, 15}, new int[]{2, 2});
        System.out.println("X 1: -------------------------------------");
//        System.out.println("Test result: " + test);
        INDArray sum = n.sum(-1);

        System.out.println("X 2: -------------------------------------");

//        System.out.println("Sum result: " + sum);
        assertEquals(test, sum);
    }

    @Test
    public void testOffsets() throws Exception {
        DataBuffer create = Nd4j.createBuffer(new double[]{1,2,3,4},2);
        assertEquals(2,create.length());
        assertEquals(4,create.underlyingLength());
        assertEquals(2,create.offset());
        assertEquals(3,create.getDouble(0),1e-1);
        assertEquals(4,create.getDouble(1),1e-1);
    }

    @Test
    public void testArraySimple1() throws Exception {
       // INDArray array2 = Nd4j.linspace(1, 100000, 100000);

        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f});

        System.out.println("------------------------");

        System.out.println(Shape.isRowVectorShape(array.shapeInfoDataBuffer()));

        System.out.println("------------------------");

        System.out.println(array.shapeInfoDataBuffer());
    }

    @Test
    public void testArraySimple2() throws Exception {
        // INDArray array2 = Nd4j.linspace(1, 100000, 100000);

        INDArray array = Nd4j.zeros(100, 100);

        System.out.println("X0: ------------------------");

        System.out.println(Shape.isRowVectorShape(array.shapeInfoDataBuffer()));

        System.out.println("X1: ------------------------");

        System.out.println(array.shapeInfoDataBuffer());

        System.out.println("X2: ------------------------");

        INDArray slice = array.getRow(12);

        System.out.println("X3: ------------------------");

   //     AtomicAllocator.getInstance().getPointer(slice.shapeInfoDataBuffer());

        System.out.println("X4: ------------------------");

        System.out.println(Shape.isRowVectorShape(slice.shapeInfoDataBuffer()));

        System.out.println("X5: ------------------------");

        System.out.println(slice.shapeInfoDataBuffer());
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
    public void testReadWrite() throws Exception {
        INDArray write = Nd4j.linspace(1, 4, 4);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(write,dos);

        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        DataInputStream dis = new DataInputStream(bis);
        INDArray read = Nd4j.read(dis);
        assertEquals(write, read);
    }

    @Test
    public void testFlattened1() throws Exception {
        List<INDArray> test = new ArrayList<>();
        for (int x = 0; x < 100; x++) {
            INDArray array = Nd4j.linspace(0, 99, 100);
            test.add(array);
        }

        INDArray ret = Nd4j.toFlattened(test);

        assertEquals(10000, ret.length());
        for (int x = 0; x < 100; x++) {
            for (int y = 0; y < 100; y++) {
                assertEquals("X: ["+x+"], Y: ["+y+"] failed: ",y, ret.getFloat((x * 100) + y), 0.01f);
            }
        }
    }

    @Test
    public void testToFlattenedOrder() throws Exception {
        INDArray concatC = Nd4j.linspace(1,4,4).reshape('c',2,2);
        INDArray concatF = Nd4j.create(new int[]{2,2},'f');
        concatF.assign(concatC);
        INDArray assertionC = Nd4j.create(new double[]{1,2,3,4,1,2,3,4});
        //INDArray testC = Nd4j.toFlattened('c',concatC,concatF);
        //assertEquals(assertionC,testC);
        System.out.println("P0: --------------------------------------------------------");
        INDArray test = Nd4j.toFlattened('f',concatC,concatF);
        System.out.println("P1: --------------------------------------------------------");
        INDArray assertion = Nd4j.create(new double[]{1,3,2,4,1,3,2,4});
        assertEquals(assertion,test);
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
    public void testDataCreation1() throws Exception {
        BaseCudaDataBuffer buffer = (BaseCudaDataBuffer) Nd4j.createBuffer(10);

        AllocationPoint point = buffer.getAllocationPoint();

        CudaContext context = AtomicAllocator.getInstance().getContextPool().acquireContextForDevice(0);

        assertEquals(true, point.isActualOnHostSide());

        buffer.put(0, 10f);

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(false, point.isActualOnDeviceSide());

        buffer.put(1, 10f);

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(false, point.isActualOnDeviceSide());


        AtomicAllocator.getInstance().getPointer(buffer, context);

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(true, point.isActualOnDeviceSide());

        System.out.println("AM ------------------------------------");
        AtomicAllocator.getInstance().getHostPointer(buffer);

        System.out.println("AN ------------------------------------");

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(true, point.isActualOnDeviceSide());

    }

    @Test
    public void testDataCreation2() throws Exception {
        BaseCudaDataBuffer buffer = (BaseCudaDataBuffer) Nd4j.createBuffer(new int[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

        AllocationPoint point = buffer.getAllocationPoint();

        CudaContext context = AtomicAllocator.getInstance().getContextPool().acquireContextForDevice(0);

        assertEquals(true, point.isActualOnDeviceSide());
        assertEquals(false, point.isActualOnHostSide());

        System.out.println("AX --------------------------");
        buffer.put(0, 10f);
        System.out.println("AZ --------------------------");

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(false, point.isActualOnDeviceSide());

        buffer.put(1, 10f);

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(false, point.isActualOnDeviceSide());


        AtomicAllocator.getInstance().getPointer(buffer, context);

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(true, point.isActualOnDeviceSide());

        System.out.println("AM ------------------------------------");
        AtomicAllocator.getInstance().getHostPointer(buffer);

        System.out.println("AN ------------------------------------");

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(true, point.isActualOnDeviceSide());

        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());
    }

    @Test
    public void testDataCreation3() throws Exception {
        BaseCudaDataBuffer buffer = (BaseCudaDataBuffer) Nd4j.createBuffer(new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

        AllocationPoint point = buffer.getAllocationPoint();

        CudaContext context = AtomicAllocator.getInstance().getContextPool().acquireContextForDevice(0);

        assertEquals(true, point.isActualOnDeviceSide());
        assertEquals(false, point.isActualOnHostSide());

        System.out.println("AX --------------------------");
        buffer.put(0, 10f);
        System.out.println("AZ --------------------------");

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(false, point.isActualOnDeviceSide());

        buffer.put(1, 10f);

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(false, point.isActualOnDeviceSide());


        AtomicAllocator.getInstance().getPointer(buffer, context);

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(true, point.isActualOnDeviceSide());

        System.out.println("AM ------------------------------------");
        AtomicAllocator.getInstance().getHostPointer(buffer);

        System.out.println("AN ------------------------------------");

        assertEquals(true, point.isActualOnHostSide());

        assertEquals(true, point.isActualOnDeviceSide());

        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());
    }

    @Test
    public void testDataCreation4() throws Exception {
        BaseCudaDataBuffer buffer = (BaseCudaDataBuffer) Nd4j.createBuffer(new int[8]);

        AllocationPoint point = buffer.getAllocationPoint();

        assertEquals(true, point.isActualOnDeviceSide());
        assertEquals(false, point.isActualOnHostSide());

        System.out.println("AX --------------------------");
        buffer.put(0, 10f);
        System.out.println("AZ --------------------------");

        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());
    }

    @Test
    public void testDataCreation5() throws Exception {
        INDArray array = Nd4j.create(new double[][]{{0, 2}, {2, 1}});

        AllocationPoint pointMain = ((BaseCudaDataBuffer) array.data()).getAllocationPoint();

        AllocationPoint pointShape = ((BaseCudaDataBuffer) array.shapeInfoDataBuffer()).getAllocationPoint();

        assertEquals(true, pointShape.isActualOnDeviceSide());
        assertEquals(true, pointShape.isActualOnHostSide());

        assertEquals(false, pointMain.isActualOnDeviceSide());
        assertEquals(true, pointMain.isActualOnHostSide());

        assertEquals(AllocationStatus.DEVICE, pointMain.getAllocationStatus());
        assertEquals(AllocationStatus.DEVICE, pointShape.getAllocationStatus());
    }

    @Test
    public void testDataCreation6() throws Exception {
        INDArray array = Nd4j.create(new double[]{0, 1, 2, 3});

        AllocationPoint pointMain = ((BaseCudaDataBuffer) array.data()).getAllocationPoint();

        AllocationPoint pointShape = ((BaseCudaDataBuffer) array.shapeInfoDataBuffer()).getAllocationPoint();

        assertEquals(true, pointShape.isActualOnDeviceSide());
        assertEquals(true, pointShape.isActualOnHostSide());

        assertEquals(true, pointMain.isActualOnDeviceSide());
        assertEquals(false, pointMain.isActualOnHostSide());

        assertEquals(AllocationStatus.DEVICE, pointMain.getAllocationStatus());
        assertEquals(AllocationStatus.DEVICE, pointShape.getAllocationStatus());
    }

    @Test
    public void testDataCreation7() throws Exception {
        INDArray array = Nd4j.zeros(1500,150);

        AllocationPoint pointMain = ((BaseCudaDataBuffer) array.data()).getAllocationPoint();

        AllocationPoint pointShape = ((BaseCudaDataBuffer) array.shapeInfoDataBuffer()).getAllocationPoint();

        assertEquals(false, pointMain.isActualOnDeviceSide());
        assertEquals(true, pointMain.isActualOnHostSide());

        assertEquals(true, pointShape.isActualOnDeviceSide());
        assertEquals(true, pointShape.isActualOnHostSide());
    }

    @Test
    public void testDataCreation8() throws Exception {
        INDArray array = Nd4j.create(new float[]{1, 2, 3, 4, 5});

        AllocationPoint pointMain = ((BaseCudaDataBuffer) array.data()).getAllocationPoint();

        AllocationPoint pointShape = ((BaseCudaDataBuffer) array.shapeInfoDataBuffer()).getAllocationPoint();
        assertFalse(pointMain.isConstant());
        assertTrue(pointShape.isConstant());
    }

    @Test
    public void testDataCreation9() throws Exception {
        INDArray array = Nd4j.create(20);

        AllocationPoint pointMain = ((BaseCudaDataBuffer) array.data()).getAllocationPoint();

        AllocationPoint pointShape = ((BaseCudaDataBuffer) array.shapeInfoDataBuffer()).getAllocationPoint();

        assertFalse(pointMain.isConstant());
        assertTrue(pointShape.isConstant());

        assertEquals(false, pointMain.isActualOnDeviceSide());
        assertEquals(true, pointMain.isActualOnHostSide());
    }

    @Test
    public void testReshape1() throws Exception {
        INDArray arrayC = Nd4j.zeros(1000);
        INDArray arrayF = arrayC.reshape('f', 10, 100);

        assertEquals(102, arrayF.shapeInfoDataBuffer().getInt(7));
        assertEquals(1, arrayF.shapeInfoDataBuffer().getInt(6));

        System.out.println(arrayC.shapeInfoDataBuffer());
        System.out.println(arrayF.shapeInfoDataBuffer());

        System.out.println("Stride: " + arrayF.elementWiseStride());

        System.out.println(arrayF.shapeInfoDataBuffer());

        assertEquals('f', Shape.getOrder(arrayF));
        assertEquals(102, arrayF.shapeInfoDataBuffer().getInt(7));
        assertEquals(1, arrayF.shapeInfoDataBuffer().getInt(6));

        INDArray arrayZ = Nd4j.create(10, 100, 'f');
    }

    @Test
    public void testReshapeDup1() throws Exception {
        INDArray arrayC = Nd4j.create(10, 100);
        INDArray arrayF = arrayC.dup('f');

        System.out.println(arrayC.shapeInfoDataBuffer());
        System.out.println(arrayF.shapeInfoDataBuffer());

        assertEquals(102, arrayF.shapeInfoDataBuffer().getInt(7));
        assertEquals(1, arrayF.shapeInfoDataBuffer().getInt(6));
    }

    @Test
    public void testReshapeDup2() throws Exception {
        INDArray arrayC = Nd4j.create(5, 10, 100);
        INDArray arrayF = arrayC.dup('f');

        System.out.println(arrayC.shapeInfoDataBuffer());
        System.out.println(arrayF.shapeInfoDataBuffer());

        assertEquals(102, arrayF.shapeInfoDataBuffer().getInt(9));
        assertEquals(1, arrayF.shapeInfoDataBuffer().getInt(8));
    }
}