package jcuda.jcublas.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.util.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.nd4j.linalg.api.shape.Shape.newShapeNoCopy;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SporadicTests {

    @Before
    public void setUp() throws Exception {
        CudaEnvironment.getInstance().getConfiguration().enableDebug(true).setVerbose(false);
    }

    @Test
    public void testIsMax1() throws Exception {
        int[] shape = new int[]{2,2};
        int length = 4;
        int alongDimension = 0;

        INDArray arrC = Nd4j.linspace(1,length, length).reshape('c',shape);
        Nd4j.getExecutioner().execAndReturn(new IsMax(arrC, alongDimension));

        //System.out.print(arrC);
        assertEquals(0.0, arrC.getDouble(0), 0.1);
        assertEquals(0.0, arrC.getDouble(1), 0.1);
        assertEquals(1.0, arrC.getDouble(2), 0.1);
        assertEquals(1.0, arrC.getDouble(3), 0.1);
    }

    @Test
    public void randomStrangeTest() {
        DataBuffer.Type type = Nd4j.dataType();
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        int a=9;
        int b=2;
        int[] shapes = new int[a];
        for (int i = 0; i < a; i++) {
            shapes[i] = b;
        }
        INDArray c = Nd4j.linspace(1, (int) (100 * 1 + 1 + 2), (int) Math.pow(b, a)).reshape(shapes);
        c=c.sum(0);
        double[] d = c.data().asDouble();
        System.out.println("d: " + Arrays.toString(d));

        DataTypeUtil.setDTypeForContext(type);
    }

    @Test
    public void testBroadcastWithPermute(){
        Nd4j.getRandom().setSeed(12345);
        int length = 4*4*5*2;
        INDArray arr = Nd4j.linspace(1,length,length).reshape('c',4,4,5,2).permute(2,3,1,0);
//        INDArray arr = Nd4j.linspace(1,length,length).reshape('f',4,4,5,2).permute(2,3,1,0);
        ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();
        INDArray arrDup = arr.dup('c');
        ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

        INDArray row = Nd4j.rand(1,2);
        assertEquals(row.length(), arr.size(1));
        assertEquals(row.length(), arrDup.size(1));

        assertEquals(arr,arrDup);



        INDArray first =  Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(arr,    row, Nd4j.createUninitialized(arr.shape(), 'c'), 1));
        INDArray second = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(arrDup, row, Nd4j.createUninitialized(arr.shape(), 'c'), 1));

        System.out.println("A1: " + Arrays.toString(arr.shapeInfoDataBuffer().asInt()));
        System.out.println("A2: " + Arrays.toString(first.shapeInfoDataBuffer().asInt()));
        System.out.println("B1: " + Arrays.toString(arrDup.shapeInfoDataBuffer().asInt()));
        System.out.println("B2: " + Arrays.toString(second.shapeInfoDataBuffer().asInt()));

        INDArray resultSameStrides = Nd4j.zeros(new int[]{4,4,5,2},'c').permute(2,3,1,0);
        assertArrayEquals(arr.stride(), resultSameStrides.stride());
        INDArray third = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(arr, row, resultSameStrides, 1));

        assertEquals(second, third);    //Original and result w/ same strides: passes
        assertEquals(first,second);     //Original and result w/ different strides: fails
    }

    @Test
    public void testBroadcastEquality1() {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'f');
        INDArray array2 = Nd4j.zeros(new int[]{4, 5}, 'f');
        INDArray row = Nd4j.create(new float[]{1, 2, 3, 4, 5});

        array.addiRowVector(row);

        System.out.println(array);

        System.out.println("-------");

        ScalarAdd add = new ScalarAdd(array2, row, array2, array2.length(), 0.0f);
        add.setDimension(0);
        Nd4j.getExecutioner().exec(add);

        System.out.println(array2);
        assertEquals(array, array2);
    }

    @Test
    public void testBroadcastEquality2() {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'c');
        INDArray array2 = Nd4j.zeros(new int[]{4, 5}, 'c');
        INDArray column = Nd4j.create(new float[]{1, 2, 3, 4}).reshape(4,1);

        array.addiColumnVector(column);

        System.out.println(array);

        System.out.println("-------");

        ScalarAdd add = new ScalarAdd(array2, column, array2, array2.length(), 0.0f);
        add.setDimension(1);
        Nd4j.getExecutioner().exec(add);

        System.out.println(array2);
        assertEquals(array, array2);

    }

    @Test
    public void testIAMax1() throws Exception {
        INDArray arrayX = Nd4j.rand('c', 128000, 4);

        Nd4j.getExecutioner().exec(new IAMax(arrayX), 1);

        long time1 = System.nanoTime();
        for (int i = 0; i < 10000; i++) {
            Nd4j.getExecutioner().exec(new IAMax(arrayX), 1);
        }
        long time2 = System.nanoTime();

        System.out.println("Time: " + ((time2 - time1) / 10000));
    }

    @Test
    public void testLocality() {
        INDArray array = Nd4j.create(new float[]{1,2,3,4,5,6,7,8,9});

        AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(array);
        assertEquals(true, point.isActualOnDeviceSide());

        INDArray arrayR = array.reshape('f', 3, 3);

        AllocationPoint pointR = AtomicAllocator.getInstance().getAllocationPoint(arrayR);
        assertEquals(true, pointR.isActualOnDeviceSide());

        INDArray arrayS = Shape.newShapeNoCopy(array,new int[]{3,3}, true);

        AllocationPoint pointS = AtomicAllocator.getInstance().getAllocationPoint(arrayS);
        assertEquals(true, pointS.isActualOnDeviceSide());

        INDArray arrayL = Nd4j.create(new int[]{3,4,4,4},'c');

        AllocationPoint pointL = AtomicAllocator.getInstance().getAllocationPoint(arrayL);
        assertEquals(true, pointL.isActualOnDeviceSide());
    }

    @Test
    public void testEnvironment() throws Exception {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'f');
        Properties properties = Nd4j.getExecutioner().getEnvironmentInformation();

        System.out.println("Props: " + properties.toString());
    }


    /**
     * This is special test that checks for memory alignment
     * @throws Exception
     */
    @Test
    @Ignore
    public void testDTypeSpam() throws Exception {
        Random rnd = new Random();
        for(int i = 0; i < 100; i++) {
            DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
            float rand[] = new float[rnd.nextInt(10) + 1];
            for (int x = 0; x < rand.length; x++) {
                rand[x] = rnd.nextFloat();
            }
            Nd4j.getConstantHandler().getConstantBuffer(rand);

            int shape[] = new int[rnd.nextInt(3)+2];
            for (int x = 0; x < shape.length; x++) {
                shape[x] = rnd.nextInt(100) + 2;
            }

            DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
            INDArray array = Nd4j.rand(shape);
            BooleanIndexing.applyWhere(array, Conditions.lessThan(rnd.nextDouble()), rnd.nextDouble());
        }
    }

    @Test
    public void testIsView() {
        INDArray array = Nd4j.zeros(100, 100);

        assertFalse(array.isView());
    }


    @Test
    public void testReplicate1() throws Exception {
        INDArray array = Nd4j.create(new float[]{1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f});
        INDArray exp = Nd4j.create(new float[]{2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f});

        log.error("Array length: {}", array.length());

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        final DeviceLocalNDArray locals = new DeviceLocalNDArray(array);

        Thread[] threads = new Thread[numDevices];
        for (int t = 0; t < numDevices; t++) {
            threads[t] = new Thread(new Runnable() {
                @Override
                public void run() {
                    locals.get().addi(1f);
                    locals.get().addi(0f);
                }
            });
            threads[t].start();
        }


        for (int t = 0; t < numDevices; t++) {
            threads[t].join();
        }


        for (int t = 0; t < numDevices; t++) {
            exp.addi(0.0f);
            assertEquals(exp, locals.get(t));
        }
    }

    @Test
    public void testReplicate2() throws Exception {
        DataBuffer buffer = Nd4j.createBuffer(new float[] {1f, 1f, 1f, 1f, 1f});

        DataBuffer buffer2 = Nd4j.getAffinityManager().replicateToDevice(1, buffer);

        assertEquals(1f, buffer2.getFloat(0), 0.001f);
    }


    @Test
    public void testReplicate3() throws Exception {
        INDArray array = Nd4j.ones(10, 10);
        INDArray exp = Nd4j.create(10).assign(10f);

        log.error("Array length: {}", array.length());

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        final DeviceLocalNDArray locals = new DeviceLocalNDArray(array);

        Thread[] threads = new Thread[numDevices];
        for (int t = 0; t < numDevices; t++) {
            threads[t] = new Thread(new Runnable() {
                @Override
                public void run() {

                    AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(locals.get());
                    log.error("Point deviceId: {}; current deviceId: {}", point.getDeviceId(), Nd4j.getAffinityManager().getDeviceForCurrentThread());


                    INDArray sum = locals.get().sum(1);
                    INDArray localExp = Nd4j.create(10).assign(10f);

                    assertEquals(localExp, sum);
                }
            });
            threads[t].start();
        }


        for (int t = 0; t < numDevices; t++) {
            threads[t].join();
        }


        for (int t = 0; t < numDevices; t++) {

            AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(locals.get(t));
            log.error("Point deviceId: {}; current deviceId: {}", point.getDeviceId(), Nd4j.getAffinityManager().getDeviceForCurrentThread());

            exp.addi(0.0f);
            assertEquals(exp, locals.get(t).sum(0));

            log.error("Point after: {}", point.getDeviceId());
        }
    }


    @Test
    public void testReplicate4() throws Exception {
        INDArray array = Nd4j.create(3,3);

        array.getRow(1).putScalar(0, 1f);
        array.getRow(1).putScalar(1, 1f);
        array.getRow(1).putScalar(2, 1f);

        final DeviceLocalNDArray locals = new DeviceLocalNDArray(array);

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int t = 0; t < numDevices; t++) {
            assertEquals(3, locals.get(t).sumNumber().floatValue(), 0.001f);
        }
    }


    @Test
    public void testReplicate5() throws Exception {
        INDArray array = Nd4j.create(3, 3);

        log.error("Original: Host pt: {}; Dev pt: {}", AtomicAllocator.getInstance().getAllocationPoint(array).getPointers().getHostPointer().address(), AtomicAllocator.getInstance().getAllocationPoint(array).getPointers().getDevicePointer().address());

        final DeviceLocalNDArray locals = new DeviceLocalNDArray(array);



        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int t = 0; t < numDevices; t++) {
            log.error("deviceId: {}; Host pt: {}; Dev pt: {}", t, AtomicAllocator.getInstance().getAllocationPoint(locals.get(t)).getPointers().getHostPointer().address(), AtomicAllocator.getInstance().getAllocationPoint(locals.get(t)).getPointers().getDevicePointer().address());
        }


        Thread[] threads = new Thread[numDevices];
        for (int t = 0; t < numDevices; t++) {
            threads[t] = new Thread(new Runnable() {
                @Override
                public void run() {
                    AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(locals.get());
                    log.error("deviceId: {}; Host pt: {}; Dev pt: {}", Nd4j.getAffinityManager().getDeviceForCurrentThread(), point.getPointers().getHostPointer().address(), point.getPointers().getDevicePointer().address());

                }
            });
            threads[t].start();
        }


        for (int t = 0; t < numDevices; t++) {
            threads[t].join();
        }
    }


    @Test
    public void testEnvInfo() throws Exception {
        Properties props = Nd4j.getExecutioner().getEnvironmentInformation();

        List<Map<String, Object>> list = (List<Map<String,Object>>) props.get("cuda.devicesInformation");
        for (Map<String, Object> map: list) {
            log.error("devName: {}", map.get("cuda.deviceName"));
            log.error("totalMem: {}", map.get("cuda.totalMemory"));
            log.error("freeMem: {}", map.get("cuda.freeMemory"));
            System.out.println();
        }
    }

    @Test
    public void testStd() {
        INDArray values = Nd4j.linspace(1, 4, 4).transpose();

        double corrected = values.std(true, 0).getDouble(0);
        double notCorrected = values.std(false, 0).getDouble(0);

        System.out.println(String.format("Corrected: %f, non corrected: %f", corrected, notCorrected));

    }
}
