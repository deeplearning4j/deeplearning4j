package jcuda.jcublas.ops;

import org.junit.Before;
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

import java.util.Arrays;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.nd4j.linalg.api.shape.Shape.newShapeNoCopy;

/**
 * @author raver119@gmail.com
 */
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
}
