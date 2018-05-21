package org.nd4j.jita.memory.impl;

import junit.framework.TestCase;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.primitives.Pair;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.tad.DeviceTADManager;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class DelayedMemoryTest extends TestCase {

    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration()
                .setFirstMemory(AllocationStatus.DEVICE)
                .setMemoryModel(Configuration.MemoryModel.DELAYED)
                .allowMultiGPU(true)
                .enableDebug(true);
    }

    /**
     * This test should be run manually
     *
     * @throws Exception
     */
    @Test
    public void testDelayedAllocation2() throws  Exception {
        AtomicAllocator allocator = AtomicAllocator.getInstance();

        INDArray array = Nd4j.create(10, 10);

        AllocationPoint pointer = allocator.getAllocationPoint(array);

        PointersPair pair = pointer.getPointers();

        // pointers should be equal, device memory wasn't allocated yet
        assertEquals(pair.getDevicePointer(), pair.getHostPointer());

        //////////////

        AllocationPoint shapePointer = allocator.getAllocationPoint(array.shapeInfoDataBuffer());

        // pointers should be equal, device memory wasn't allocated yet
        assertEquals(shapePointer.getPointers().getDevicePointer(), shapePointer.getPointers().getHostPointer());

        assertEquals(pointer.getAllocationStatus(), AllocationStatus.HOST);
        assertEquals(shapePointer.getAllocationStatus(), AllocationStatus.HOST);

        float sum = array.sumNumber().floatValue();

        assertEquals(0.0f, sum, 0.0001f);

        shapePointer = allocator.getAllocationPoint(array.shapeInfoDataBuffer());
        pointer = allocator.getAllocationPoint(array);

        assertEquals(AllocationStatus.CONSTANT, shapePointer.getAllocationStatus());
        assertEquals(AllocationStatus.DEVICE, pointer.getAllocationStatus());

        // at this point all pointers show be different, since we've used OP (sumNumber)
        assertNotEquals(shapePointer.getPointers().getDevicePointer(), shapePointer.getPointers().getHostPointer());
    }

    @Test
    public void testDelayedAllocation1() throws Exception {
        final AtomicAllocator allocator = AtomicAllocator.getInstance();
        final int limit = 6;
        final INDArray[] arrays = new INDArray[limit];
        final Thread threads[] = new Thread[limit];
        final int cards[] = new int[limit];


        for (int c = 0; c < arrays.length; c++) {
            arrays[c] = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

            // we ensure, that both buffers are located in host memory now
            assertEquals(AllocationStatus.HOST, allocator.getAllocationPoint(arrays[c]).getAllocationStatus());
            assertEquals(AllocationStatus.HOST, allocator.getAllocationPoint(arrays[c].shapeInfoDataBuffer()).getAllocationStatus());
        }
/*
        for (int c = 0; c < arrays.length; c++) {
            System.out.println(arrays[c]);

            assertEquals(AllocationStatus.DEVICE, allocator.getAllocationPoint(arrays[c]).getAllocationStatus());
            assertEquals(AllocationStatus.CONSTANT, allocator.getAllocationPoint(arrays[c].shapeInfoDataBuffer()).getAllocationStatus());
        }
*/

        for (int c = 0; c < arrays.length; c++) {
            final int cnt = c;
            threads[cnt] = new Thread(new Runnable() {
                @Override
                public void run() {
                    float sum = arrays[cnt].sumNumber().floatValue();

                    cards[cnt] = allocator.getDeviceId();

                    assertEquals("Failed on C: " + cnt,15f, sum, 0.001f);
                }
            });

            threads[cnt].start();
        }

        for (int c = 0; c < arrays.length; c++ ){
            threads[c].join();
        }

        // check if all devices present in system were used
        for (int c = 0; c < arrays.length; c++) {
            assertNotEquals(allocator.getAllocationPoint(arrays[c]).getPointers().getDevicePointer(), allocator.getAllocationPoint(arrays[c]).getPointers().getHostPointer());
            assertNotEquals(allocator.getAllocationPoint(arrays[c].shapeInfoDataBuffer()).getPointers().getDevicePointer(), allocator.getAllocationPoint(arrays[c].shapeInfoDataBuffer()).getPointers().getHostPointer());
        }

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int c = 0; c < numDevices; c++) {
            assertTrue("Failed to find device ["+ c +"] in used devices", ArrayUtils.contains(cards, c));
        }
    }

    @Test
    public void testDelayedAllocation3() throws Exception {
        INDArray array = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

        AllocationPoint pointer = AtomicAllocator.getInstance().getAllocationPoint(array);

        PointersPair pair = pointer.getPointers();

        // pointers should be equal, device memory wasn't allocated yet
        assertEquals(pair.getDevicePointer(), pair.getHostPointer());

        assertEquals(2.0f, array.getFloat(1), 0.001f);

        assertEquals(pair.getDevicePointer(), pair.getHostPointer());
    }

    @Test
    public void testDelayedAllocation4() throws Exception {
        INDArray array = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

        AllocationPoint pointer = AtomicAllocator.getInstance().getAllocationPoint(array);

        PointersPair pair = pointer.getPointers();

        // pointers should be equal, device memory wasn't allocated yet
        assertEquals(pair.getDevicePointer(), pair.getHostPointer());

        assertEquals(2.0f, array.getFloat(1), 0.001f);

        assertEquals(pair.getDevicePointer(), pair.getHostPointer());


        String temp = System.getProperty("java.io.tmpdir");

        String outPath = FilenameUtils.concat(temp,"dl4jtestserialization.bin");

        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(outPath)))){
            Nd4j.write(array,dos);
        }

        INDArray in;
        try(DataInputStream dis = new DataInputStream(new FileInputStream(outPath))){
            in = Nd4j.read(dis);
        }


        assertEquals(AtomicAllocator.getInstance().getAllocationPoint(in).getPointers().getDevicePointer(), AtomicAllocator.getInstance().getAllocationPoint(in).getPointers().getHostPointer());

        assertEquals(array, in);
    }

    @Test
    public void testDelayedTAD1() throws Exception {
        TADManager tadManager = new DeviceTADManager();

        INDArray array = Nd4j.create(128, 256);

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(array, new int[]{0});

        DataBuffer tadBuffer = tadBuffers.getFirst();
        DataBuffer offBuffer = tadBuffers.getSecond();

        AllocationPoint pointTad = AtomicAllocator.getInstance().getAllocationPoint(tadBuffer);
        AllocationPoint pointOff = AtomicAllocator.getInstance().getAllocationPoint(offBuffer);

        assertEquals(AllocationStatus.CONSTANT, pointTad.getAllocationStatus());
        assertEquals(AllocationStatus.DEVICE, pointOff.getAllocationStatus());
    }

    @Test
    public void testDelayedDup1() throws Exception {
        INDArray array = Nd4j.linspace(1,1000, 1000).reshape(10, 10, 10);

        AllocationPoint pointShape = AtomicAllocator.getInstance().getAllocationPoint(array.shapeInfoDataBuffer());
        AllocationPoint pointArray = AtomicAllocator.getInstance().getAllocationPoint(array);

        assertEquals(AllocationStatus.HOST, pointArray.getAllocationStatus());
        assertEquals(AllocationStatus.HOST, pointShape.getAllocationStatus());

        float sum = array.sumNumber().floatValue();

        pointShape = AtomicAllocator.getInstance().getAllocationPoint(array.shapeInfoDataBuffer());
        pointArray = AtomicAllocator.getInstance().getAllocationPoint(array);

        assertEquals(AllocationStatus.DEVICE, pointArray.getAllocationStatus());
        assertEquals(AllocationStatus.CONSTANT, pointShape.getAllocationStatus());

        INDArray dup = array.dup();

        AllocationPoint dupShape = AtomicAllocator.getInstance().getAllocationPoint(dup.shapeInfoDataBuffer());
        AllocationPoint dupArray = AtomicAllocator.getInstance().getAllocationPoint(dup);

        assertEquals(AllocationStatus.DEVICE, dupArray.getAllocationStatus());
        assertEquals(AllocationStatus.CONSTANT, dupShape.getAllocationStatus());
    }

    @Test
    public void testDelayedZeroes1() throws Exception {
        INDArray zeroes = Nd4j.zeros(10);

        zeroes.putScalar(1, 1f);
        zeroes.putScalar(2, 1f);

        float sum = zeroes.sumNumber().floatValue();

        assertEquals(2f, sum, 0.0001f);
    }
}
