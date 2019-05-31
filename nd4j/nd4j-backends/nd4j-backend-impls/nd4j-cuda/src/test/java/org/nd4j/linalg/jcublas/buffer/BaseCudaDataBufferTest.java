package org.nd4j.linalg.jcublas.buffer;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

@Slf4j
public class BaseCudaDataBufferTest {

    @Test
    public void testShapeCache_1() {
        val x = Nd4j.create(DataType.FLOAT, 3, 5);

        assertEquals(DataType.FLOAT, x.dataType());
        assertArrayEquals(new long[]{3, 5}, x.shape());
        assertArrayEquals(new long[]{5, 1}, x.stride());
        assertEquals(1, x.elementWiseStride());
        assertEquals('c', x.ordering());

        val pair = Nd4j.getShapeInfoProvider().createShapeInformation(x.shape(), x.stride(), x.elementWiseStride(), x.ordering(), x.dataType(), x.isEmpty());
        val db = pair.getFirst();
        val jvm = pair.getSecond();

        log.info("array shapeInfo: {}", x.shapeInfoJava());
        log.info("direct shapeInfo: {}", jvm);

        val pointX = AtomicAllocator.getInstance().getAllocationPoint(x.shapeInfoDataBuffer());
        val pointM = AtomicAllocator.getInstance().getAllocationPoint(db);

        assertNotNull(pointX);
        assertNotNull(pointM);

        assertNotNull(pointX.getHostPointer());
        assertNotNull(pointX.getDevicePointer());

        assertNotNull(pointM.getHostPointer());
        assertNotNull(pointM.getDevicePointer());


        log.info("X hPtr: {}; dPtr: {}", pointX.getHostPointer().address(), pointX.getDevicePointer().address());
        log.info("M hPtr: {}; dPtr: {}", pointM.getHostPointer().address(), pointM.getDevicePointer().address());

        assertEquals(pointM.getHostPointer().address(), pointX.getHostPointer().address());
        assertEquals(pointM.getDevicePointer().address(), pointX.getDevicePointer().address());

        assertArrayEquals(x.shapeInfoJava(), jvm);
    }

    @Test
    public void testTadCache_1() {
        val x = Nd4j.create(DataType.FLOAT, 3, 5);
        val row = x.getRow(1);
        val tad = x.tensorAlongDimension(1, 1);

        val pointX = AtomicAllocator.getInstance().getAllocationPoint(row.shapeInfoDataBuffer());
        val pointM = AtomicAllocator.getInstance().getAllocationPoint(tad.shapeInfoDataBuffer());

        assertNotNull(pointX);
        assertNotNull(pointM);

        assertNotNull(pointX.getHostPointer());
        assertNotNull(pointX.getDevicePointer());

        assertNotNull(pointM.getHostPointer());
        assertNotNull(pointM.getDevicePointer());


        log.info("X hPtr: {}; dPtr: {}", pointX.getHostPointer().address(), pointX.getDevicePointer().address());
        log.info("M hPtr: {}; dPtr: {}", pointM.getHostPointer().address(), pointM.getDevicePointer().address());

        assertEquals(pointM.getHostPointer().address(), pointX.getHostPointer().address());
        assertEquals(pointM.getDevicePointer().address(), pointX.getDevicePointer().address());

        assertArrayEquals(row.shapeInfoJava(), tad.shapeInfoJava());
    }
}