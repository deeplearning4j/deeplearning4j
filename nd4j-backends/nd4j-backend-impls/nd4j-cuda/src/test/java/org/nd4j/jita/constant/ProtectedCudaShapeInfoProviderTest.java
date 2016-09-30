package org.nd4j.jita.constant;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.ShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * Created by raver119 on 30.09.16.
 */
public class ProtectedCudaShapeInfoProviderTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testPurge1() throws Exception {
        INDArray array = Nd4j.create(10, 10);

        ProtectedCudaShapeInfoProvider provider = (ProtectedCudaShapeInfoProvider) ProtectedCudaShapeInfoProvider.getInstance();

        assertEquals(true, provider.protector.containsDataBuffer(0, new ShapeDescriptor(array.shape(), array.stride(),0, array.elementWiseStride(), array.ordering())));

        provider.purgeCache();

        assertEquals(false, provider.protector.containsDataBuffer(0, new ShapeDescriptor(array.shape(), array.stride(),0, array.elementWiseStride(), array.ordering())));
    }

}