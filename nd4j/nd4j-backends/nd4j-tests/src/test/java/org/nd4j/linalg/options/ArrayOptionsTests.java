package org.nd4j.linalg.options;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

@Slf4j
@RunWith(Parameterized.class)
public class ArrayOptionsTests extends BaseNd4jTest {
    private static long[] shapeInfo;

    public ArrayOptionsTests(Nd4jBackend backend) {
        super(backend);
    }


    @Before
    public void setUp() throws Exception {
        shapeInfo = new long[]{2, 2, 2, 2, 1, 0, 1, 99};
    }

    @Test
    public void testArrayType_0() {
        assertEquals(ArrayType.DENSE, ArrayOptionsHelper.arrayType(shapeInfo));
    }

    @Test
    public void testArrayType_1() {
        ArrayOptionsHelper.setOptionBit(shapeInfo, ArrayType.EMPTY);

        assertEquals(ArrayType.EMPTY, ArrayOptionsHelper.arrayType(shapeInfo));
    }

    @Test
    public void testArrayType_2() {
        ArrayOptionsHelper.setOptionBit(shapeInfo, ArrayType.SPARSE);

        assertEquals(ArrayType.SPARSE, ArrayOptionsHelper.arrayType(shapeInfo));
    }

    @Test
    public void testArrayType_3() {
        ArrayOptionsHelper.setOptionBit(shapeInfo, ArrayType.COMPRESSED);

        assertEquals(ArrayType.COMPRESSED, ArrayOptionsHelper.arrayType(shapeInfo));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
