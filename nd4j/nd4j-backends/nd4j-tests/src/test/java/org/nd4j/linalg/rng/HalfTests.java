package org.nd4j.linalg.rng;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import static junit.framework.TestCase.assertTrue;

/**
 * This test suit contains tests related to Half precision and RNG
 */
@Slf4j
@RunWith(Parameterized.class)
public class HalfTests extends BaseNd4jTest {

    private DataBuffer.Type initialType;

    public HalfTests(Nd4jBackend backend) {
        super(backend);
    }

    @Before
    public void setUp() throws Exception {
        if (!Nd4j.getExecutioner().getClass().getSimpleName().toLowerCase().contains("cuda"))
            return;

        initialType = Nd4j.dataType();
        Nd4j.setDataType(DataBuffer.Type.HALF);
    }

    @After
    public void tearDown() throws Exception {
        if (!Nd4j.getExecutioner().getClass().getSimpleName().toLowerCase().contains("cuda"))
            return;

        Nd4j.setDataType(initialType);
    }

    @Test
    public void testRandomNorman_1() {
        val array = Nd4j.randn(new long[]{20, 30});

        val sum = Transforms.abs(array).sumNumber().doubleValue();

        assertTrue(sum > 0.0);
    }

    public char ordering() {
        return 'c';
    }

}
