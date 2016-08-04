package jcuda.jcublas.ops;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * @author raver119@gmail.com
 */
public class ShufflesTests {

    @Test
    public void testSimpleShuffle1() {
        INDArray array = Nd4j.zeros(10, 10);
        for (int x = 0; x < 10; x++) {
            array.getRow(x).assign(x);
        }

        System.out.println(array);

        OrderScanner2D scanner = new OrderScanner2D(array);

        assertArrayEquals(new float[]{0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, scanner.getMap(), 0.01f);

        Nd4j.shuffle(array, 1);

        System.out.println(array);

        assertTrue(scanner.compare(array));
    }




    public static class OrderScanner2D {
        private float[] map;

        public OrderScanner2D(INDArray data) {
            map = measureState(data);
        }

        public float[] measureState(INDArray data) {
            float[] result = new float[data.rows()];

            for (int x = 0; x < data.rows(); x++) {
                result[x] = data.getRow(x).getFloat(0);
            }

            return result;
        }

        public boolean compare(INDArray newData) {
            float[] newMap = measureState(newData);

            if (newMap.length != map.length) {
                System.out.println("Different map lengths");
                return false;
            }

            if (Arrays.equals(map, newMap)) {
                System.out.println("Maps are equal");
                return false;
            }

            for (int x = 0; x < newData.rows(); x++) {
                INDArray row = newData.getRow(x);
                for (int y = 0; y < row.lengthLong(); y++ ) {
                    if (Math.abs(row.getFloat(y) - newMap[x]) > Nd4j.EPS_THRESHOLD) {
                        System.out.print("Different data in a row");
                        return false;
                    }
                }
            }

            return true;
        }

        public float[] getMap() {
            return map;
        }
    }
}
