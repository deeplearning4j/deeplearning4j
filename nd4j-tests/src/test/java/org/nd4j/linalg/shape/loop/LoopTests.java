package org.nd4j.linalg.shape.loop;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.loop.coordinatefunction.CoordinateFunction;

import java.util.Arrays;

/**
 * Created by agibsonccc on 9/15/15.
 */
public class LoopTests extends BaseNd4jTest {
   @Test
    public void testLoop2d() {

       Shape.iterate(0, 2, new int[]{2, 2}, new int[2], 0, 2, new int[]{2, 3}, new int[2], new CoordinateFunction() {
           @Override
           public void process(int[]... coord) {
               for(int i = 0; i < coord.length; i++) {
                   System.out.println(Arrays.toString(coord[i]));
               }
           }
       });

    }

    @Override
    public char ordering() {
        return 'f';
    }
}
