package org.deeplearning4j.nn.conf.preprocessor;

import static org.junit.Assert.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.junit.Test;

/**
 * Created by merlin on 7/31/15.
 */
public class ReshapePreProcessorTest {

        private static int rows = 28;
        private static int cols = 28;
        INDArray in2D = Nd4j.create(1, 784);
        INDArray in3D = Nd4j.create(1, 784, 7);
        INDArray in4D = Nd4j.create(20, 1, 28, 28);


        @Test
        public void testPreProcessor2to4() {
                ReshapePreProcessor processor = new ReshapePreProcessor(new int[]{1, 784}, new int[]{1, 1, 28, 28});

                INDArray check2to4 = processor.preProcess(in2D,null);
                int val2to4 = check2to4.shape().length;
                assertTrue(val2to4 == 4);
        }
        @Test
        public void testPreProcessor4to2() {
                ReshapePreProcessor processor = new ReshapePreProcessor(new int[] {20, 1, 28, 28}, new int[]{20, 784});
                INDArray result2 = processor.preProcess(in4D,null);
                int val4to4 = result2.shape().length;
                assertTrue(val4to4 == 2);
        }


        @Test
        public void testPreProcessorBackprop2to4() {
                ReshapePreProcessor processor = new ReshapePreProcessor(new int[]{1, 784}, new int[]{1, 1, 28, 28});

                INDArray check2to2 = processor.backprop(in2D,null);
                int val2to2 = check2to2.shape().length;
                assertTrue(val2to2 == 2);
        }

        @Test
        public void testPreProcessorBackprop3to2() {
                ReshapePreProcessor processor = new ReshapePreProcessor(new int[]{1, 784, 7}, new int[] {1, 5488});

                INDArray check3to2 = processor.backprop(in3D,null);
                int val3to2 = check3to2.shape().length;
                assertTrue(val3to2 == 3);

        }



}