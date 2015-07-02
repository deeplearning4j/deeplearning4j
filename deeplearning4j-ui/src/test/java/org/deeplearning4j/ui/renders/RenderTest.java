package org.deeplearning4j.ui.renders;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class RenderTest {
    @Test
    public void testRender() {
        INDArray test = Nd4j.rand(new int[]{328,400,4});
        FilterRenderer render = new FilterRenderer();
        INDArray rendered = render.render(test,1,1);
        assertArrayEquals(new int[]{7619,95},rendered.shape());
    }

}
