package org.deeplearning4j.plot;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;

/**
 * @author Adam Gibson
 */
public class RenderTest {
    @Test
    public void testRender() {
        INDArray test = Nd4j.rand(new int[]{328,400,4});
        PlotFilters render = new PlotFilters();
        INDArray rendered = render.render(test,1,1);
        assertArrayEquals(new int[]{7619, 95}, rendered.shape());
    }

}
