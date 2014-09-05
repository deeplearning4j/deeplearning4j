package org.nd4j.linalg.lossfunctions;

import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

/**
 * Testing loss function
 *
 * @author Adam Gibson
 */
public abstract class LossFunctionTests {

    private static Logger log = LoggerFactory.getLogger(LossFunctionTests.class);


    @Test
    public void testReconEntropy() {
        Nd4j.factory().setOrder('f');
        INDArray input = Nd4j.create(
                new double[]{1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new int[]{7, 6});
        INDArray w = Nd4j.create(
                new double[]{-0.005221025740321007, -0.0025006434506737304, -0.013585431005440437, -0.021996946700655134, 0.007678447599654643, -0.0037941287958231052, -0.014933056402715545, -0.012875289265542541, 0.001635482018910717, 0.00893829162129914, 0.017003519496588012, -0.004271078749979736, 0.0015816435136811352, 0.008638074705740708, -0.004393004605647038, -0.006249587919004255, -0.011017655538216209, -0.0015862988109404338, 0.01079760516931169, -0.0010491291520692704, 0.006626023289526534, 0.004658989751677583, -0.0022132443508813535, -0.00979834812384658}, new int[]{6, 4});
        INDArray vBias = Nd4j.create(
                new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new int[]{6});
        INDArray hBias = Nd4j.create(
                new double[]{0.0, 0.0, 0.0, 0.0}, new int[]{4});
        double reconEntropy = LossFunctions.reconEntropy(input,hBias,vBias,w, Activations.sigmoid());
        double assertion = -0.5937198421625942;
        assertEquals(assertion,reconEntropy,1e-1);
    }


    @Test
    public void testRMseXent() {
        INDArray in = Nd4j.create(new float[][]{{1, 2}, {3, 4}});
        INDArray out = Nd4j.create(new float[][]{{5, 6}, {7, 8}});
        float diff = LossFunctions.score(in, LossFunctions.LossFunction.RMSE_XENT,out,0,false);
        assertEquals(8,diff,1e-1);
    }

    @Test
    public void testMcXent() {
        INDArray in = Nd4j.create(new float[][]{{1, 2}, {3, 4}});
        INDArray out = Nd4j.create(new float[][]{{5, 6}, {7, 8}});
        LossFunctions.score(in, LossFunctions.LossFunction.MCXENT,out,0,false);
    }


}
