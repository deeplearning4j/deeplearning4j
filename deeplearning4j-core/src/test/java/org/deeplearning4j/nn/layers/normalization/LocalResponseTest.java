package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * Created by nyghtowl on 10/30/15.
 */
public class LocalResponseTest {

    private INDArray x = Nd4j.create(new double[]{
            0.88128096, -0.96666986, -0.61832994,  0.26418415,  0.05694608,
            0.2950289 ,  0.99222249,  0.24541704,  0.4219842 ,  0.96430975,
            0.19299535, -0.06658337, -0.27603117,  0.24216647,  0.21834095,
            0.03863283, -0.82313406, -0.37236378, -0.77667993,  0.66295379,
            -0.34406275, -0.25924176,  0.26652309, -0.58964926, -0.46907067,
            0.34666502,  0.81208313, -0.17042427, -0.22470538,  0.8348338 ,
            0.50494033,  0.45004508,  0.58735144, -0.87217808, -0.74788797,
            -0.04363599,  0.72276866,  0.52476895, -0.52383977,  0.1311436 ,
            0.2628099 ,  0.77274454,  0.86400729, -0.35246921, -0.03399619,
            -0.502312  ,  0.42834607,  0.85534132,  0.90083021,  0.24571614,
            0.63058525, -0.82919437,  0.57236177, -0.0913529 , -0.7102778 ,
            0.81631756, -0.89004314,  0.43995622, -0.26112801, -0.76135367,
            0.65180862, -0.54667377,  0.94908774,  0.59298772,  0.36457643,
            0.58892179, -0.52951556,  0.31559938, -0.55268252,  0.8272332 ,
            0.37911707, -0.96299696, -0.40717798,  0.43324658,  0.2589654 ,
            -0.15605508,  0.96334064, -0.31666604,  0.19781154,  0.09908111,
            0.64796048, -0.99037546,  0.67919868,  0.43810204
    }, new int[] {2,7,3,2});

    private INDArray yExpected = Nd4j.create(new double[]{
            0.52365203, -0.5745584 , -0.36757737,  0.15702588,  0.03385123,
            0.17541009,  0.58943964,  0.14584421,  0.25084497,  0.57315239,
            0.11472178, -0.03958213, -0.16396555,  0.14390601,  0.12975887,
            0.02296177, -0.48928441, -0.22130304, -0.46144658,  0.39406449,
            -0.20447753, -0.15404278,  0.15839269, -0.35045122, -0.27873584,
            0.20604356,  0.48260604, -0.10130161, -0.13353914,  0.49606363,
            0.30005923,  0.26749433,  0.34905838, -0.51843025, -0.44457183,
            -0.02593014,  0.42960015,  0.31195952, -0.3113277 ,  0.07795486,
            0.15622793,  0.45925347,  0.51334482, -0.20951259, -0.02020523,
            -0.29854962,  0.25464078,  0.50833746,  0.53513777,  0.14604072,
            0.37465439, -0.49276752,  0.34023748, -0.05428484, -0.42189589,
            0.48515803, -0.52874757,  0.26138692, -0.15521783, -0.45226377,
            0.38725017, -0.32489423,  0.56381026,  0.35233612,  0.21664843,
            0.34991751, -0.31468491,  0.18756773, -0.32832105,  0.491464  ,
            0.22527787, -0.57214104, -0.24202688,  0.2575524 ,  0.15388405,
            -0.09271994,  0.57244818, -0.18818023,  0.11759805,  0.05890742,
            0.38516517, -0.58850795,  0.40362193,  0.26037796
    }, new int[] {2,7,3,2});

    @Test
    public void testActivate(){
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .seed(123)
                .layer(new LocalResponseNormalization.Builder().k(2)
                        .build())
                .build();

        Layer layer = LayerFactories.getFactory(new LocalResponseNormalization()).create(conf);
        INDArray yActual = layer.activate(x);

        // Precision is off from the expected results because expected results generated in numpy
        assertEquals(yExpected.getDouble(5), yActual.getDouble(5), 1e-4);
        assertEquals(yExpected.getDouble(10), yActual.getDouble(10), 1e-4);
        }

    @Test
    public void testBackpropGradient(){

    }

}
