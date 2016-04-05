package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.assertTrue;

/**Gradient checking tests with masking (i.e., variable length time series inputs, one-to-many and many-to-one etc)
 */
public class GradientCheckTestsMasking {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;

    static {
        //Force Nd4j initialization, then set data type to double:
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void gradientCheckMaskingOutputSimple(){

        int timeSeriesLength = 5;
        boolean[][] mask = new boolean[5][0];
        mask[0] = new boolean[]{true,true,true,true,true};          //No masking
        mask[1] = new boolean[]{false,true,true,true,true};         //mask first output time step
        mask[2] = new boolean[]{false,false,false,false,true};      //time series classification: mask all but last
        mask[3] = new boolean[]{false,false,true,false,true};       //time series classification w/ variable length TS
        mask[4] = new boolean[]{true,true,true,false,true};         //variable length TS

        int nIn = 4;
        int layerSize = 3;
        int nOut = 2;


        Random r = new Random(12345L);
        INDArray input = Nd4j.zeros(1, nIn, timeSeriesLength);
        for( int m=0; m<1; m++ ){
            for( int j=0; j<nIn; j++ ){
                for( int k=0; k<timeSeriesLength; k++ ){
                    input.putScalar(new int[]{m,j,k},r.nextDouble() - 0.5);
                }
            }
        }

        INDArray labels = Nd4j.zeros(1,nOut,timeSeriesLength);
        for( int m=0; m<1; m++){
            for( int j=0; j<timeSeriesLength; j++ ){
                int idx = r.nextInt(nOut);
                labels.putScalar(new int[]{m,idx,j}, 1.0f);
            }
        }

        for(int i=0; i<mask.length; i++ ) {

            //Create mask array:
            INDArray maskArr = Nd4j.create(1,timeSeriesLength);
            for(int j=0; j<mask[i].length; j++){
                maskArr.putScalar(new int[]{0,j},mask[i][j] ? 1.0 : 0.0);
            }

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .regularization(false)
                    .seed(12345L)
                    .list()
                    .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).weightInit(WeightInit.DISTRIBUTION)
                            .dist(new NormalDistribution(0, 1)).updater(Updater.NONE).build())
                    .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax").nIn(layerSize).nOut(nOut)
                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).updater(Updater.NONE).build())
                    .pretrain(false).backprop(true)
                    .build();
            MultiLayerNetwork mln = new MultiLayerNetwork(conf);
            mln.init();

            mln.setLayerMaskArrays(null,maskArr);

            boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                    PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

            String msg = "gradientCheckMaskingOutputSimple() - timeSeriesLength=" + timeSeriesLength + ", miniBatchSize=" + 1;
            assertTrue(msg, gradOK);

        }
    }

}
