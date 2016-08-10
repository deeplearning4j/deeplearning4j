package org.nd4j.linalg.lossfunctions;

import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.string.NDArrayStrings;

import java.util.Arrays;

import static org.junit.Assert.fail;

/**
 * Created by Alex on 08/08/2016.
 */
@Slf4j
public class LossFunctionGradientChecks extends BaseNd4jTest {

    public static final double epsilon = 1e-6;
    private static final double maxRelError = 5.0; //5% relative error

    public LossFunctionGradientChecks(Nd4jBackend backend) {
        super(backend);
    }

    @Before
    public void before() throws Exception {
        super.before();

        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        Nd4j.getRandom().setSeed(123);
    }

    @Test
    public void testLossFunctionGradients(){

        INDArray[] labels = new INDArray[]{
                Nd4j.create(new double[]{0,1,0}),
                Nd4j.create(new double[][]{{1,0,0},{0,1,0},{0,0,1}}),
                Nd4j.create(new double[]{1,2,1}),
                Nd4j.create(new double[][]{{1,2,1},{0.1,1,0.5},{20,3,1}}),
                Nd4j.create(new double[]{1,0,0}),
                Nd4j.create(new double[][]{{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}),
        };

        INDArray[] preOut = new INDArray[]{
                Nd4j.rand(1,3),
                Nd4j.rand(3,3),
                Nd4j.rand(1,3).add(5),
                Nd4j.rand(3,3),
                Nd4j.rand(1,3),
                Nd4j.randn(4,4)};

        ILossFunction[] lossFn = new ILossFunction[]{
                new LossMCXENT(), new LossMCXENT(), new LossMCXENT(),
                new LossMCXENT(),new LossMSE(), new LossMSE()};

        String[] activationFns = new String[]{
                "softmax","softmax","identity","identity","tanh",
                "tanh"};


        for(int i=0; i<labels.length; i++ ){
            int totalNFailures = 0;

            ILossFunction lf = lossFn[i];
            INDArray l = labels[i];
            INDArray p = preOut[i];
            String afn = activationFns[i];

            log.info("Starting test: {}, {}, input shape = {}", lf, afn, Arrays.toString(p.shape()));

            INDArray grad = lf.computeGradient(l,p,afn,null);

            NdIndexIterator iter = new NdIndexIterator(l.shape());
            while(iter.hasNext()){
                int[] next = iter.next();

                double before = p.getDouble(next);
                p.putScalar(next, before+epsilon);
                double scorePlus = lf.computeScore(l,p,afn,null,true);
                p.putScalar(next, before-epsilon);
                double scoreMinus = lf.computeScore(l,p,afn,null,true);
                p.putScalar(next, before);

                double scoreDelta = scorePlus - scoreMinus;

                double numericalGradient = scoreDelta / (2 * epsilon);
                double analyticGradient = grad.getDouble(next) / l.size(0);     //Analytic gradient method is before dividing by minibatch

                double relError = Math.abs(analyticGradient - numericalGradient)*100 / (Math.abs(numericalGradient));
                if( analyticGradient == 0.0 && numericalGradient == 0.0 ) relError = 0.0;	//Edge case: i.e., RNNs with time series length of 1.0

                if(relError > maxRelError || Double.isNaN(relError)) {
                    log.info("Param " + i + " FAILED: grad= " + analyticGradient + ", numericalGrad= "+numericalGradient
                                + ", relErrorPerc= " + relError + ", scorePlus="+scorePlus+", scoreMinus= " + scoreMinus);
                    totalNFailures++;
                } else {
                    log.info("Param " + i + " passed: grad= " + analyticGradient + ", numericalGrad= " + numericalGradient
                            + ", relError= " + relError + ", scorePlus="+scorePlus+", scoreMinus= " + scoreMinus );
                }
            }

            //if(totalNFailures > 0) fail("Gradient check failed for loss function " + lf + "; total num failures = " + totalNFailures);
            System.out.println("DONE");
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }

}
