package org.nd4j.linalg.lossfunctions;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.lossfunctions.impl.*;

import java.util.Arrays;

/**
 * Created by Alex on 08/08/2016.
 */
@Slf4j
public class LossFunctionGradientChecks extends BaseNd4jTest {

    public static final double epsilon = 1e-6;
    private static final double maxRelError = 5.0; //5% relative error
    DataBuffer.Type initialType;

    public LossFunctionGradientChecks(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Before
    public void before() throws Exception {
        super.before();

        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        Nd4j.getRandom().setSeed(123);
    }

    @Test
    public void testLossFunctionGradients() {

        INDArray[] labels = new INDArray[] {Nd4j.create(new double[] {0, 1, 0}), Nd4j.create(new double[] {0, 1, 1}),
                        /*Nd4j.create(new double[][]{{1,0,0},{0,1,0},{0,0,1}}),
                        Nd4j.create(new double[]{1,2,1}),
                        Nd4j.create(new double[][]{{1,2,1},{0.1,1,0.5},{20,3,1}}),
                        Nd4j.create(new double[]{1,0,0}),
                        Nd4j.create(new double[][]{{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}),
                        Nd4j.create(new double[]{1,2,1}),
                        Nd4j.create(new double[][]{{101,21,110},{10.1,1,0.5},{200,30,0.001}}),
                        */
                        Nd4j.create(new double[] {1, 2, 1}),
                        Nd4j.create(new double[][] {{101, 21, 110}, {10.1, 1, 0.5}, {200, 30, 0.001}}),
                        Nd4j.create(new double[] {1, 2, 1}),
                        Nd4j.create(new double[][] {{101, 21, 110}, {10.1, 1, 0.5}, {200, 30, 0.001}}),
                        Nd4j.create(new double[] {1, 2, 1}),
                        Nd4j.create(new double[][] {{101, 21, 110}, {10.1, 1, 0.5}, {200, 30, 0.001}}),
                        Nd4j.create(new double[] {1, 2, 1}),
                        Nd4j.create(new double[][] {{101, 21, 110}, {10.1, 1, 0.5}, {200, 30, 0.001}}),
                        //Nd4j.create(new double[][] {{-1,-1,1},{-1,1,1},{-1,1,1}}),
                        Nd4j.create(new double[][] {{-1, 1, -1}, {1, 1, -1}, {-1, 1, 1}}),
                        Nd4j.create(new double[][] {{-1, 1, -1}, {1, 1, -1}, {-1, 1, 1}}),
                        Nd4j.create(new double[][] {{1, 0, 0}, {0, 1, 0}, {1, 0, 1}}),
                        //Nd4j.create(new double[][] {{10,1,3},{1,10,1},{1,2,5}}),
                        //Nd4j.create(new double[][] {{10,-1,3},{1,10,1},{1,2,-5}}),
        };

        INDArray[] preOut = new INDArray[] {Nd4j.rand(1, 3), Nd4j.rand(1, 3),
                        /*
                        Nd4j.rand(3,3),
                        Nd4j.rand(1,3).add(5),
                        Nd4j.rand(3,3),
                        Nd4j.rand(1,3).add(5),
                        Nd4j.rand(4,4),*/
                        Nd4j.randn(1, 3), Nd4j.randn(3, 3).add(10), Nd4j.rand(1, 3), Nd4j.randn(3, 3).add(10),
                        Nd4j.randn(1, 3), Nd4j.randn(3, 3).add(10), Nd4j.rand(1, 3), Nd4j.randn(3, 3).add(10),
                        /*
                        Nd4j.rand(1,3),
                        Nd4j.randn(3,3).add(10),
                        */
                        Nd4j.rand(3, 3).addi(-0.5), //adding a neg num makes some +ve/ some -ve
                        Nd4j.rand(3, 3).addi(-0.5), //adding a neg num makes some +ve/ some -ve
                        Nd4j.rand(3, 3).addi(-0.5),
                        // Nd4j.rand(3,3),
                        //Nd4j.randn(3,3)
        };

        ILossFunction[] lossFn = new ILossFunction[] {new LossBinaryXENT(), new LossBinaryXENT(),
                        /*new LossMCXENT(), new LossMCXENT(),
                        new LossMCXENT(),new LossMSE(), new LossMSE(), new LossKLD(), new LossKLD(), new LossMAE(), new LossMAE(),*/
                        new LossMAE(), new LossMAE(), new LossMSE(), new LossMSE(), new LossL1(), new LossL1(),
                        new LossL2(), new LossL2(), new LossSquaredHinge(), new LossHinge(), new LossMultiLabel()
                        //new LossPoisson(),
                        //new LossCosineProximity()
        };

        String[] activationFns = new String[] {"identity", "tanh",
                        /*"softmax","tanh","identity","tanh",
                        "tanh","identity","identity","identity","identity",*/
                        "identity", "identity", "identity", "identity", "sigmoid", "relu", "sigmoid", "relu",
                        "identity", "identity", "identity"
                        //"relu",
                        //"identity"
        };


        for (int i = 0; i < labels.length; i++) {
            //if (i != labels.length-2)  continue;
            int totalNFailures = 0;

            ILossFunction lf = lossFn[i];
            INDArray l = labels[i];
            INDArray p = preOut[i];
            String afn = activationFns[i];

            System.out.printf("Starting test: %s, %s, input shape = %s\n", lf, afn, Arrays.toString(p.shape()));

            INDArray grad = lf.computeGradient(l, p, activationInstance(afn), null);

            NdIndexIterator iter = new NdIndexIterator(l.shape());
            while (iter.hasNext()) {
                val next = iter.next();

                double before = p.getDouble(next);
                p.putScalar(next, before + epsilon);
                double scorePlus = lf.computeScore(l, p, activationInstance(afn), null, true);
                p.putScalar(next, before - epsilon);
                double scoreMinus = lf.computeScore(l, p, activationInstance(afn), null, true);
                p.putScalar(next, before);

                double scoreDelta = scorePlus - scoreMinus;

                double numericalGradient = scoreDelta / (2 * epsilon);
                double analyticGradient = grad.getDouble(next) / l.size(0); //Analytic gradient method is before dividing by minibatch

                double relError = Math.abs(analyticGradient - numericalGradient) * 100 / (Math.abs(numericalGradient));
                if (analyticGradient == 0.0 && numericalGradient == 0.0)
                    relError = 0.0; //Edge case: i.e., RNNs with time series length of 1.0


                if (relError > maxRelError || Double.isNaN(relError)) {
                    System.out.println("Param " + i + " FAILED: grad= " + analyticGradient + ", numericalGrad= "
                                    + numericalGradient + ", relErrorPerc= " + relError + ", scorePlus=" + scorePlus
                                    + ", scoreMinus= " + scoreMinus);
                    totalNFailures++;
                } else {
                    System.out.println("Param " + i + " passed: grad= " + analyticGradient + ", numericalGrad= "
                                    + numericalGradient + ", relError= " + relError + ", scorePlus=" + scorePlus
                                    + ", scoreMinus= " + scoreMinus);
                }

                //System.out.println("Param " + i + " passed: grad= " + analyticGradient + ", numericalGrad= " + numericalGradient
                //        + ", relError= " + relError + ", scorePlus="+scorePlus+", scoreMinus= " + scoreMinus );
            }

            if (totalNFailures > 0)
                System.out.println("Gradient check failed for loss function " + lf + "; total num failures = "
                                + totalNFailures);
            System.out.println("DONE");
        }
    }

    /*
    public static List<INDArray> makeLabels(String activation,int[]labelSize) {
        //edge cases are label size of one for everything except softmax which is two
        //+ve and -ve values, zero and non zero values, less than zero/greater than zero
        List<INDArray> returnVals = new ArrayList<>(labelSize.length);
        for (int i=0; i< labelSize.length; i++) {
            int aLabelSize = labelSize[i];
            Random r = new Random();
            double[] someVals = new double[aLabelSize];
            double someValsSum = 0;
            for (int j=0; j<aLabelSize; j++) {
                double someVal = r.nextGaussian();
                double transformVal = 0;
                switch (activation) {
                    case "identity":
                        transformVal = someVal;
                    case "softmax":
                        transformVal = someVal;
                        break;
                    case "sigmoid":
                        transformVal = Math.sin(someVal);
                        break;
                    case "tanh":
                        transformVal = Math.tan(someVal);
                        break;
                    case "reul":
                        transformVal = someVal * someVal + 4;
                        break;
                }
                someVals[j] = transformVal;
                someValsSum += someVals[j];
            }
            if (activation == "sigmoid") {
                for (int j=0; j<aLabelSize; j++) {
                    someVals[j] /= someValsSum;
                }
            }
            returnVals.add(Nd4j.create(someVals));
        }
        return returnVals;
    }
    */

    public static IActivation activationInstance(String activation) {

        IActivation activationFn = new ActivationSigmoid();
        switch (activation) {
            case "identity":
                activationFn = new ActivationIdentity();
            case "softmax":
                activationFn = new ActivationSoftmax();
                break;
            case "sigmoid":
                activationFn = new ActivationSigmoid();
                break;
            case "tanh":
                activationFn = new ActivationTanH();
                break;
            case "reul":
                activationFn = new ActivationReLU();
                break;
        }

        return activationFn;

    }

    @After
    public void after() {
        DataTypeUtil.setDTypeForContext(this.initialType);
        System.out.println("AFTER DATATYPE HERE: " + Nd4j.dataType());
    }

    @Override
    public char ordering() {
        return 'f';
    }

}
