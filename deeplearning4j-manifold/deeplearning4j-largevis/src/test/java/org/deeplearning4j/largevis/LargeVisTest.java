package org.deeplearning4j.largevis;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nadam;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

public class LargeVisTest {

    public final static  double MAX_REL_ERROR = 1e-3;


    @Test
    public void testLargeVisRun() throws Exception {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        DataSet iris = new IrisDataSetIterator(150,150).next();
        LargeVis largeVis = LargeVis.builder().iterationCount(300)
                .vec(iris.getFeatureMatrix()).updater(new Nadam(1.0))
                .normalize(true)
                .seed(42).build();
        largeVis.fit();
        assertNotNull(largeVis.getResult());
        List<String> list = new ArrayList<>();
        INDArray labels = Nd4j.argMax(iris.getLabels(),1);
        for(int i = 0; i < iris.numExamples(); i++) {
            list.add(String.valueOf(labels.getInt(i)));
        }
        largeVis.saveAsFile(list,"/tmp/iris-vis.txt");
    }


    @Test
    public void testLargeVisGrad() {
        Nd4j.getRandom().setSeed(12345);
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        DataSet iris = new IrisDataSetIterator(150,150).next();
        LargeVis largeVis = LargeVis.builder().gamma(1.0)
                .vec(iris.getFeatureMatrix())
                .normalize(true).gradientNormalization(GradientNormalization.None)
                .seed(42).build();

        largeVis.initWeights();

        int x = 0;
        int y = 1;
        int i = 0;
        double currLr = 1;
        Nd4j.getRandom().setSeed(12345);

        INDArray grads = largeVis.gradientsFor(x,y,0,currLr,true);
        INDArray visX = largeVis.getVis().slice(x);
        INDArray visY = largeVis.getVis().slice(y);
        double epsilon = 1e-6;

        for (int v = 0; v < visX.length(); v++) {
            double backpropGradient = grads.getRow(1).getDouble(v);
            Nd4j.getRandom().setSeed(12345);
            double origParamValue = visY.getDouble(v);
            visY.putScalar(v, origParamValue + epsilon);
            double scorePlus = largeVis.errorWrt(x, y, 2, currLr, false,false).getDouble(v);
            visY.putScalar(v, origParamValue - epsilon);
            Nd4j.getRandom().setSeed(12345);
            double scoreMinus = largeVis.errorWrt(x, y, 2, currLr, false,false).getDouble(v);
            visY.putScalar(v, origParamValue); //reset param so it doesn't affect later calcs


            double numericalGradient = (scorePlus - scoreMinus) / (2 * epsilon);

            double relError;
            if (backpropGradient == 0.0 && numericalGradient == 0.0)
                relError = 0.0;
            else {
                relError = Math.abs(backpropGradient - numericalGradient)
                        / (Math.abs(backpropGradient) + Math.abs(numericalGradient));
            }

            String msg = "grad: i=" + i + ", -" +   relError +", scorePlus=" + scorePlus + ", scoreMinus=" + scoreMinus
                    + ", numGrad=" + numericalGradient + ", backpropGrad = " + backpropGradient;

            if (relError > MAX_REL_ERROR)
                fail(msg);
            else
                System.out.println(msg);
        }

    }

}
