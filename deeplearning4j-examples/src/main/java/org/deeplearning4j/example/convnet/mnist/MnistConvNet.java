package org.deeplearning4j.example.convnet.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawMnistGreyScale;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.rbm.ConvolutionalRBM;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 */
public class MnistConvNet {

    private static Logger log = LoggerFactory.getLogger(MnistConvNet.class);

    public static void main(String[] args) throws Exception {

        ConvolutionalRBM r = new ConvolutionalRBM.Builder()
                .withNumFilters(10)
                .withVisibleSize(new int[]{28,28})
                .withFilterSize(new int[]{10,10})
                .numberOfVisible(28).numHidden(28)
                .build();


        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(1,50);

        while(iter.hasNext()) {
            DataSet next = iter.next();
            DoubleMatrix reshape = next.getFirst().reshape(28,28);
            r.trainTillConvergence(reshape, 1e-1, new Object[]{1, 1e-1, 5000});

        }






        iter.reset();

        FilterRenderer render = new FilterRenderer();
        render.renderFilters(r.getW(), "example-render.jpg", 28, 28);




        //Iterate over the data set after done training and show the 2 side by side (you have to drag the test image over to the right)
        while(iter.hasNext()) {
            DataSet first = iter.next();
            DoubleMatrix reconstruct = r.reconstruct(first.getFirst());
            for(int j = 0; j < first.numExamples(); j++) {

                DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
                DoubleMatrix reconstructed2 = reconstruct.getRow(j);
                DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2, 1, new MersenneTwister(123)).mul(255);

                DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
                d.title = "REAL";
                d.draw();
                DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2,1000,1000);
                d2.title = "TEST";
                d2.draw();
                Thread.sleep(10000);
                d.frame.dispose();
                d2.frame.dispose();
            }


        }
    }

}
