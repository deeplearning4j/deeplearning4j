package org.deeplearning4j.example.curves;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.da.DenoisingAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.CurvesDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DenoisingAutoEncoderMnistExample {


    private static Logger log = LoggerFactory.getLogger(DenoisingAutoEncoderMnistExample.class);

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        DenoisingAutoEncoder autoEncoder = new DenoisingAutoEncoder.Builder()
                .withOptmizationAlgo(NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .numberOfVisible(784).numHidden(600).build();


        //batches of 10, 60000 examples total
        DataSetIterator iter = new MultipleEpochsIterator(10,new CurvesDataSetIterator(10,10));
        while(iter.hasNext()) {
            DataSet next = iter.next();
            for(int i = 0; i < 100; i++) {
                autoEncoder.train(next.getFirst(), 1e-1, 0.3, i);

                log.info("Error on iteration " + i + " is " + autoEncoder.getReConstructionCrossEntropy());

            }

        }


        iter.reset();


        FilterRenderer render = new FilterRenderer();
        render.renderFilters(autoEncoder.getW(), "example-render.jpg", 28, 28,10);




        //Iterate over the data applyTransformToDestination after done training and show the 2 side by side (you have to drag the test image over to the right)
        while(iter.hasNext()) {
            DataSet first = iter.next();
            DoubleMatrix reconstruct = autoEncoder.reconstruct(first.getFirst());
            for(int j = 0; j < first.numExamples(); j++) {

                DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
                DoubleMatrix reconstructed2 = reconstruct.getRow(j);
                DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

                DrawReconstruction d = new DrawReconstruction(draw1);
                d.title = "REAL";
                d.draw();
                DrawReconstruction d2 = new DrawReconstruction(draw2,1000,1000);
                d2.title = "TEST";
                d2.draw();
                Thread.sleep(10000);
                d.frame.dispose();
                d2.frame.dispose();
            }


        }

    }

}
