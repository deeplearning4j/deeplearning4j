package org.deeplearning4j.example.lfw;

import java.io.File;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.api.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.api.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LFWRBMExample {

    private static Logger log = LoggerFactory.getLogger(LFWRBMExample.class);
    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        DataSetIterator iter = new LFWDataSetIterator(10,150000,28,28);
        log.info("Loading LFW...");
        DataSet all = iter.next(300);
        iter = new SamplingDataSetIterator(all,10,100);
        int cols = iter.inputColumns();
        log.info("Learning from " + cols);



        RBM r = new RBM.Builder()
                .withVisible(RBM.VisibleUnit.GAUSSIAN)
                .withHidden(RBM.HiddenUnit.RECTIFIED)
                .numberOfVisible(iter.inputColumns())
                .withOptmizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .numHidden(600).renderWeights(1)
                .withLossFunction(LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .build();

        NeuralNetPlotter plotter = new NeuralNetPlotter();
        int numIter = 0;
        /*
        Note that this is a demo meant to run faster, the faces learned are too fast, and the learning rate should be 1e-6
        or lower for proper learning to take place. Thi sis purely for demo purposes
         */
        while(iter.hasNext()) {
            DataSet curr = iter.next();
            curr.normalizeZeroMeanZeroUnitVariance();
            r.fit(curr.getFeatureMatrix(), 1e-1f, new Object[]{1, 1e-1f, 10000});
            if(numIter % 10 == 0) {
                FilterRenderer render = new FilterRenderer();
                try {
                    render.renderFilters(r.getW(), "currimg.png", (int)Math.sqrt(r.getW().rows()) , (int) Math.sqrt( r.getW().rows()),curr.numExamples());
                } catch (Exception e) {
                    log.error("Unable to plot filter, continuing...",e);
                }
            }

            numIter++;

        }
        File f = new File("faces-rbm.bin");
        log.info("Saving to " + f.getAbsolutePath());
        SerializationUtils.saveObject(r, f);
        iter.reset();






        //Iterate over the data applyTransformToDestination after done training and show the 2 side by side (you have to drag the test image over to the right)
        while(iter.hasNext()) {
            DataSet first = iter.next();
            INDArray reconstruct = r.transform(first.getFeatureMatrix());
            for(int j = 0; j < first.numExamples(); j++) {

                INDArray draw1 = first.get(j).getFeatureMatrix().mul(255);
                INDArray reconstructed2 = reconstruct.getRow(j).div(255);
                //MatrixUtil.scaleByMax(reconstructed2);
                INDArray draw2 = reconstructed2.mul(255);

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
