package org.deeplearning4j.plot;

import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.jblas.DoubleMatrix;

/**
 * Iterates through a dataset and draws reconstructions
 */
public class DeepAutoEncoderDataSetReconstructionRender {
    private DataSetIterator iter;
    private DeepAutoEncoder encoder;

    public DeepAutoEncoderDataSetReconstructionRender(DataSetIterator iter, DeepAutoEncoder encoder) {
        this.iter = iter;
        this.encoder = encoder;
    }

    public void draw() throws InterruptedException {
        while(iter.hasNext()) {
            DataSet first = iter.next();
            DoubleMatrix reconstruct = encoder.output(first.getFirst());
            for(int j = 0; j < first.numExamples(); j++) {

                DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
                DoubleMatrix reconstructed2 = reconstruct.getRow(j);
                DoubleMatrix draw2 = reconstructed2.mul(255);

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
