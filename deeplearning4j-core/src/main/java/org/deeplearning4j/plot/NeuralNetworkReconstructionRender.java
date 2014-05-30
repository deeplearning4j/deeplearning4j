package org.deeplearning4j.plot;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.nn.NeuralNetwork;
import org.jblas.DoubleMatrix;

/**
 *
 * Neural Network reconstruction renderer
 * @author Adam Gibson
 */
public class NeuralNetworkReconstructionRender {

    private DataSetIterator iter;
    private NeuralNetwork network;

    public NeuralNetworkReconstructionRender(DataSetIterator iter, NeuralNetwork network) {
        this.iter = iter;
        this.network = network;
    }

    public void draw() throws InterruptedException {
        while(iter.hasNext()) {
            DataSet first = iter.next();
            DoubleMatrix reconstruct =  network.reconstruct(first.getFirst());
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
