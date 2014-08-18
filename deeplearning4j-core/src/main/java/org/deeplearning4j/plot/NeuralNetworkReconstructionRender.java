package org.deeplearning4j.plot;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.NeuralNetwork;

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
            INDArray reconstruct =  network.reconstruct(first.getFeatureMatrix());
            for(int j = 0; j < first.numExamples(); j++) {

                INDArray draw1 = first.get(j).getFeatureMatrix().mul(255);
                INDArray reconstructed2 = reconstruct.getRow(j);
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
