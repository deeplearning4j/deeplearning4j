package org.deeplearning4j.plot;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.jblas.DoubleMatrix;

/**
 * Reconstruction renders for a multi layer network
 */
public class MultiLayerNetworkReconstructionRender {

    private DataSetIterator iter;
    private BaseMultiLayerNetwork network;
    private int reconLayer = -1;

    public MultiLayerNetworkReconstructionRender(DataSetIterator iter, BaseMultiLayerNetwork network,int reconLayer) {
        this.iter = iter;
        this.network = network;
        this.reconLayer = reconLayer;
    }
    public MultiLayerNetworkReconstructionRender(DataSetIterator iter, BaseMultiLayerNetwork network) {
        this(iter,network,-1);
    }
    public void draw() throws InterruptedException {
        while(iter.hasNext()) {
            DataSet first = iter.next();
            DoubleMatrix reconstruct = reconLayer < 0 ? network.reconstruct(first.getFirst()) : network.reconstruct(first.getFirst(),reconLayer);
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
