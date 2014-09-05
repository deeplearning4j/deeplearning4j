package org.deeplearning4j.plot;

import org.deeplearning4j.models.featuredetectors.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.transformation.MatrixTransform;


/**
 * Iterates through a dataset and draws reconstructions
 */
public class DeepAutoEncoderDataSetReconstructionRender {
    private DataSetIterator iter;
    private DeepAutoEncoder encoder;
    private int rows,columns;
    private MatrixTransform picDraw,reconDraw;

    /**
     * Initialize with the given rows and columns, this will reshape the
     * matrix in to the specified rows and columns
     * @param iter
     * @param encoder
     * @param rows rows
     * @param columns columns
     */
    public DeepAutoEncoderDataSetReconstructionRender(DataSetIterator iter, DeepAutoEncoder encoder,int rows, int columns) {
        this.iter = iter;
        this.encoder = encoder;
        this.rows = rows;
        this.columns = columns;
    }

    public void draw() throws InterruptedException {
        while(iter.hasNext()) {
            DataSet first = iter.next();
            INDArray reconstruct = encoder.output(first.getFeatureMatrix());
            for(int j = 0; j < first.numExamples(); j++) {

                INDArray draw1 = first.get(j).getFeatureMatrix().mul(255);
                if(picDraw != null)
                    draw1 = picDraw.apply(draw1);
                INDArray reconstructed2 = reconstruct.getRow(j);
                if(reconDraw != null)
                    reconstructed2 = reconDraw.apply(reconstructed2);
                INDArray draw2 = reconstructed2.mul(255);

                DrawReconstruction d = new DrawReconstruction(draw1.reshape(rows,columns));
                d.title = "REAL";
                d.draw();
                DrawReconstruction d2 = new DrawReconstruction(draw2.reshape(rows,columns),1000,1000);
                d2.title = "TEST";
                d2.draw();
                Thread.sleep(10000);
                d.frame.dispose();
                d2.frame.dispose();
            }


        }
    }

    public MatrixTransform getPicDraw() {
        return picDraw;
    }

    public void setPicDraw(MatrixTransform picDraw) {
        this.picDraw = picDraw;
    }

    public MatrixTransform getReconDraw() {
        return reconDraw;
    }

    public void setReconDraw(MatrixTransform reconDraw) {
        this.reconDraw = reconDraw;
    }
}
