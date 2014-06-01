package org.deeplearning4j.example.deepautoencoder;

import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.plot.DeepAutoEncoderDataSetReconstructionRender;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collections;
import java.util.List;

/**
 * Read a DBN from a file and use that as the basis for the deep autoencoder
 */
public class DeepAutoEncoderFromFile {
    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderFromFile.class);

    public static void main(String[] args) throws Exception {
        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(10,300);


        DBN dbn = SerializationUtils.readObject(new File(args[0]));
        dbn.setRenderWeightsEveryNEpochs(10);
        for(int i = 0; i < dbn.getnLayers(); i++)
            dbn.getLayers()[i].setRenderEpochs(10);
        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);
        encoder.setRoundCodeLayerInput(true);
        encoder.setOutputLayerLossFunction(OutputLayer.LossFunction.RMSE_XENT);
        encoder.setOutputLayerActivation(Activations.sigmoid());
        while(iter.hasNext()) {
            DataSet next = iter.next();
            if(next == null)
                break;
            log.info("Training on " + next.numExamples());
            log.info("Coding layer is " + encoder.encode(next.getFirst()));
            encoder.finetune(next.getFirst(),1e-2,1000);
            FilterRenderer f = new FilterRenderer();
            f.renderFilters(encoder.getDecoder().getOutputLayer().getW(),"outputlayer.png",28,28,next.numExamples());
            DeepAutoEncoderDataSetReconstructionRender render = new DeepAutoEncoderDataSetReconstructionRender(next.iterator(next.numExamples()),encoder);
            render.draw();
        }


        SerializationUtils.saveObject(encoder,new File("deepautoencoder.ser"));

        iter.reset();

        DeepAutoEncoderDataSetReconstructionRender render = new DeepAutoEncoderDataSetReconstructionRender(iter,encoder);
        render.draw();

    }
}
