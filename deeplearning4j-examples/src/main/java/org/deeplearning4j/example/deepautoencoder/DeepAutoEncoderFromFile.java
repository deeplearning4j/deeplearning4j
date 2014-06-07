package org.deeplearning4j.example.deepautoencoder;

import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.plot.DeepAutoEncoderDataSetReconstructionRender;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Read a DBN from a file and use that as the basis for the deep autoencoder
 */
public class DeepAutoEncoderFromFile {
    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderFromFile.class);

    public static void main(String[] args) throws Exception {
        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(10,60000,false);


        DBN dbn = SerializationUtils.readObject(new File(args[0]));
        dbn.setOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT);


        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);
        encoder.setRoundCodeLayerInput(true);
        encoder.setNormalizeCodeLayerOutput(false);
        encoder.setOutputLayerLossFunction(OutputLayer.LossFunction.RMSE_XENT);

        encoder.setOutputLayerActivation(Activations.sigmoid());
        encoder.setVisibleUnit(RBM.VisibleUnit.GAUSSIAN);
        encoder.setHiddenUnit(RBM.HiddenUnit.BINARY);

        int testSets = 0;
        int count = 0;
        while(iter.hasNext()) {
            DataSet next = iter.next();
            next.scale();

            List<Integer> labels = new ArrayList<>();
            for(int i = 0; i < next.numExamples(); i++)
                labels.add(next.get(i).outcome());

            if(next == null)
                break;
            log.info("Labels " + labels);
            log.info("Training on " + next.numExamples());
            log.info(("Coding layer is " + encoder.encodeWithScaling(next.getFirst())).replaceAll(";","\n"));
            encoder.finetune(next.getFirst(),1e-1,1000);
            if(true) {
                NeuralNetPlotter plotter = new NeuralNetPlotter();
                encoder.getDecoder().feedForward(encoder.encodeWithScaling(next.getFirst()));
                String[] layers = new String[encoder.getDecoder().getnLayers()];
                DoubleMatrix[] weights = new DoubleMatrix[layers.length];
                for(int i = 0; i < encoder.getDecoder().getnLayers(); i++) {
                    layers[i] = "" + i;
                    weights[i] = encoder.getDecoder().getLayers()[i].getW();
                }

                plotter.histogram(layers, weights);

                FilterRenderer f = new FilterRenderer();
                f.renderFilters(encoder.getDecoder().getOutputLayer().getW(), "outputlayer.png", 28, 28, next.numExamples());
                DeepAutoEncoderDataSetReconstructionRender render = new DeepAutoEncoderDataSetReconstructionRender(next.iterator(next.numExamples()),encoder);
                render.draw();
            }

            count++;

        }


        SerializationUtils.saveObject(encoder,new File("deepautoencoder.ser"));

        iter.reset();

        DeepAutoEncoderDataSetReconstructionRender render = new DeepAutoEncoderDataSetReconstructionRender(iter,encoder);
        render.draw();

    }
}
