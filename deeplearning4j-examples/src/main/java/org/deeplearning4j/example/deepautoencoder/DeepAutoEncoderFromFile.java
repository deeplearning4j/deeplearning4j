package org.deeplearning4j.example.deepautoencoder;

import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.plot.DeepAutoEncoderDataSetReconstructionRender;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.plot.NeuralNetPlotter;

import org.deeplearning4j.util.SerializationUtils;
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
        DataSetIterator iter = new MultipleEpochsIterator(10,new MnistDataSetIterator(1000,60000,false));


        DBN dbn = SerializationUtils.readObject(new File(args[0]));
        dbn.setOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm.GRADIENT_DESCENT);
        dbn.setMomentum(9e-1f);

        DeepAutoEncoder encoder = new DeepAutoEncoder.Builder().withEncoder(dbn).build();
        encoder.setSampleFromHiddenActivations(false);
        encoder.setOutputActivationFunction(Activations.linear());
        encoder.setOutputLayerLossFunction(OutputLayer.LossFunction.RMSE_XENT);
        encoder.setLineSearchBackProp(false);
        encoder.setRoundCodeLayerInput(false);

        int count = 0;
        while(iter.hasNext()) {
            DataSet next = iter.next();
            List<Integer> labels = new ArrayList<>();
            for(int i = 0; i < next.numExamples(); i++)
                labels.add(next.get(i).outcome());


            if(next == null)
                break;
            log.info("Labels " + labels);
            log.info("Training on " + next.numExamples());
            //log.info(("Coding layer is " + encoder.encodeWithScaling(next.getFirst())).replaceAll(";","\n"));

            FilterRenderer f2 = new FilterRenderer();
            f2.renderFilters(encoder.getOutputLayer().getW(), "outputlayer.png", 28, 28, next.numExamples());
            INDArray recon =  encoder.output(next.getFeatureMatrix());

            encoder.finetune(next.getFeatureMatrix(),1e-3f,10);

            if(count % 10 == 0) {
                NeuralNetPlotter plotter = new NeuralNetPlotter();
                String[] layers = new String[encoder.getLayers().length];
                INDArray[] weights = new INDArray[layers.length];
                for(int i = 0; i < encoder.getLayers().length; i++) {
                    layers[i] = "" + i;
                    weights[i] = encoder.getLayers()[i].getW();
                }

                plotter.scatter(layers, weights);

                FilterRenderer f = new FilterRenderer();
                f.renderFilters(encoder.getOutputLayer().getW(), "outputlayer.png", 28, 28, next.numExamples());
               // DeepAutoEncoderDataSetReconstructionRender render = new DeepAutoEncoderDataSetReconstructionRender(next.iterator(next.numExamples()),encoder,28,28);
               // render.draw();
            }

            count++;

        }


        SerializationUtils.saveObject(encoder,new File("deepautoencoder.ser"));

        iter.reset();

        DeepAutoEncoderDataSetReconstructionRender render = new DeepAutoEncoderDataSetReconstructionRender(iter,encoder,28,28);
        render.draw();

    }
}
