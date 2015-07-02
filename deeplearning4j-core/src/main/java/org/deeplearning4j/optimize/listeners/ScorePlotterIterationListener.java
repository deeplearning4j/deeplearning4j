package org.deeplearning4j.optimize.listeners;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Reference: https://cs231n.github.io/neural-networks-3/
 */
public class ScorePlotterIterationListener implements IterationListener {
    private MultiLayerNetwork network;
    private INDArray input;
    private INDArray labels;
    private int iterations = 1;
    private NeuralNetPlotter plotter = new NeuralNetPlotter();
    private boolean renderFirst = false;
    private ArrayList<Double> scores = new ArrayList<>();
    private ArrayList<Double> accuracy = new ArrayList<>();
    private boolean renderAccuracy = false;

    /**
     *
     * @param iterations the number of iterations to render every plot
     * @param network the model which must be multiple layers
     * @param input the training data input
     * @param labels the training data labels
     */
    public ScorePlotterIterationListener(int iterations, MultiLayerNetwork network, INDArray input, INDArray labels) {
        this.iterations = iterations;
        this.network = network;
        this.input = input;
        this.labels = labels;
        this.renderAccuracy = true;

    }

    /**
     *
     * @param iterations the number of iterations to render every plot
     * @param network the model which must be multiple layers
     * @param data the training data input
     */
    public ScorePlotterIterationListener(int iterations, MultiLayerNetwork network, DataSet data) {
        this.iterations = iterations;
        this.network = network;
        this.input = data.getFeatures();
        this.labels = data.getLabels();
        this.renderAccuracy = true;

    }


    /**
     *
     * @param iterations the number of iterations to render every
     */
    public ScorePlotterIterationListener(int iterations) {
        this.iterations = iterations;
    }

    protected String storeData(ArrayList data)  {
        try {
            String filePath = plotter.getDataFilePath();
            String tmpFilePath = UUID.randomUUID().toString();
            File write = new File(filePath,tmpFilePath);
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
            write.deleteOnExit();
            StringBuilder sb = new StringBuilder();
            for(Object value : data) {
                sb.append(String.format("%.10f", (Double) value));
                sb.append(",");
            }
            String line = sb.toString();
            line = line.substring(0, line.length()-1);
            bos.write(line.getBytes());
            bos.flush();
            bos.close();
            return filePath+tmpFilePath;

        } catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    protected double calculateAccuracy() {
        Evaluation eval = new Evaluation();
        INDArray output = network.output(input);
        eval.eval(labels, output);
        return eval.accuracy();
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        scores.add(-model.score());
        if (renderAccuracy) {
            double iterationAccuracy = this.calculateAccuracy();
            accuracy.add(iterationAccuracy);
        }

        if (iteration == 0 && renderFirst || iteration > 0 && iteration % this.iterations == 0) {
            plotter.updateGraphDirectory((Layer) model);
            String dataFilePath = storeData(scores);
            plotter.renderGraph("loss", dataFilePath, plotter.getLayerGraphFilePath() + "loss.png");
            if (renderAccuracy) {
                dataFilePath = storeData(accuracy);
                plotter.renderGraph("accuracy", dataFilePath, plotter.getLayerGraphFilePath() + "accuracy.png");
            }
        }
    }

}
