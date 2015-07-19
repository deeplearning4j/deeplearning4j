package org.deeplearning4j.plot.iterationlistener;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

import java.util.ArrayList;

/**
 *
 * Reference: https://cs231n.github.io/neural-networks-3/
 */

public class AccuracyPlotterIterationListener implements IterationListener {
    private int epochs = 1; // number times the model saw all the data
    private INDArray input;
    private MultiLayerNetwork network;
    private INDArray labels;
    private NeuralNetPlotter plotter = new NeuralNetPlotter();
    private boolean renderFirst = false;
    private ArrayList<Double> accuracy = new ArrayList<>();
    private boolean invoked = false;

    @Override
    public boolean invoked(){ return invoked; }

    @Override
    public void invoke() { this.invoked = true; }


    /**
     *
     * @param epochs the number of iterations to render
     * @param renderFirst render the graph on first pass
     **/
    public AccuracyPlotterIterationListener(int epochs, boolean renderFirst) {
        this.epochs = epochs;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param epochs the number of iterations to render
     * @param plotter the plotter to use
     */
    public AccuracyPlotterIterationListener(int epochs, NeuralNetPlotter plotter) {
        this.epochs = epochs;
        this.plotter = plotter;
    }

    /**
     *
     * @param epochs the number of iterations to render
     * @param plotter the plotter to use
     * @param renderFirst render the graph on first pass
     */
    public AccuracyPlotterIterationListener(int epochs, NeuralNetPlotter plotter, boolean renderFirst) {
        this.epochs = epochs;
        this.plotter = plotter;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param epochs the number of iterations to render every plot
     * @param network the model which must be multiple layers
     * @param data the training data input
     */
    public AccuracyPlotterIterationListener(int epochs, MultiLayerNetwork network, DataSet data) {
        this.epochs = epochs;
        this.network = network;
        this.input = data.getFeatures();
        this.labels = data.getLabels();

    }


    /**
     *
     * @param epochs the number of iterations to render every plot
     * @param network the model which must be multiple layers
     * @param data the training data input
     */
    public AccuracyPlotterIterationListener(int epochs, MultiLayerNetwork network, DataSet data, boolean renderFirst) {
        this.epochs = epochs;
        this.network = network;
        this.input = data.getFeatures();
        this.labels = data.getLabels();
        this.renderFirst = renderFirst;

    }

    /**
     *
     * @param epochs the number of iterations to render every plot
     * @param network the model which must be multiple layers
     * @param input the training data input
     * @param labels the training data labels
     */
    public AccuracyPlotterIterationListener(int epochs, MultiLayerNetwork network, INDArray input, INDArray labels) {
        this.epochs = epochs;
        this.network = network;
        this.input = input;
        this.labels = labels;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     */
    public AccuracyPlotterIterationListener(int iterations) {
        this.epochs = epochs;
    }


    private double calculateAccuracy() {
        Evaluation eval = new Evaluation();
        INDArray output = network.output(input);
        eval.eval(labels, output);
        return eval.accuracy();
    }

    @Override
    public void iterationDone(Model model, int epochs) {
        double iterationAccuracy = this.calculateAccuracy();
        accuracy.add(iterationAccuracy);
        if (epochs == 0 && renderFirst || epochs > 0 && epochs % this.epochs == 0) {
            this.invoke();
            String dataFilePath = plotter.writeArray(accuracy);
            plotter.renderGraph("accuracy", dataFilePath, plotter.getLayerGraphFilePath() + "accuracy.png");
        }
    }

}
