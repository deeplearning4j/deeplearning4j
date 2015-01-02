package org.deeplearning4j.plot;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;


/**
 * Created by agibsonccc on 1/1/15.
 */
public class BarnesHutTsne extends Tsne implements Model {
    private INDArray x;
    private int n;
    private int d;
    private INDArray y;
    private int numDimensions;
    private double perplexity;
    private double theta;
    private int maxIter = 1000;
    private int stopLyingIteration = 250;
    private int momentumSwitchIteration = 250;
    private double momentum = 0.5;
    private double finalMomentum = 0.8;
    private double learningRate = 200;

    public BarnesHutTsne(INDArray x, int n, int d, INDArray y, int numDimensions, double perplexity, double theta, int maxIter, int stopLyingIteration, int momentumSwitchIteration, double momentum, double finalMomentum, double learningRate) {
        super();
        this.x = x;
        this.n = n;
        this.d = d;
        this.y = y;
        this.numDimensions = numDimensions;
        this.perplexity = perplexity;
        this.theta = theta;
        this.maxIter = maxIter;
        this.stopLyingIteration = stopLyingIteration;
        this.momentumSwitchIteration = momentumSwitchIteration;
        this.momentum = momentum;
        this.finalMomentum = finalMomentum;
        this.learningRate = learningRate;
    }

    @Override
    public void fit() {
        INDArray dY = Nd4j.create(n,d);
        INDArray uY = Nd4j.create(n,d);
        INDArray gains = Nd4j.ones(n,d);
        INDArray p;
        x = Transforms.normalizeZeroMeanAndUnitVariance(x);

        boolean exact = theta == 0.0;
        if(exact) {
            p = Nd4j.create(n,n);

        }
        else {

        }
    }



    @Override
    public void update(Gradient gradient) {

    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public INDArray transform(INDArray data) {
        return null;
    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public int numParams() {
        return 0;
    }

    @Override
    public void setParams(INDArray params) {

    }

    @Override
    public void fit(INDArray data) {

    }

    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient getGradient() {
        return null;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return null;
    }

    @Override
    public int batchSize() {
        return 0;
    }

    @Override
    public NeuralNetConfiguration conf() {
        return null;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {

    }
}
