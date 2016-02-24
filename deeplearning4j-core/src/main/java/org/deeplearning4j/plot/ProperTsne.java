package org.deeplearning4j.plot;

import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.AdaGrad;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author raver119@gmail.com
 */
public class ProperTsne {
    protected int maxIter = 1000;
    protected double realMin = Nd4j.EPS_THRESHOLD;
    protected double initialMomentum = 0.5;
    protected double finalMomentum = 0.8;
    protected  double minGain = 1e-2;
    protected double momentum = initialMomentum;
    protected int switchMomentumIteration = 100;
    protected boolean normalize = true;
    protected boolean usePca = false;
    protected int stopLyingIteration = 250;
    protected double tolerance = 1e-5;
    protected double learningRate = 500;
    protected AdaGrad adaGrad;
    protected boolean useAdaGrad = true;
    protected double perplexity = 30;
    protected INDArray gains,yIncs;
    protected INDArray y;
    protected transient IterationListener iterationListener;
    protected static final Logger log = LoggerFactory.getLogger(Tsne.class);

    protected ProperTsne() {
        ;
    }

    protected void init() {

    }

    public void calculate(INDArray X, int targetDimensions, double perplexity) {

    }

    public static class Builder {
        protected int maxIter = 1000;
        protected double realMin = 1e-12f;
        protected double initialMomentum = 5e-1f;
        protected double finalMomentum = 8e-1f;
        protected double momentum = 5e-1f;
        protected int switchMomentumIteration = 100;
        protected boolean normalize = true;
        protected boolean usePca = false;
        protected int stopLyingIteration = 100;
        protected double tolerance = 1e-5f;
        protected double learningRate = 1e-1f;
        protected boolean useAdaGrad = false;
        protected double perplexity = 30;
        protected double minGain = 1e-1f;


        public Builder minGain(double minGain) {
            this.minGain = minGain;
            return this;
        }

        public Builder perplexity(double perplexity) {
            this.perplexity = perplexity;
            return this;
        }

        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }


        public Builder tolerance(double tolerance) {
            this.tolerance = tolerance;
            return this;
        }

        public Builder stopLyingIteration(int stopLyingIteration) {
            this.stopLyingIteration = stopLyingIteration;
            return this;
        }

        public Builder usePca(boolean usePca) {
            this.usePca = usePca;
            return this;
        }

        public Builder normalize(boolean normalize) {
            this.normalize = normalize;
            return this;
        }

        public Builder setMaxIter(int maxIter) {
            this.maxIter = maxIter;
            return this;
        }

        public Builder setRealMin(double realMin) {
            this.realMin = realMin;
            return this;
        }

        public Builder setInitialMomentum(double initialMomentum) {
            this.initialMomentum = initialMomentum;
            return this;
        }

        public Builder setFinalMomentum(double finalMomentum) {
            this.finalMomentum = finalMomentum;
            return this;
        }

        public Builder setMomentum(double momentum) {
            this.momentum = momentum;
            return this;
        }

        public Builder setSwitchMomentumIteration(int switchMomentumIteration) {
            this.switchMomentumIteration = switchMomentumIteration;
            return this;
        }

        public ProperTsne build() {
            ProperTsne tsne = new ProperTsne();
            tsne.finalMomentum = this.finalMomentum;
            tsne.initialMomentum = this.initialMomentum;
            tsne.maxIter = this.maxIter;
            tsne.learningRate = this.learningRate;
            tsne.minGain = this.minGain;
            tsne.momentum = this.momentum;
            tsne.normalize = this.normalize;
            tsne.perplexity = this.perplexity;
            tsne.tolerance = this.tolerance;
            tsne.realMin = this.realMin;
            tsne.stopLyingIteration = this.stopLyingIteration;
            tsne.switchMomentumIteration = this.switchMomentumIteration;
            tsne.usePca = this.usePca;
            tsne.useAdaGrad = this.useAdaGrad;

            tsne.init();

            return tsne;
        }
    }
}
