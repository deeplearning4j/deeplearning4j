package org.deeplearning4j.plot;

import com.google.common.base.Function;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.Condition;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.nd4j.linalg.ops.transforms.Transforms.sign;

/**
 * Tsne calculation
 * @author Adam Gibson
 */
public class TsneCalculation {

    private int maxIter = 1000;
    private float realMin = 1e-12f;
    private float initialMomentum = 0.5f;
    private float finalMomentum = 0.8f;
    private int eta = 500;
    private final float minGain = 1e-2f;
    private float momentum = initialMomentum;
    private int switchMomentumIteration = 100;
    private boolean normalize = true;
    private boolean usePca = false;
    private int stopLyingIteration = 100;
    private float tolerance = 1e-5f;

    private static Logger log = LoggerFactory.getLogger(TsneCalculation.class);

    public TsneCalculation(
            int maxIter,
            float realMin,
            float initialMomentum,
            float finalMomentum,
            int eta,
            float momentum,
            int switchMomentumIteration,
            boolean normalize,
            boolean usePca,
            int stopLyingIteration,
            float tolerance) {
        this.tolerance = tolerance;
        this.stopLyingIteration = stopLyingIteration;
        this.maxIter = maxIter;
        this.realMin = realMin;
        this.normalize = normalize;
        this.initialMomentum = initialMomentum;
        this.usePca = usePca;
        this.finalMomentum = finalMomentum;
        this.eta = eta;
        this.momentum = momentum;
        this.switchMomentumIteration = switchMomentumIteration;
    }

    /**
     * Computes a gaussian kernel
     * given a vector of squared euclidean distances
     *
     * @param d
     * @param beta
     * @return
     */
    public Pair<Float,INDArray> hBeta(INDArray d,INDArray beta) {
        if(d.length() != beta.length() && !beta.isScalar())
            throw new IllegalArgumentException("D != beta");

        INDArray P = Transforms.exp(d.neg().muli(beta));
        float sum = P.sum(Integer.MAX_VALUE).get(0);

        float H = (float) Math.log(sum) + Nd4j.getBlasWrapper().dot(d,P) / sum;
        P.divi(sum + 1e-6f);

        return new Pair<>(H,P);
    }


    /**
     * Reduce the dimension of x
     * to the specified number of dimensions
     * @param X the x to reduce
     * @param nDims the number of dimensions to reduce to
     * @return the reduced dimension
     */
    public INDArray pca(INDArray X,int nDims) {
        if(normalize) {
            INDArray mean = X.mean(0);
            X = X.subiRowVector(mean);

        }

        IComplexNDArray xWrapped = Nd4j.createComplex(X);

        IComplexNDArray xTimesX = xWrapped.transpose().mmul(xWrapped);
        IComplexNDArray M = Eigen.eigenvectors(xTimesX)[1];
        INDArray ret = xWrapped.mmul(M.get(NDArrayIndex.interval(0, M.rows()), NDArrayIndex.interval(0, nDims))).getReal();
        return ret;


    }


    /**
     * Convert data to probability
     * co-occurrences
     * @param d the data to convert
     * @param u the perplexity of the model
     * @param tolerance the tolerance for convergence
     * @return the probabilities of co-occurrence
     */
    public INDArray d2p(final INDArray d,final float u,final float tolerance) {
        int n = d.rows();
        final INDArray p = Nd4j.create(n, n);
        final INDArray beta = Nd4j.ones(n, 1);
        final float logU = (float) Math.log(u);
        log.info("Calculating probabilities of data similarities..");
        ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for(int i = 0; i < n; i++) {
            if(i % 500 == 0)
                log.info("Handled " + i + " records");
            final int j = i;
            service.submit(new Runnable() {
                @Override
                public void run() {
                    float betaMin = Float.NEGATIVE_INFINITY;
                    float betaMax = Float.POSITIVE_INFINITY;

                    Pair<Float,INDArray> pair =  hBeta(d.getRow(j),beta);

                    float hDiff = pair.getFirst() - (logU);
                    int tries = 0;
                    while(Math.abs(hDiff) > tolerance && tries < 50) {
                        if(hDiff > 0) {
                            betaMin = beta.get(j);
                            if(Float.isInfinite(betaMax))
                                beta.putScalar(j,beta.get(j) * 2);
                            else
                                beta.putScalar(j,(beta.get(j) + betaMax) / 2);
                        }
                        else {
                            betaMax = beta.get(j);
                            if(Float.isInfinite(betaMin))
                                beta.putScalar(j,beta.get(j) / 2);
                            else
                                beta.putScalar(j,(beta.get(j) + betaMin) / 2);
                        }

                        pair = hBeta(d.getRow(j),Nd4j.scalar(beta.get(j)));
                        hDiff = pair.getFirst() -  logU;
                        tries++;
                    }

                    p.putRow(j,pair.getSecond());

                }
            });
        }

        service.shutdown();
        try {
            service.awaitTermination(1, TimeUnit.DAYS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }


        return p;

    }






    /**
     *
     * @param X
     * @param nDims
     * @param initialDims
     * @param perplexity
     */
    public  INDArray calculate(INDArray X,int nDims,int initialDims,float perplexity) {
        if(usePca)
            X = pca(X,initialDims);
        if(nDims > X.shape().length)
            nDims = X.shape().length;

        //normalization
        if(normalize) {
            X = X.sub(X.min(Integer.MAX_VALUE));
            X = X.divi(X.max(Integer.MAX_VALUE));
            X = X.subiRowVector(X.mean(0));
        }


        INDArray sumX = Transforms.pow(X,2).sum(1);
        INDArray D = X.mmul(X.transpose()).muli(-2).addiColumnVector(sumX.transpose()).addiRowVector(sumX);




        INDArray y = Nd4j.randn(X.rows(),nDims).muli(1e-3f);
        INDArray yIncs = Nd4j.create(y.shape());
        INDArray gains = Nd4j.ones(y.shape());

        INDArray p = d2p(D,perplexity,tolerance);

        Nd4j.doAlongDiagonal(p,new Function<Number, Number>() {
            @Override
            public Number apply(Number input) {
                return 0;
            }
        });

        //diagonal op here
        p = p.add(p.transpose()).muli(0.5f);
        p = Transforms.max(p.diviRowVector(p.sum(0).addi(1e-6f)),realMin);
        float constant = Nd4j.getBlasWrapper().dot(p,Transforms.log(p));
        //lie for better local minima
        p.muli(4);
        float epsilon = 500;
        float costCheck = Float.NEGATIVE_INFINITY;
        for(int i = 0; i < maxIter; i++) {

            INDArray sumY = Transforms.pow(y,2).sum(1);
            //num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata'))))
            //Student-t distribution
            INDArray num = y.mmul(y.transpose()).muli(-2).addiColumnVector(sumY.transpose()).addiRowVector(sumY).addi(1).rdivi(1);
            Nd4j.doAlongDiagonal(num,new Function<Number, Number>() {
                @Override
                public Number apply(Number input) {
                    return 0;
                }
            });


            //Q = max(num ./ sum(num(:)), realmin);
            // normalize to get probabilities
            INDArray  q = Transforms.max(num.diviRowVector(num.sum(0).addi(1e-6f)),realMin);
            //L = (P - Q) .* num;
            INDArray L = p.sub(q).muli(num);
            // y_grads = 4 * (diag(sum(L, 1)) - L) * ydata;
            INDArray yGrads = Nd4j.diag(L.sum(0)).subi(L).mmul(y);
            if(i < stopLyingIteration)
                 yGrads.muli(4);
           // gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
            //        + (gains * .8) .* (sign(y_grads) == sign(y_incs));
            gains = gains.add(.2f).muli(sign(yGrads).eps(sign(yIncs)))
                    .addi(gains.mul(momentum))
                    .muli(sign(yGrads).eq(sign(yIncs)));

            BooleanIndexing.applyWhere(gains,new Condition() {
                @Override
                public Boolean apply(Number input) {
                    return input.floatValue() < minGain;
                }

                @Override
                public Boolean apply(IComplexNumber input) {
                    return input.absoluteValue().floatValue() < minGain;

                }
            },new Function<Number, Number>() {
                @Override
                public Number apply(Number input) {
                    return minGain;
                }
            });

            //y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
            yIncs = yIncs.mul(momentum).sub(epsilon).mul(gains.mul(yGrads));
            // ydata = ydata + y_incs;
            y.addi(yIncs);
            // ydata = bsxfun(@minus, ydata, mean(ydata, 1));
            y.subiRowVector(y.mean(0));

            if(i == switchMomentumIteration)
                momentum = finalMomentum;
            if(i % 10 == 0) {
                float cost = constant - Nd4j.getBlasWrapper().dot(p,Transforms.log(q));
                if(!Float.isInfinite(costCheck)) {
                    float diff = Math.abs(costCheck - cost);
                    if(diff < 1e-6)
                        break;
                }
                else
                      costCheck = cost;
                log.info("Cost " + cost + " at iteration " + i);
            }

            if(i == stopLyingIteration)
                p.divi(4);

        }

        return y;
    }

    public static class Builder {
        private int maxIter = 1000;
        private float realMin = 1e-6f;
        private float initialMomentum = 5e-1f;
        private float finalMomentum = 8e-1f;
        private int eta = 500;
        private float momentum = 5e-1f;
        private int switchMomentumIteration = 100;
        private boolean normalize = true;
        private boolean usePca = false;
        private int stopLyingIteration = 100;
        private float tolerance = 1e-5f;

        public Builder tolerance(float tolerance) {
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

        public Builder setRealMin(float realMin) {
            this.realMin = realMin;
            return this;
        }

        public Builder setInitialMomentum(float initialMomentum) {
            this.initialMomentum = initialMomentum;
            return this;
        }

        public Builder setFinalMomentum(float finalMomentum) {
            this.finalMomentum = finalMomentum;
            return this;
        }

        public Builder setEta(int eta) {
            this.eta = eta;
            return this;
        }

        public Builder setMomentum(float momentum) {
            this.momentum = momentum;
            return this;
        }

        public Builder setSwitchMomentumIteration(int switchMomentumIteration) {
            this.switchMomentumIteration = switchMomentumIteration;
            return this;
        }

        public TsneCalculation build() {
            return new TsneCalculation(maxIter, realMin, initialMomentum, finalMomentum, eta, momentum, switchMomentumIteration,normalize,usePca,stopLyingIteration,tolerance);
        }

    }
}
