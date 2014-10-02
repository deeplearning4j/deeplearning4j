package org.deeplearning4j.plot;

import com.google.common.base.Function;
import org.apache.commons.lang3.time.StopWatch;
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
    private final float minGain = 1e-1f;
    private float momentum = initialMomentum;
    private int switchMomentumIteration = 100;
    private boolean normalize = true;

    public TsneCalculation(int maxIter, float realMin, float initialMomentum, float finalMomentum, int eta, float momentum, int switchMomentumIteration,boolean normalize) {
        this.maxIter = maxIter;
        this.realMin = realMin;
        this.normalize = normalize;
        this.initialMomentum = initialMomentum;
        this.finalMomentum = finalMomentum;
        this.eta = eta;
        this.momentum = momentum;
        this.switchMomentumIteration = switchMomentumIteration;
    }

    public Pair<Float,INDArray> hBeta(INDArray d,INDArray beta) {
        if(d.length() != beta.length() && !beta.isScalar())
            throw new IllegalArgumentException("D != beta");

        INDArray P = Transforms.exp(d.neg().muli(beta));
        float sum = P.sum(Integer.MAX_VALUE).get(0);
        // H = log(sumP) + beta * sum(D .* P) / sumP;

        float H = (float) Math.log(sum) + d.mul(P).muli(beta).sum(Integer.MAX_VALUE).get(0) / sum;
        P.divi(sum + 1e-6f);

        return new Pair<>(H,P);
    }




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

    public INDArray x2p(INDArray X,float tol,float perplexity) {
        int n = X.rows();

        INDArray  sumX = Transforms.pow(X,2).sum(1);

        INDArray D = X.mul(-2).mmul(X.transpose()).addiRowVector(sumX).transpose().addiRowVector(sumX);

        INDArray P = Nd4j.zeros(n, n);

        INDArray beta = Nd4j.ones(n,1);

        float logU = (float) Math.log(perplexity);
        StopWatch watch = new StopWatch();

        for(int i = 0; i < n; i++) {
            watch.start();
            if (i % 500 == 0)
                System.out.println( "Computing P-values for point "+ i + " of " +  n +  "...");

            float betaMin = Float.NEGATIVE_INFINITY;
            float betaMax = Float.POSITIVE_INFINITY;

            INDArray di = D.getRow(i);
            Pair<Float,INDArray> hPair = hBeta(di,beta);
            float h = hPair.getFirst();
            float hDiff = h - logU;
            int tries = 0;

            while(Math.abs(hDiff) > tol && tries < 50) {
                if(hDiff > 0) {
                    betaMin = beta.get(i);
                    if(Float.isInfinite(betaMax)) {
                        beta.putScalar(i,beta.get(i) * 2);
                    }
                    else
                        beta.putScalar(i,(beta.get(i) + betaMin) / 2);

                }
                else {
                    betaMax = beta.get(i);
                    if(Float.isInfinite(betaMin)) {
                        beta.putScalar(i,beta.get(i) / 2);
                    }
                    else
                        beta.putScalar(i,(beta.get(i) + betaMin) / 2);
                }

                hPair = hBeta(di,Nd4j.scalar(beta.get(i)));
                hDiff = hPair.getFirst() * logU;

                tries++;
            }

            P.putRow(i,hPair.getSecond());
            watch.stop();
            System.out.println("Row took " + watch.getTime());
            watch.reset();

        }

        return P;
    }



    public  void calculate(INDArray X,int nDims,int initialDims,float perplexity) {
        X = pca(X,initialDims);
        if(nDims > X.shape().length)
            nDims = X.shape().length;
        INDArray y = Nd4j.randn(X.rows(),nDims);
        INDArray iy = Nd4j.create(X.rows(),nDims);
        INDArray gains = Nd4j.ones(X.rows(),nDims);

        INDArray p = x2p(X,1e-5f,perplexity);
        p.addi(p.transpose());
        p.divi(p.sum(Integer.MAX_VALUE));
        p.muli(4);
        p.addi(1e-6f);
        float epsilon = 500;
        for(int i = 0; i < maxIter; i++) {

            INDArray sumY = Transforms.pow(y,2).sum(1);
            INDArray inside = y.mmul(y.transpose()).muli(-2).addi(sumY).transpose().addi(sumY).addi(1);
            INDArray num = Nd4j.ones(p.shape()).divi(inside);
            num.put(new NDArrayIndex[] {NDArrayIndex.interval(0,X.rows()),NDArrayIndex.interval(0,X.rows())},Nd4j.create(X.shape()));
            INDArray  q = Transforms.max(num.div(num.sum(Integer.MAX_VALUE)),realMin);
            INDArray L = p.sub(q).mul(num);
            // y_grads = 4 * (diag(sum(L, 1)) - L) * ydata;
            INDArray yGrads = Nd4j.diag(L.sum(0)).sub(L).mul(y);
            gains = gains.add(.2f).muli(sign(yGrads).eps(sign(iy))).addi(gains.mul(0.8f)).muli(sign(yGrads).eq(sign(iy)));

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
            iy.addi(iy.mul(momentum).sub(epsilon).mul(gains.mul(yGrads)));
            // ydata = ydata + y_incs;
            y.addi(iy);
            // ydata = bsxfun(@minus, ydata, mean(ydata, 1));
            y.subiRowVector(y.mean(0));

            if(i == switchMomentumIteration)
                momentum = finalMomentum;

        }
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

        public Builder normallize(boolean normalize) {
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
            return new TsneCalculation(maxIter, realMin, initialMomentum, finalMomentum, eta, momentum, switchMomentumIteration,normalize);
        }

    }
}
