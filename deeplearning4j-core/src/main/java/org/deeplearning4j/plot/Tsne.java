package org.deeplearning4j.plot;

import com.google.common.primitives.Ints;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.nd4j.linalg.factory.Nd4j.*;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * dl4j port of original t-sne algorithm described/implemented by van der Maaten and Hinton
 *
 * DECOMPOSED VERSION, DO NOT USE IT EVER
 *
 * @author raver119@gmail.com
 * @author Adam Gibson
 */
public class Tsne {
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
    //protected INDArray gains,yIncs;
    protected INDArray Y;
    protected transient IterationListener iterationListener;

    protected static final Logger logger = LoggerFactory.getLogger(Tsne.class);

    protected Tsne() {
        ;
    }

    protected void init() {

    }

    public INDArray calculate(INDArray X, int targetDimensions, double perplexity) {
        // pca hook
        if (usePca) {
            X = PCA.pca(X, Math.min(50,X.columns()),normalize);
        } else if(normalize) {
            X.subi(X.min(Integer.MAX_VALUE));
            X = X.divi(X.max(Integer.MAX_VALUE));
            X = X.subiRowVector(X.mean(0));
        }


        int n = X.rows();
        // FIXME: this is wrong, another distribution required here
        Y =randn(X.rows(),targetDimensions,Nd4j.getRandom());
        INDArray dY = Nd4j.zeros(n, targetDimensions);
        INDArray iY = Nd4j.zeros(n, targetDimensions);
        INDArray gains = Nd4j.ones(n, targetDimensions);

        boolean stopLying = false;
        logger.debug("Y:Shape is = " + Arrays.toString(Y.shape()));

        // compute P-values
        INDArray P = x2p(X, tolerance, perplexity);

        // do training
        for (int i = 0; i < maxIter; i++) {
            INDArray sumY =  pow(Y, 2).sum(1).transpose();

            //Student-t distribution
            //also un normalized q
            // also known as num in original implementation
            INDArray qu = Y.mmul(Y.transpose()).muli(-2)
                    .addiRowVector(sumY)
                    .transpose()
                    .addiRowVector(sumY)
                    .addi(1)
                    .rdivi(1);

  //          doAlongDiagonal(qu,new Zero());

            INDArray  Q =  qu.div(qu.sumNumber().doubleValue());
            BooleanIndexing.applyWhere(Q, Conditions.lessThan(1e-12), new Value(1e-12));

            INDArray PQ = P.sub(Q).muli(qu);

            logger.debug("PQ shape is: " + Arrays.toString(PQ.shape()));
            logger.debug("PQ.sum(1) shape is: " + Arrays.toString(PQ.sum(1).shape()));

            dY = diag(PQ.sum(1)).subi(PQ).mmul(Y).muli(4);


            if (i < switchMomentumIteration) {
                momentum = initialMomentum;
            } else {
                momentum = finalMomentum;
            }

            gains = gains.add(.2)
                    .muli(dY.cond(Conditions.greaterThan(0)).neqi(iY.cond(Conditions.greaterThan(0))))
                    .addi(gains.mul(0.8).muli(dY.cond(Conditions.greaterThan(0)).eqi(iY.cond(Conditions.greaterThan(0)))));


            BooleanIndexing.applyWhere(gains, Conditions.lessThan(minGain), new Value(minGain));


            INDArray gradChange = gains.mul(dY);

            gradChange.muli(learningRate);

            iY.muli(momentum).subi(gradChange);

            double cost = P.mul(log(P.div(Q),false)).sumNumber().doubleValue();
            logger.info("Iteration ["+ i +"] error is: [" + cost +"]");

            Y.addi(iY);
          //  Y.addi(iY).subiRowVector(Y.mean(0));
            INDArray tiled = Nd4j.tile(Y.mean(0), new int[]{Y.rows(), 1});
            Y.subi(tiled);

            if (!stopLying && (i > maxIter / 2 || i >= stopLyingIteration)) {
                P.divi(4);
                stopLying = true;
            }
        }
        return Y;
    }

    public INDArray diag(INDArray ds) {
        boolean isLong = ds.rows() > ds.columns();
        INDArray sliceZero = ds.slice(0);
        int dim = Math.max(ds.columns(),ds.rows());
        INDArray result = Nd4j.create(dim, dim);
        for (int i = 0; i < dim; i++) {
            INDArray sliceSrc = ds.slice(i);
            INDArray sliceDst = result.slice(i);
            for (int j = 0; j < dim; j++) {
                if(i==j) {
                    if(isLong)
                        sliceDst.putScalar(j, sliceSrc.getDouble(0));
                    else
                        sliceDst.putScalar(j, sliceZero.getDouble(i));
                }
            }
        }

        return result;
    }

    public void plot(INDArray matrix, int nDims, List<String> labels, String path) throws IOException {

        calculate(matrix,nDims,perplexity);

        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path),true));

        for(int i = 0; i < Y.rows(); i++) {
            if(i >= labels.size())
                break;
            String word = labels.get(i);
            if(word == null)
                continue;
            StringBuffer sb = new StringBuffer();
            INDArray wordVector = Y.getRow(i);
            for(int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if(j < wordVector.length() - 1)
                    sb.append(",");
            }

            sb.append(",");
            sb.append(word);
            sb.append(" ");

            sb.append("\n");
            write.write(sb.toString());

        }

        write.flush();
        write.close();
    }

    /**
     * Computes a gaussian kernel
     * given a vector of squared distance distances
     *
     * @param d the data
     * @param beta
     * @return
     */
    public Pair<Double,INDArray> hBeta(INDArray d,double beta) {
        INDArray P =  exp(d.neg().muli(beta));
        double sumP = P.sumNumber().doubleValue();
        double logSumP = FastMath.log(sumP);
        Double H = logSumP + ((beta * (d.mul(P).sumNumber().doubleValue())) / sumP);
        P.divi(sumP);
        return new Pair<>(H,P);
    }

    /**
     * This method build probabilities for given source data
     *
     * @param X
     * @param tolerance
     * @param perplexity
     * @return
     */
    private INDArray x2p(final INDArray X, double tolerance, double perplexity) {
        int n = X.rows();
        final INDArray p = zeros(n, n);
        final INDArray beta =  ones(n, 1);
        final double logU =  Math.log(perplexity);

        INDArray sumX =  pow(X, 2).sum(1);

        logger.debug("sumX shape: " + Arrays.toString(sumX.shape()));

        INDArray times = X.mmul(X.transpose()).muli(-2);

        logger.debug("times shape: " + Arrays.toString(times.shape()));

        INDArray prodSum = times.transpose().addiColumnVector(sumX);

        logger.debug("prodSum shape: " + Arrays.toString(prodSum.shape()));

        INDArray D = X.mmul(X.transpose()).mul(-2) // thats times
                .transpose().addColumnVector(sumX) // thats prodSum
                .addRowVector(sumX.transpose()); // thats D

        logger.info("Calculating probabilities of data similarities...");
        logger.debug("Tolerance: " + tolerance);
        for(int i = 0; i < n; i++) {
            if(i % 500 == 0 && i > 0)
                logger.info("Handled [" + i + "] records out of ["+ n +"]");

            double betaMin = Double.NEGATIVE_INFINITY;
            double betaMax = Double.POSITIVE_INFINITY;
            int[] vals = Ints.concat(ArrayUtil.range(0,i),ArrayUtil.range(i + 1, n ));
            INDArrayIndex[] range = new INDArrayIndex[]{new SpecifiedIndex(vals)};

            INDArray row = D.slice(i).get(range);
            Pair<Double,INDArray> pair =  hBeta(row,beta.getDouble(i));
            //INDArray hDiff = pair.getFirst().sub(logU);
            double hDiff = pair.getFirst() - logU;
            int tries = 0;

            //while hdiff > tolerance
            while(Math.abs(hDiff) > tolerance && tries < 50) {
                //if hdiff > 0
                if(hDiff > 0) {
                    betaMin = beta.getDouble(i);
                    if(Double.isInfinite(betaMax))
                        beta.putScalar(i,beta.getDouble(i) * 2.0);
                    else
                        beta.putScalar(i,(beta.getDouble(i) + betaMax) / 2.0);
                } else {
                    betaMax = beta.getDouble(i);
                    if(Double.isInfinite(betaMin))
                        beta.putScalar(i,beta.getDouble(i) / 2.0);
                    else
                        beta.putScalar(i,(beta.getDouble(i) + betaMin) / 2.0);
                }

                pair = hBeta(row,beta.getDouble(i));
                hDiff = pair.getFirst() - logU;
                tries++;
            }
            p.slice(i).put(range,pair.getSecond());
        }


        //dont need data in memory after
        logger.info("Mean value of sigma " + sqrt(beta.rdiv(1)).mean(Integer.MAX_VALUE));
        BooleanIndexing.applyWhere(p,Conditions.isNan(),new Value(1e-12));

        //set 0 along the diagonal
        INDArray permute = p.transpose();

        INDArray pOut = p.add(permute);

        pOut.divi(pOut.sumNumber().doubleValue() + 1e-6);

        pOut.muli(4);

        BooleanIndexing.applyWhere(pOut,Conditions.lessThan(1e-12),new Value(1e-12));
        //ensure no nans

        return pOut;
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

        public Tsne build() {
            Tsne tsne = new Tsne();
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
