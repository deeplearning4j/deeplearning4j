package org.deeplearning4j.plot;

import com.google.common.primitives.Ints;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;
import org.nd4j.linalg.indexing.functions.Zero;
import org.nd4j.linalg.io.ClassPathResource;
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

import static org.nd4j.linalg.factory.Nd4j.doAlongDiagonal;
import static org.nd4j.linalg.factory.Nd4j.ones;
import static org.nd4j.linalg.factory.Nd4j.zeros;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * dl4j port of original t-sne algorithm described/implemented by van der Maaten and Hinton
 *
 * DECOMPOSED VERSION, DO NOT USE IT EVER
 *
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
    //protected INDArray gains,yIncs;
    protected INDArray Y;
    protected transient IterationListener iterationListener;

    protected static final Logger logger = LoggerFactory.getLogger(ProperTsne.class);

    protected ProperTsne() {
        ;
    }

    protected void init() {

    }

    public void calculate(INDArray X, int targetDimensions, double perplexity) {
        // pca hook
        if (usePca) {

        }

        // normalization hook
        if (normalize) {

        }

        int n = X.rows();
        // FIXME: this is wrong, another distribution required here
        Y = Nd4j.rand(new int[]{n, targetDimensions}, new NormalDistribution());
        INDArray dY = Nd4j.zeros(n, targetDimensions);
        INDArray iY = Nd4j.zeros(n, targetDimensions);
        INDArray gains = Nd4j.ones(n, targetDimensions);

        boolean stopLying = false;
        System.out.println("Y:Shape is = " + Arrays.toString(Y.shape()));

        // compute P-values
        INDArray P = x2p(X, tolerance, perplexity);

        // do training
        for (int i = 0; i < maxIter; i++) {
            INDArray sumY =  pow(Y, 2).sum(1);

            //Student-t distribution
            //also un normalized q
            // also known as num in original implementation
            INDArray qu = Y.mmul(
                    Y.transpose())
                    .muli(-2)
                    .addiRowVector(sumY).transpose()
                    .addiRowVector(sumY)
                    .addi(1).rdivi(1);

  //          doAlongDiagonal(qu,new Zero());

            INDArray  Q =  qu.div(qu.sumNumber().doubleValue());
            BooleanIndexing.applyWhere(Q, Conditions.lessThan(1e-12), new Value(1e-12));

            INDArray PQ = P.sub(Q);
            for (int y = 0; y < n; y++) {
                INDArray sum1 = Nd4j.tile(PQ.getRow(i).mul(qu.getRow(i)), new int[]{Y.columns(), 1})
                        .transpose().mul(Y.getRow(i).broadcast(Y.shape()).sub(y)).sum(0);
                dY.putRow(i, sum1);
            }

            gains = gains.add(.2)
                    .muli(dY.cond(Conditions.greaterThan(0)).neqi(iY.cond(Conditions.greaterThan(0))))
                    .addi(gains.mul(0.8).muli(dY.cond(Conditions.greaterThan(0)).eqi(iY.cond(Conditions.greaterThan(0)))));


            BooleanIndexing.applyWhere(gains, Conditions.lessThan(1e-12), new Value(1e-12));


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

    private INDArray square(INDArray X) {
        INDArray F = X.dup();
        for (int y = 0; y< F.rows(); y++) {
            INDArray slice = F.slice(y);
            for (int x = 0; x < F.columns(); x++) {
                slice.putScalar(x, Math.pow(slice.getDouble(x), 2));
            }
        }
        return F;
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
        double sum = P.sumNumber().doubleValue();
        double logSum = FastMath.log(sum);
        Double H = logSum + beta * (d.mul(P).sumNumber().doubleValue()) / sum;
        P.divi(sum + 1e-6);
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

        INDArray D = X.mmul(X.transpose()).muli(-2)
                .addRowVector(sumX)
                .transpose()
                .addRowVector(sumX);

        logger.info("Calculating probabilities of data similarities..");
        for(int i = 0; i < n; i++) {
            if(i % 500 == 0 && i > 0)
                logger.info("Handled " + i + " records");

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
        //    logger.info("numTries: " + tries);
//            System.out.println("Row: " + Arrays.toString(pair.getSecond().data().asFloat()));
            p.slice(i).put(range,pair.getSecond());

//            System.out.println("Slice: " + Arrays.toString(p.slice(i).dup().data().asFloat()));
        }

        //scanND(p, 1e-40f, 1);

        //dont need data in memory after
        logger.info("Mean value of sigma " + sqrt(beta.rdiv(1)).mean(Integer.MAX_VALUE));
        BooleanIndexing.applyWhere(p,Conditions.isNan(),new Value(1e-12));

        //scanND(p, 1e-40f, 1);

        //set 0 along the diagonal
        INDArray permute = p.transpose();

        INDArray pOut = p.add(permute);

        pOut.divi(pOut.sumNumber().doubleValue());

        BooleanIndexing.applyWhere(pOut,Conditions.lessThan(1e-12),new Value(1e-12));
        //ensure no nans

   //     scanND(pOut, 1e-40f, 1);

        pOut.muli(4);

        return pOut;
    }

    protected void scanND(INDArray array, double threshold, int op) {
        for (int y = 0; y < array.rows(); y++) {
            INDArray slice = array.slice(y);
            //logger.info("Slice " + y + ": " + Arrays.toString(slice.data().asFloat()));
            for (int x = 0; x < array.columns(); x++) {
                if (slice.getDouble(x) > 0)
                    logger.info("position ["+y + "," + x +"] is ["+slice.getDouble(x)+"]");
            }
        }
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
