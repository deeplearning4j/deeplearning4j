/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.ui.plot;


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
import org.nd4j.linalg.indexing.functions.Zero;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.List;

import static org.nd4j.linalg.factory.Nd4j.*;
import static org.nd4j.linalg.ops.transforms.Transforms.*;


/**
 * Tsne calculation
 * @author Adam Gibson
 */
@Deprecated
public class LegacyTsne implements Serializable {

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
    protected static ClassPathResource r = new ClassPathResource("/scripts/tsne.py");
    protected static final Logger log = LoggerFactory.getLogger(LegacyTsne.class);

    public LegacyTsne() {}

    public LegacyTsne(
            int maxIter,
            double realMin,
            double initialMomentum,
            double finalMomentum,
            double momentum,
            int switchMomentumIteration,
            boolean normalize,
            boolean usePca,
            int stopLyingIteration,
            double tolerance,double learningRate,boolean useAdaGrad,double perplexity,double minGain) {
        this.tolerance = tolerance;
        this.minGain = minGain;
        this.useAdaGrad = useAdaGrad;
        this.learningRate = learningRate;
        this.stopLyingIteration = stopLyingIteration;
        this.maxIter = maxIter;
        this.realMin = realMin;
        this.normalize = normalize;
        this.initialMomentum = initialMomentum;
        this.usePca = usePca;
        this.finalMomentum = finalMomentum;
        this.momentum = momentum;
        this.switchMomentumIteration = switchMomentumIteration;
        this.perplexity = perplexity;
    }

    /**
     * Computes a gaussian kernel
     * given a vector of squared distance distances
     *
     * @param d the data
     * @param beta
     * @return
     */
    public Pair<INDArray,INDArray> hBeta(INDArray d,double beta) {
        INDArray P =  exp(d.neg().muli(beta));
        double sum = P.sumNumber().doubleValue();
        double logSum = FastMath.log(sum);
        INDArray H = d.mul(P).sum(0).muli(beta).divi(sum).addi(logSum);
        P.divi(sum);
        return new Pair<>(H,P);
    }





    /**
     * Convert data to probability
     * co-occurrences (aka calculating the kernel)
     * @param d the data to convert
     * @param u the perplexity of the model
     * @return the probabilities of co-occurrence
     */
    public INDArray computeGaussianPerplexity(final INDArray d,  double u) {
        int n = d.rows();
        final INDArray p = zeros(n, n);
        final INDArray beta =  ones(n, 1);
        final double logU =  Math.log(u);

        log.info("Calculating probabilities of data similarities..");
        for(int i = 0; i < n; i++) {
            if(i % 500 == 0 && i > 0)
                log.info("Handled " + i + " records");

            double betaMin = Double.NEGATIVE_INFINITY;
            double betaMax = Double.POSITIVE_INFINITY;
            int[] vals = Ints.concat(ArrayUtil.range(0,i),ArrayUtil.range(i + 1,d.columns()));
            INDArrayIndex[] range = new INDArrayIndex[]{new SpecifiedIndex(vals)};

            INDArray row = d.slice(i).get(range);
            Pair<INDArray,INDArray> pair =  hBeta(row,beta.getDouble(i));
            INDArray hDiff = pair.getFirst().sub(logU);
            int tries = 0;


            //while hdiff > tolerance
            while(BooleanIndexing.and(abs(hDiff), Conditions.greaterThan(tolerance)) && tries < 50) {
                //if hdiff > 0
                if(BooleanIndexing.and(hDiff,Conditions.greaterThan(0))) {
                    if(Double.isInfinite(betaMax))
                        beta.putScalar(i,beta.getDouble(i) * 2.0);
                    else
                        beta.putScalar(i,(beta.getDouble(i) + betaMax) / 2.0);
                    betaMin = beta.getDouble(i);
                }
                else {
                    if(Double.isInfinite(betaMin))
                        beta.putScalar(i,beta.getDouble(i) / 2.0);
                    else
                        beta.putScalar(i,(beta.getDouble(i) + betaMin) / 2.0);
                    betaMax = beta.getDouble(i);
                }

                pair = hBeta(row,beta.getDouble(i));
                hDiff = pair.getFirst().subi(logU);
                tries++;
            }

            p.slice(i).put(range,pair.getSecond());


        }



        //dont need data in memory after
        log.info("Mean value of sigma " + sqrt(beta.rdiv(1)).mean(Integer.MAX_VALUE));
        BooleanIndexing.applyWhere(p,Conditions.isNan(),new Value(realMin));

        //set 0 along the diagonal
        INDArray permute = p.transpose();



        INDArray pOut = p.add(permute);

        pOut.divi(pOut.sum(Integer.MAX_VALUE));
        BooleanIndexing.applyWhere(pOut,Conditions.lessThan(Nd4j.EPS_THRESHOLD),new Value(Nd4j.EPS_THRESHOLD));
        //ensure no nans
        return pOut;

    }






    /**
     *
     * @param X
     * @param nDims
     * @param perplexity
     */
    public  INDArray calculate(INDArray X,int nDims,double perplexity) {
        if(usePca)
            X = PCA.pca(X, Math.min(50,X.columns()),normalize);
            //normalization (don't normalize again after pca)
         if(normalize) {
            X.subi(X.min(Integer.MAX_VALUE));
            X = X.divi(X.max(Integer.MAX_VALUE));
            X = X.subiRowVector(X.mean(0));
        }

        if(nDims > X.columns())
            nDims = X.columns();

        INDArray sumX =  pow(X, 2).sum(1);


        INDArray D = X.mmul(
                X.transpose()).muli(-2)
                .addRowVector(sumX)
                .transpose()
                .addRowVector(sumX);


        //output
        if(y == null)
            y = randn(X.rows(),nDims,Nd4j.getRandom()).muli(1e-3f);



        INDArray p = computeGaussianPerplexity(D, perplexity);


        //lie for better local minima
        p.muli(4);

        //init adagrad where needed
        if(useAdaGrad) {
            if(adaGrad == null) {
                adaGrad = new AdaGrad(learningRate);
            }
        }




        for(int i = 0; i < maxIter; i++) {
            step(p,i);

            if(i == switchMomentumIteration)
                momentum = finalMomentum;
            if(i == stopLyingIteration)
                p.divi(4);

            if(iterationListener != null)
                iterationListener.iterationDone(null,i);

        }

        return y;
    }


    /* compute the gradient given the current solution, the probabilities and the constant */
    protected Pair<Double,INDArray> gradient(INDArray p) {
        INDArray sumY =  pow(y, 2).sum(1);
        if(yIncs == null)
            yIncs =  zeros(y.shape());
        if(gains == null)
            gains = ones(y.shape());



        //Student-t distribution
        //also un normalized q
        INDArray qu = y.mmul(
                y.transpose())
                .muli(-2)
                .addiRowVector(sumY).transpose()
                .addiRowVector(sumY)
                .addi(1).rdivi(1);

        int n = y.rows();

        //set diagonal to zero
        doAlongDiagonal(qu,new Zero());



        // normalize to get probabilities
        INDArray  q =  qu.div(qu.sum(Integer.MAX_VALUE));

        BooleanIndexing.applyWhere(
                q,
                Conditions.lessThan(realMin),
                new Value(realMin));


        INDArray PQ = p.sub(q);

        INDArray yGrads = getYGradient(n,PQ,qu);

        gains = gains.add(.2)
                .muli(yGrads.cond(Conditions.greaterThan(0)).neqi(yIncs.cond(Conditions.greaterThan(0))))
                .addi(gains.mul(0.8).muli(yGrads.cond(Conditions.greaterThan(0)).eqi(yIncs.cond(Conditions.greaterThan(0)))));

        BooleanIndexing.applyWhere(
                gains,
                Conditions.lessThan(minGain),
                new Value(minGain));


        INDArray gradChange = gains.mul(yGrads);

        if(useAdaGrad)
            gradChange = adaGrad.getGradient(gradChange,0);
        else
            gradChange.muli(learningRate);


        yIncs.muli(momentum).subi(gradChange);


        double cost = p.mul(log(p.div(q),false)).sum(Integer.MAX_VALUE).getDouble(0);
        return new Pair<>(cost,yIncs);
    }


    public INDArray getYGradient(int n,INDArray PQ,INDArray qu) {
        INDArray yGrads = Nd4j.create(y.shape());
        for(int i = 0; i < n; i++) {
            INDArray sum1 = Nd4j.tile(PQ.getRow(i).mul(qu.getRow(i)), new int[]{y.columns(), 1})
                    .transpose().mul(y.getRow(i).broadcast(y.shape()).sub(y)).sum(0);
            yGrads.putRow(i, sum1);
        }

        return yGrads;
    }

    /**
     * An individual iteration
     * @param p the probabilities that certain points
     *          are near each other
     * @param i the iteration (primarily for debugging purposes)
     */
    public void step(INDArray p,int i) {
        Pair<Double,INDArray> costGradient = gradient(p);
        INDArray yIncs = costGradient.getSecond();
        log.info("Cost at iteration " + i + " was " + costGradient.getFirst());
        y.addi(yIncs);
        y.addi(yIncs).subiRowVector(y.mean(0));
        INDArray tiled = Nd4j.tile(y.mean(0), new int[]{y.rows(), 1});
        y.subi(tiled);

    }


    /**
     * Plot tsne (write the coordinates file)
     * @param matrix the matrix to plot
     * @param nDims the number of dimensions
     * @param labels
     * @throws IOException
     */
    public void plot(INDArray matrix,int nDims,List<String> labels) throws IOException {
        plot(matrix, nDims, labels, "coords.csv");
    }

    /**
     * Plot tsne
     * @param matrix the matrix to plot
     * @param nDims the number
     * @param labels
     * @param path the path to write
     * @throws IOException
     */
    public void plot(INDArray matrix,int nDims,List<String> labels,String path) throws IOException {

        calculate(matrix,nDims,perplexity);

        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path),true));

        for(int i = 0; i < y.rows(); i++) {
            if(i >= labels.size())
                break;
            String word = labels.get(i);
            if(word == null)
                continue;
            StringBuffer sb = new StringBuffer();
            INDArray wordVector = y.getRow(i);
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


    public INDArray getY() {
        return y;
    }

    public void setY(INDArray y) {
        this.y = y;
    }

    public IterationListener getIterationListener() {
        return iterationListener;
    }

    public void setIterationListener(IterationListener iterationListener) {
        this.iterationListener = iterationListener;
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

        public LegacyTsne build() {
            return new LegacyTsne(maxIter, realMin, initialMomentum, finalMomentum, momentum, switchMomentumIteration,normalize,usePca,stopLyingIteration,tolerance,learningRate,useAdaGrad,perplexity,minGain);
        }

    }
}
