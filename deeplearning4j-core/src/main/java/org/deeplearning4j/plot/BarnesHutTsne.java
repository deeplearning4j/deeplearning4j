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

package org.deeplearning4j.plot;


import static org.nd4j.linalg.ops.transforms.Transforms.*;

import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.sptree.SpTree;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;
import org.nd4j.linalg.learning.AdaGrad;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.nd4j.linalg.factory.Nd4j.ones;
import static org.nd4j.linalg.factory.Nd4j.randn;
import static org.nd4j.linalg.factory.Nd4j.zeros;


/**
 * Barnes hut algorithm for TSNE, uses a dual tree approximation approach.
 * Work based on:
 * http://lvdmaaten.github.io/tsne/
 * @author Adam Gibson
 */
public class BarnesHutTsne extends Tsne implements Model {
    private int N;
    private double theta;
    private INDArray rows;
    private INDArray cols;
    private INDArray vals;
    private String simiarlityFunction = "cosinesimilarity";
    private boolean invert = true;
    private INDArray x;
    private int numDimensions = 0;
    public final static String Y_GRAD = "yIncs";
    private SpTree tree;
    private INDArray gains;
    private INDArray yIncs;

    public BarnesHutTsne(INDArray x,
                         INDArray y,
                         int numDimensions,
                         double perplexity,
                         double theta,
                         int maxIter,
                         int stopLyingIteration,
                         int momentumSwitchIteration,
                         double momentum,
                         double finalMomentum,
                         double learningRate) {

        this.Y = y;
        this. x = x;
        this.numDimensions = numDimensions;
        this.perplexity = perplexity;
        this.theta = theta;
        this.maxIter = maxIter;
        this.stopLyingIteration = stopLyingIteration;
        this.momentum = momentum;
        this.finalMomentum = finalMomentum;
        this.learningRate = learningRate;
        this.switchMomentumIteration = momentumSwitchIteration;
    }

    public BarnesHutTsne(INDArray x,
                         INDArray y,
                         int numDimensions,
                         String simiarlityFunction,
                         double theta,
                         boolean invert,
                         int maxIter,
                         double realMin,
                         double initialMomentum,
                         double finalMomentum,
                         double momentum,
                         int switchMomentumIteration,
                         boolean normalize,
                         boolean usePca,
                         int stopLyingIteration,
                         double tolerance,
                         double learningRate,
                         boolean useAdaGrad,
                         double perplexity,
                         double minGain) {
        super();
        // maxIter, realMin,initialMomentum,finalMomentum,momentum,switchMomentumIteration,normalize, usePca,stopLyingIteration,tolerance,learningRate,useAdaGrad,perplexity,minGain
        this.maxIter = maxIter;
        this.realMin = realMin;
        this.initialMomentum = initialMomentum;
        this.finalMomentum = finalMomentum;
        this.momentum = momentum;
        this.normalize = normalize;
        this.useAdaGrad = useAdaGrad;
        this.usePca = usePca;
        this.stopLyingIteration = stopLyingIteration;
        this.learningRate = learningRate;
        this.switchMomentumIteration = switchMomentumIteration;
        this.tolerance = tolerance;
        this.perplexity = perplexity;
        this.minGain = minGain;

        this.Y = y;
        this.x = x;
        this.numDimensions = numDimensions;
        this.simiarlityFunction = simiarlityFunction;
        this.theta = theta;
        this.invert = invert;
    }


    public String getSimiarlityFunction() {
        return simiarlityFunction;
    }

    public void setSimiarlityFunction(String simiarlityFunction) {
        this.simiarlityFunction = simiarlityFunction;
    }

    public boolean isInvert() {
        return invert;
    }

    public void setInvert(boolean invert) {
        this.invert = invert;
    }

    public double getTheta(){
        return theta;
    }

    public double getPerplexity(){
        return perplexity;
    }

    /**
     * Convert data to probability
     * co-occurrences (aka calculating the kernel)
     * @param d the data to convert
     * @param u the perplexity of the model
     * @return the probabilities of co-occurrence
     */
    public INDArray computeGaussianPerplexity(final INDArray d,  double u) {
        N = d.rows();

        final int k = (int) (3 * u);
        if(u > k)
            throw new IllegalStateException("Illegal k value " + k + "greater than " + u);


        rows = zeros(1,N + 1);
        cols = zeros(1,N * k);
        vals = zeros(1,N * k);

        for(int n = 0; n < N; n++)
            rows.putScalar(n + 1,rows.getDouble(n) + k);


        final INDArray beta =  ones(N, 1);

        final double logU =  FastMath.log(u);
        VPTree tree = new VPTree(d,simiarlityFunction,invert);

        logger.info("Calculating probabilities of data similarities...");
        for(int i = 0; i < N; i++) {
            if(i % 500 == 0)
                logger.info("Handled " + i + " records");

            double betaMin = -Double.MAX_VALUE;
            double betaMax = Double.MAX_VALUE;
            List<DataPoint> results = new ArrayList<>();
            tree.search(new DataPoint(i,d.slice(i)),k + 1,results,new ArrayList<Double>());
            double betas = beta.getDouble(i);

            INDArray cArr = VPTree.buildFromData(results);
            Pair<INDArray,Double> pair =  computeGaussianKernel(cArr, beta.getDouble(i),k);
            INDArray currP = pair.getFirst();
            double hDiff =  pair.getSecond() - logU;
            int tries = 0;
            boolean found = false;
            //binary search
            while(!found && tries < 200) {
                if(hDiff < tolerance && -hDiff < tolerance)
                    found = true;
                else {
                    if(hDiff > 0) {
                        betaMin = betas;

                        if(betaMax == Double.MAX_VALUE || betaMax == -Double.MAX_VALUE)
                            betas *= 2;
                        else
                            betas = (betas + betaMax) / 2.0;
                    }
                    else {
                        betaMax = betas;
                        if(betaMin == -Double.MAX_VALUE || betaMin == Double.MAX_VALUE)
                            betas /= 2.0;
                        else
                            betas = (betas + betaMin) / 2.0;
                    }

                    pair = computeGaussianKernel(cArr, betas,k);
                    hDiff = pair.getSecond() - logU;
                    tries++;
                }

            }


            currP.divi(currP.sum(Integer.MAX_VALUE));
            INDArray indices = Nd4j.create(1,k + 1);
            for(int j = 0; j < indices.length(); j++) {
                if(j >= results.size())
                    break;
                indices.putScalar(j, results.get(j).getIndex());
            }

            for(int l = 0; l < k; l++) {
                cols.putScalar(rows.getInt(i) + l,indices.getDouble(l + 1));
                vals.putScalar(rows.getInt(i) + l,currP.getDouble(l));
            }



        }
        return vals;

    }

    @Override
    public INDArray input() {
        return x;
    }

    @Override
    public void validateInput() {

    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return null;
    }

    @Override
    public INDArray getParam(String param) {
        return null;
    }

    @Override
    public void initParams() {

    }

    @Override
    public Map<String, INDArray> paramTable() {
        return null;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {

    }

    @Override
    public void setParam(String key, INDArray val) {

    }

    @Override
    public void clear(){}

    /* compute the gradient given the current solution, the probabilities and the constant */
    protected Pair<Double,INDArray> gradient(INDArray p) {
        throw new UnsupportedOperationException();
    }






    /**
     * Symmetrize the value matrix
     * @param rowP
     * @param colP
     * @param valP
     * @return
     */
    public INDArray symmetrized(INDArray rowP,INDArray colP,INDArray valP) {
        INDArray rowCounts = Nd4j.create(N);
        for(int n = 0; n < N; n++) {
            int begin = rowP.getInt(n);
            int end = rowP.getInt(n + 1);
            for(int i = begin; i < end; i++) {
                boolean present = false;
                for(int m = rowP.getInt(colP.getInt(i)); m < rowP.getInt(colP.getInt(i) + 1); m++)
                    if(colP.getInt(m) == n) {
                        present = true;
                    }


                if(present)
                    rowCounts.putScalar(n,rowCounts.getDouble(n) + 1);

                else {
                    rowCounts.putScalar(n,rowCounts.getDouble(n) + 1);
                    rowCounts.putScalar(colP.getInt(i),rowCounts.getDouble(colP.getInt(i)) + 1);
                }
            }
        }


        int numElements = rowCounts.sum(Integer.MAX_VALUE).getInt(0);
        INDArray offset = Nd4j.create(N);
        INDArray symRowP = Nd4j.create(N + 1);
        INDArray symColP = Nd4j.create(numElements);
        INDArray symValP = Nd4j.create(numElements);

        for(int n = 0; n < N; n++)
            symRowP.putScalar(n + 1,symRowP.getDouble(n) + rowCounts.getDouble(n));




        for(int n = 0; n < N; n++) {
            for(int i = rowP.getInt(n); i < rowP.getInt(n + 1); i++) {
                boolean present = false;
                for(int m = rowP.getInt(colP.getInt(i)); m < rowP.getInt(colP.getInt(i)) + 1; m++) {
                    if(colP.getInt(m) == n) {
                        present = true;
                        if(n < colP.getInt(i)) {
                            // make sure we do not add elements twice
                            symColP.putScalar(symRowP.getInt(n) + offset.getInt(n),colP.getInt(i));
                            symColP.putScalar(symRowP.getInt(colP.getInt(i)) + offset.getInt(colP.getInt(i)), n);
                            symValP.putScalar(symRowP.getInt(n) + offset.getInt(n),valP.getDouble(i) + valP.getDouble(m));
                            symValP.putScalar(symRowP.getInt(colP.getInt(i)) + offset.getInt(colP.getInt(i)) ,valP.getDouble(i) + valP.getDouble(m));
                        }
                    }
                }

                // If (colP[i], n) is not present, there is no addition involved
                if(!present) {
                    int colPI = colP.getInt(i);
                    if(n < colPI) {
                        symColP.putScalar(symRowP.getInt(n) + offset.getInt(n), colPI);
                        symColP.putScalar(symRowP.getInt(colP.getInt(i)) + offset.getInt(colPI),n);
                        symValP.putScalar(symRowP.getInt(n) + offset.getInt(n),valP.getDouble(i));
                        symValP.putScalar(symRowP.getInt(colPI) + offset.getInt(colPI),valP.getDouble(i));
                    }

                }

                // Update offsets
                if(!present || (present && n < colP.getInt(i))) {
                    offset.putScalar(n,offset.getInt(n)+ 1);
                    int colPI = colP.getInt(i);
                    if(colPI != n)
                        offset.putScalar(colPI,offset.getDouble(colPI) + 1);
                }
            }
        }

        // Divide the result by two
        symValP.divi(2.0);


        return symValP;

    }

    /**
     * Computes a gaussian kernel
     * given a vector of squared distance distances
     *
     * @param distances
     * @param beta
     * @return
     */
    public Pair<INDArray,Double> computeGaussianKernel(INDArray distances, double beta,int k) {
        // Compute Gaussian kernel row
        INDArray currP = Nd4j.create(k);
        for(int m = 0; m < k; m++)
            currP.putScalar(m, FastMath.exp(-beta * distances.getDouble(m + 1)));

        double sum = currP.sum(Integer.MAX_VALUE).getDouble(0);
        double h = 0.0;
        for(int m = 0; m < k; m++)
            h += beta * (distances.getDouble(m + 1) * currP.getDouble(m));

        h = (h / sum) + FastMath.log(sum);

        return new Pair<>(currP,h);
    }



    @Override
    public void fit() {
        boolean exact = theta == 0.0;
        if(exact)
            Y = super.calculate(x,numDimensions,perplexity);

        else {
            //output
            if(Y == null)
                Y = randn(x.rows(),numDimensions,Nd4j.getRandom()).muli(1e-3f);


            computeGaussianPerplexity(x,perplexity);
            vals = symmetrized(rows, cols, vals).divi(vals.sum(Integer.MAX_VALUE));
            //lie about gradient
            vals.muli(12);
            for(int i = 0; i < maxIter; i++) {
                step(vals,i);

                if(i == switchMomentumIteration)
                    momentum = finalMomentum;
                if(i == stopLyingIteration)
                    vals.divi(12);


                if(iterationListener != null)
                    iterationListener.iterationDone(this,i);
                logger.info("Error at iteration " + i + " is " + score());
            }
        }
    }


    /**
     * An individual iteration
     * @param p the probabilities that certain points
     *          are near each other
     * @param i the iteration (primarily for debugging purposes)
     */
    public void step(INDArray p, int i) {
        update(gradient().getGradientFor(Y_GRAD), Y_GRAD);
    }


    @Override
    public void update(INDArray gradient, String paramType) {
        INDArray yGrads = gradient;

        gains = gains.add(.2)
                .muli(sign(yGrads)).neqi(sign(yIncs))
                .addi(gains.mul(0.8).muli(sign(yGrads)).neqi(sign(yIncs)));

        BooleanIndexing.applyWhere(
                gains,
                Conditions.lessThan(minGain),
                new Value(minGain));


        INDArray gradChange = gains.mul(yGrads);

        if(useAdaGrad) {
            if(adaGrad == null)
                adaGrad = new AdaGrad();
            gradChange = adaGrad.getGradient(gradChange,0);

        }

        else
            gradChange.muli(learningRate);

        yIncs.muli(momentum).subi(gradChange);
        Y.addi(yIncs);

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

        fit(matrix, nDims);

        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path)));

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


    @Override
    public double score() {
        // Get estimate of normalization term
        INDArray buff = Nd4j.create(numDimensions);
        AtomicDouble sum_Q = new AtomicDouble(0.0);
        for(int n = 0; n < N; n++)
            tree.computeNonEdgeForces(n, theta, buff, sum_Q);

        // Loop over all edges to compute t-SNE error
        double C = .0;
        INDArray linear = Y;
        for(int n = 0; n < N; n++) {
            int begin = rows.getInt(n);
            int end = rows.getInt(n + 1);
            int ind1 = n;
            for(int i = begin; i < end; i++) {
                int ind2 = cols.getInt(i);
                buff.assign(linear.slice(ind1));
                buff.subi(linear.slice(ind2));

                double Q = pow(buff,2).sum(Integer.MAX_VALUE).getDouble(0);
                Q = (1.0 / (1.0 + Q)) / sum_Q.doubleValue();
                C += vals.getDouble(i) * FastMath.log(vals.getDouble(i) + Nd4j.EPS_THRESHOLD) / (Q + Nd4j.EPS_THRESHOLD);
            }
        }

        return C;
    }

    @Override
    public void computeGradientAndScore() {

    }

    @Override
    public void accumulateScore(double accum) {

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
    public int numParams(boolean backwards) {
        return 0;
    }

    @Override
    public void setParams(INDArray params) {

    }

    @Override
    public void setParamsViewArray(INDArray params) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void applyLearningRateScoreDecay() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void fit(INDArray data) {
        this.x  = data;
        fit();
    }
    
    public void fit(INDArray data, int nDims) {
        this.x = data;
        this.numDimensions = nDims;
        fit();
    }

    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient gradient() {
        if(yIncs == null)
            yIncs =  zeros(Y.shape());
        if(gains == null)
            gains = ones(Y.shape());

        AtomicDouble sumQ = new AtomicDouble(0);
        /* Calculate gradient based on barnes hut approximation with positive and negative forces */
        INDArray posF = Nd4j.create(Y.shape());
        INDArray negF = Nd4j.create(Y.shape());
        if(tree == null)
            tree = new SpTree(Y);
        tree.computeEdgeForces(rows,cols,vals,N,posF);

        for(int n = 0; n < N; n++)
            tree.computeNonEdgeForces(n,theta,negF.slice(n),sumQ);


        INDArray dC = posF.subi(negF.divi(sumQ));

        Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(Y_GRAD,dC);
        return ret;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(),score());
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


    public static class Builder extends  Tsne.Builder {
        private double theta = 0.0;
        private boolean invert = true;
        private String similarityFunction = "cosinesimilarity";



        public Builder similarityFunction(String similarityFunction) {
            this.similarityFunction = similarityFunction;
            return this;
        }

        public Builder invertDistanceMetric(boolean invert){
            this.invert = invert;
            return this;
        }

        public Builder theta(double theta) {
            this.theta = theta;
            return this;
        }

        @Override
        public Builder minGain(double minGain) {
            super.minGain(minGain);
            return this;
        }

        @Override
        public Builder perplexity(double perplexity) {
            super.perplexity(perplexity);
            return this;
        }

        @Override
        public Builder useAdaGrad(boolean useAdaGrad) {
            super.useAdaGrad(useAdaGrad);
            return this;
        }

        @Override
        public Builder learningRate(double learningRate) {
            super.learningRate(learningRate);
            return this;
        }

        @Override
        public Builder tolerance(double tolerance) {
            super.tolerance(tolerance);
            return this;
        }

        @Override
        public Builder stopLyingIteration(int stopLyingIteration) {
            super.stopLyingIteration(stopLyingIteration);
            return this;
        }

        @Override
        public Builder usePca(boolean usePca) {
            super.usePca(usePca);
            return this;
        }

        @Override
        public Builder normalize(boolean normalize) {
            super.normalize(normalize);
            return this;
        }

        @Override
        public Builder setMaxIter(int maxIter) {
            super.setMaxIter(maxIter);
            return this;
        }

        @Override
        public Builder setRealMin(double realMin) {
            super.setRealMin(realMin);
            return this;
        }

        @Override
        public Builder setInitialMomentum(double initialMomentum) {
            super.setInitialMomentum(initialMomentum);
            return this;
        }

        @Override
        public Builder setFinalMomentum(double finalMomentum) {
            super.setFinalMomentum(finalMomentum);
            return this;
        }

        @Override
        public Builder setMomentum(double momentum) {
            super.setMomentum(momentum);
            return this;
        }

        @Override
        public Builder setSwitchMomentumIteration(int switchMomentumIteration) {
            super.setSwitchMomentumIteration(switchMomentumIteration);
            return this;
        }

        @Override
        public BarnesHutTsne build() {
            return new BarnesHutTsne(null, null, 2, similarityFunction,theta,invert,
                    maxIter,realMin,initialMomentum,finalMomentum,momentum,switchMomentumIteration,normalize,
                    usePca,stopLyingIteration,tolerance,learningRate,useAdaGrad,perplexity,minGain);
        }


    }

}
