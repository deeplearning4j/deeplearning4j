/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.plot;


import static org.nd4j.linalg.ops.transforms.Transforms.*;

import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.clustering.quadtree.QuadTree;
import org.deeplearning4j.clustering.vptree.VpTreeNode;
import org.deeplearning4j.clustering.vptree.VpTreePointINDArray;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;
import org.nd4j.linalg.learning.AdaGrad;


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
    private double perplexity;
    private double theta;
    private INDArray rows;
    private INDArray cols;
    private INDArray vals;
    private INDArray p;
    private INDArray x;
    private int numDimensions = 0;
    public final static String Y_GRAD = "yIncs";

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

        this.y = y;
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



    /**
     * Convert data to probability
     * co-occurrences (aka calculating the kernel)
     * @param d the data to convert
     * @param u the perplexity of the model
     * @return the probabilities of co-occurrence
     */
    @Override
    public INDArray computeGaussianPerplexity(final INDArray d,  double u) {
        N = d.rows();

        final int k = (int) (3 * u);
        if(u > k)
            throw new IllegalStateException("Illegal k value " + k + "greater than " + u);


        rows = zeros(N + 1);
        cols = zeros(N * k);
        vals = zeros(N * k);

        for(int n = 0; n < N; n++)
            rows.putScalar(n + 1,rows.getDouble(n) + k);


        final INDArray beta =  ones(N, 1);

        final double logU =  Math.log(u);

        final List<VpTreePointINDArray> list = VpTreePointINDArray.dataPoints(d);
        final VpTreeNode<VpTreePointINDArray> tree = VpTreeNode.buildVpTree(list);

        log.info("Calculating probabilities of data similarities...");
        for(int i = 0; i < N; i++) {
            if(i % 500 == 0)
                log.info("Handled " + i + " records");

            double betaMin = Double.NEGATIVE_INFINITY;
            double betaMax = Double.POSITIVE_INFINITY;
            Counter<VpTreePointINDArray> c = tree.findNearByPointsWithDistancesK(list.get(i),k + 1);

            INDArray cArr = toNDArray(c);
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

                    pair = computeGaussianKernel(toNDArray(c), beta.getDouble(i),k);
                    hDiff = pair.getSecond() - logU;
                    tries++;
                }

            }


            currP.divi(currP.sum(Integer.MAX_VALUE));
            INDArray indices = toIndex(c,k);


            for(int l = 0; l < indices.length(); l++) {
                cols.putScalar(rows.getInt(i) + l,l < indices.length() - 1 ? indices.getDouble(l + 1) : 0.0);
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
    public void clear() {
        this.p = null;
    }

    /* compute the gradient given the current solution, the probabilities and the constant */
    @Override
    protected Pair<Double,INDArray> gradient(INDArray p) {
        throw new UnsupportedOperationException();
    }




    private INDArray toIndex(Counter<VpTreePointINDArray> counter,int k) {
        INDArray ret = Nd4j.create(k);
        List<VpTreePointINDArray> list = counter.getSortedKeys();
        for(int i = 0; i < list.size(); i++) {
            ret.putScalar(i,list.get(i).getIndex());
        }
        return ret;
    }

    private INDArray toNDArray(Counter<VpTreePointINDArray> counter) {
        INDArray ret = Nd4j.create(N - 1);
        List<VpTreePointINDArray> list = counter.getSortedKeys();
        for(int i = 0; i < list.size(); i++) {
            ret.putScalar(i,counter.getCount(list.get(i)));
        }
        return ret;
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
                        break;
                    }


                if(present)
                    rowCounts.putScalar(n,rowCounts.getDouble(n) + 1);

                else {
                    rowCounts.putScalar(n,rowCounts.getDouble(n) + 1);
                    double putIdx = rowCounts.getDouble(colP.getInt(i));
                    rowCounts.putScalar(colP.getInt(i),putIdx + 1);
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
                        if(n <= colP.getInt(i)) {
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
                    if(n <= colPI) {
                        symColP.putScalar(symRowP.getInt(n) + offset.getInt(n), colPI);
                        symColP.putScalar(symRowP.getInt(colP.getInt(i)) + offset.getInt(colPI),n);
                        symValP.putScalar(symRowP.getInt(n) + offset.getInt(n),valP.getDouble(i));
                        symValP.putScalar(symRowP.getInt(colPI) + offset.getInt(colPI),valP.getDouble(i));
                    }

                }

                // Update offsets
                if(!present || (present && n <= colP.getInt(i))) {
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
     * given a vector of squared euclidean distances
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
        double h = Double.MIN_VALUE;
        for(int m = 0; m < k; m++)
            h += beta * (distances.getDouble(m + 1) * currP.getDouble(m));

        h = (h / sum) + FastMath.log(sum);

        return new Pair<>(currP,h);
    }



    @Override
    public void fit() {
        boolean exact = theta == 0.0;
        if(exact)
            y = super.calculate(x,numDimensions,perplexity);

        else {
            //output
            if(y == null)
                y = randn(x.rows(),numDimensions,Nd4j.getRandom()).muli(1e-3f);


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
                log.info("Error at iteration " + i + " is " + score());


            }

        }
    }


    /**
     * An individual iteration
     * @param p the probabilities that certain points
     *          are near each other
     * @param i the iteration (primarily for debugging purposes)
     */
    @Override
    public void step(INDArray p,int i) {
        Gradient g = gradient();
        update(g);
        y.addi(yIncs);

    }


    @Override
    public void update(Gradient gradient) {
        INDArray yGrads = gradient.gradientForVariable().get(Y_GRAD);

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
                adaGrad = new AdaGrad(gradChange.shape());
            gradChange = adaGrad.getGradient(gradChange);

        }
        else
            gradChange.muli(learningRate);


        yIncs.muli(momentum).subi(gradChange);


        gradient.gradientForVariable().put(Y_GRAD,yIncs);

    }

    @Override
    public double score() {
        // Get estimate of normalization term
        int QT_NO_DIMS = 2;
        QuadTree tree = new QuadTree(y);
        INDArray buff = Nd4j.create(QT_NO_DIMS);
        AtomicDouble sum_Q = new AtomicDouble(0.0);
        for(int n = 0; n < N; n++)
            tree.computeNonEdgeForces(n, theta, buff, sum_Q);

        // Loop over all edges to compute t-SNE error
        double C = .0;
        INDArray linear = y.linearView();
        for(int n = 0; n < N; n++) {
            int begin = rows.getInt(n);
            int end = rows.getInt(n + 1);
            int ind1 = n * QT_NO_DIMS;
            for(int i = begin; i < end; i++) {
                int ind2 = cols.getInt(i) * QT_NO_DIMS;
                buff.assign(linear.get(NDArrayIndex.interval(ind1, ind1 + QT_NO_DIMS)));
                buff.subi(linear.get(NDArrayIndex.interval(ind2, ind2 + QT_NO_DIMS)));

                double Q = pow(buff,2).sum(Integer.MAX_VALUE).getDouble(0);
                Q = (1.0 / (1.0 + Q)) / sum_Q.doubleValue();
                double val = vals.getDouble(i);
                double add = Math.log((val + Double.MIN_VALUE) / (Q + Double.MAX_VALUE));
                if(Double.isNaN(add) || Double.isInfinite(add))
                    add = Nd4j.EPS_THRESHOLD;
                C += val * add;
            }
        }

        return C;
    }

    @Override
    public void setScore() {

    }

    @Override
    public void accumulateScore(double accum) {

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
        this.x  = data;
        fit();
    }

    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient gradient() {
        if(yIncs == null)
            yIncs =  zeros(y.shape());
        if(gains == null)
            gains = ones(y.shape());

        AtomicDouble sumQ = new AtomicDouble(0);
        /* Calculate gradient based on barnes hut approximation with positive and negative forces */
        INDArray posF = Nd4j.create(y.shape());
        INDArray negF = Nd4j.create(y.shape());

        QuadTree quad = new QuadTree(y);
        quad.computeEdgeForces(rows,cols,vals,N,posF);

        for(int n = 0; n < N; n++)
            quad.computeNonEdgeForces(n,theta,negF.slice(n),sumQ);


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
            BarnesHutTsne t = new BarnesHutTsne(null,null,2,perplexity,theta,maxIter,this.stopLyingIteration,this.switchMomentumIteration,this.momentum,this.finalMomentum,this.learningRate);
            t.useAdaGrad = useAdaGrad;
            t.usePca = usePca;
            return t;
        }
    }
}
