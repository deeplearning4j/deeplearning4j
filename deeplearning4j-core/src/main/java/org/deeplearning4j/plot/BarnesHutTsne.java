package org.deeplearning4j.plot;

import com.google.common.util.concurrent.AtomicDouble;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.clustering.quadtree.QuadTree;
import org.deeplearning4j.clustering.vptree.VpTreeNode;
import org.deeplearning4j.clustering.vptree.VpTreePointINDArray;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;


import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.nd4j.linalg.factory.Nd4j.ones;
import static org.nd4j.linalg.factory.Nd4j.zeros;
import static org.nd4j.linalg.ops.transforms.Transforms.*;


/**
 * Barnes hut algorithm for TSNE, uses a dual tree approximation approach.
 * Work based on:
 * http://lvdmaaten.github.io/tsne/
 * @author Adam Gibson
 */
public class BarnesHutTsne extends Tsne implements Model {
    private int n;
    private int d;
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
                         int n,
                         int d,
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
        super();
        this.n = n;
        this.d = d;
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
        int N = d.rows();
        final int k = (int) (3 * u);
        rows = zeros(N + 1);

        cols = zeros(N,k);

        vals = zeros(N,k);

        for(int n = 1; n < N; n++)
            rows.putScalar(n,rows.getDouble(n - 1) + k);


        final INDArray beta =  ones(N, 1);

        final double logU =  Math.log(u);

        final List<VpTreePointINDArray> list = VpTreePointINDArray.dataPoints(d);
        final VpTreeNode<VpTreePointINDArray> tree = VpTreeNode.buildVpTree(list);

        log.info("Calculating probabilities of data similarities...");
        ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        for(int i = 0; i < N; i++) {
            if(i % 500 == 0)
                log.info("Handled " + i + " records");
            final int j = i;
            service.submit(new Runnable() {
                @Override
                public void run() {
                    double betaMin = Float.NEGATIVE_INFINITY;
                    double betaMax = Float.POSITIVE_INFINITY;
                    Counter<VpTreePointINDArray> c = tree.findNearByPointsWithDistancesK(list.get(j),k + 1);

                    INDArray row = d.slice(j);
                    Pair<INDArray,INDArray> pair =  hBeta(row,toNDArray(c),beta.getDouble(j));
                    INDArray currP = pair.getSecond();
                    INDArray hDiff = pair.getFirst().sub(logU);
                    int tries = 0;


                    //while hdiff > tolerance
                    while(BooleanIndexing.and(abs(hDiff), Conditions.greaterThan(tolerance)) && tries < 50) {
                        //if hdiff > 0
                        if(BooleanIndexing.and(hDiff,Conditions.greaterThan(0))) {
                            if(Double.isInfinite(betaMax))
                                beta.putScalar(j,beta.getDouble(j) * 2.0);
                            else
                                beta.putScalar(j,(beta.getDouble(j) + betaMax) / 2.0);
                            betaMin = beta.getDouble(j);
                        }
                        else {
                            if(Double.isInfinite(betaMin))
                                beta.putScalar(j,beta.getDouble(j) / 2.0);
                            else
                                beta.putScalar(j,(beta.getDouble(j) + betaMin) / 2.0);
                            betaMax = beta.getDouble(j);
                        }

                        pair = hBeta(row,toNDArray(c),beta.getDouble(j));
                        hDiff = pair.getFirst().subi(logU);
                        tries++;
                    }

                    INDArray currPAssign = currP.div(currP.sum(Integer.MAX_VALUE));
                    INDArray indices = toIndex(c);


                    for(int i = 0; i < k; i++) {
                        cols.putScalar(new int[]{rows.getInt(n),i},indices.getDouble(i + 1));
                        vals.putScalar(new int[]{rows.getInt(n),i},currPAssign.getDouble(i));
                    }

                    cols.slice(j).assign(toIndex(c));


                }
            });

        }


        try {
            service.shutdown();
            service.awaitTermination(1, TimeUnit.DAYS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
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

    /* compute the gradient given the current solution, the probabilities and the constant */
    protected Pair<Double,INDArray> gradient(INDArray p) {
        this.p = p;
        return new Pair<>(score(),getGradient().gradientLookupTable().get(Y_GRAD));
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

    private INDArray toIndex(Counter<VpTreePointINDArray> counter) {
        INDArray ret = Nd4j.create(counter.size());
        List<VpTreePointINDArray> list = counter.getSortedKeys();
        for(int i = 0; i < list.size(); i++) {
            ret.putScalar(i,list.get(i).getIndex());
        }
        return ret;
    }

    private INDArray toNDArray(Counter<VpTreePointINDArray> counter) {
        INDArray ret = Nd4j.create(counter.size());
        List<VpTreePointINDArray> list = counter.getSortedKeys();
        for(int i = 0; i < list.size(); i++) {
            ret.putScalar(i,counter.getCount(list.get(i)));
        }
        return ret;
    }


    /**
     * Computes a gaussian kernel
     * given a vector of squared euclidean distances
     *
     * @param d the data
     * @param beta
     * @return
     */
    public Pair<INDArray,INDArray> hBeta(INDArray d,INDArray distances,double beta) {
        INDArray P =  exp(d.neg().muli(beta).muli(distances));
        INDArray sum = P.sum(Integer.MAX_VALUE);
        INDArray otherSum = d.mul(P).sum(0);
        INDArray H = log(sum)
                .addi(otherSum.muli(beta).muli(distances).divi(sum));

        P.divi(sum);
        return new Pair<>(H,P);
    }



    @Override
    public void fit() {
        boolean exact = theta == 0.0;
        if(exact)
            y = super.calculate(x,numDimensions,perplexity);


        else {
            INDArray p = computeGaussianPerplexity(x,perplexity);

            for(int i = 0; i < maxIter; i++) {
                step(p,i);

                if(i == switchMomentumIteration)
                    momentum = finalMomentum;
                if(i == stopLyingIteration)
                    p.divi(4);

                if(iterationListener != null)
                    iterationListener.iterationDone(i);

            }

        }
    }






    @Override
    public void update(Gradient gradient) {

    }

    @Override
    public double score() {
        // Get estimate of normalization term
        int QT_NO_DIMS = 2;
        QuadTree tree = new QuadTree(y);
        INDArray buff = Nd4j.create(QT_NO_DIMS);
        AtomicDouble sum_Q = new AtomicDouble(0.0);
        for(int n = 0; n < y.rows(); n++)
            tree.computeNonEdgeForces(n, theta, buff, sum_Q);

        // Loop over all edges to compute t-SNE error
        double C = .0, Q;
        for(int n = 0; n < y.rows(); n++) {
            INDArray row1 = rows.slice(n);
            int begin = row1.getInt(0);
            int end = row1.getInt(1);
            for(int i = begin; i < end; i++) {
                buff.assign(y.slice(n));
                buff.subi(cols.getRow(i));
                Q = Nd4j.getBlasWrapper().dot(buff,buff);
                Q = (1.0 / (1.0 + Q)) / sum_Q.doubleValue();
                double val = vals.getDouble(i,0);
                C += val * Math.log((val + Float.MIN_VALUE) / (Q + Float.MAX_VALUE));
            }
        }

        return C;
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
        if(yIncs == null)
            yIncs =  zeros(y.shape());
        if(gains == null)
            gains = ones(y.shape());

        AtomicDouble sum_Q = new AtomicDouble(0);
        INDArray pos_f = Nd4j.create(p.rows(), p.columns());
        INDArray neg_f = Nd4j.create(p.rows() ,p.columns());

        QuadTree quad = new QuadTree(p);
        quad.computeEdgeForces(rows,cols,p,p.rows(),pos_f);

        for(int n = 0; n < p.rows(); n++) {
            quad.computeNonEdgeForces(n,theta,neg_f,sum_Q);
        }

        INDArray dC = pos_f.subi(neg_f.divi(sum_Q));

        INDArray yGrads = dC;

        gains = gains.add(.2)
                .muli(yGrads.cond(Conditions.greaterThan(0)).neqi(yIncs.cond(Conditions.greaterThan(0))))
                .addi(gains.mul(0.8).muli(yGrads.cond(Conditions.greaterThan(0)).eqi(yIncs.cond(Conditions.greaterThan(0)))));

        BooleanIndexing.applyWhere(
                gains,
                Conditions.lessThan(minGain),
                new Value(minGain));


        INDArray gradChange = gains.mul(yGrads);

        if(useAdaGrad)
            gradChange = adaGrad.getGradient(gradChange);
        else
            gradChange.muli(learningRate);


        yIncs.muli(momentum).subi(gradChange);


        Gradient ret = new DefaultGradient();
        ret.gradientLookupTable().put("yIncs",yIncs);
        return ret;
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
