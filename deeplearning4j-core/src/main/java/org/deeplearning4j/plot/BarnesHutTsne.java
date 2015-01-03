package org.deeplearning4j.plot;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.clustering.vptree.VpTreeNode;
import org.deeplearning4j.clustering.vptree.VpTreePointINDArray;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.nd4j.linalg.factory.Nd4j.ones;
import static org.nd4j.linalg.factory.Nd4j.zeros;
import static org.nd4j.linalg.ops.transforms.Transforms.abs;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;


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



    /**
     * Convert data to probability
     * co-occurrences (aka calculating the kernel)
     * @param d the data to convert
     * @param u the perplexity of the model
     * @return the probabilities of co-occurrence
     */
    public INDArray computeGaussianPerplexity(final INDArray d,  double u) {
        int n = d.rows();
        int k = (int) (3 * u);
        final INDArray p = zeros(n, n);
        final INDArray beta =  ones(n, 1);
        final double logU =  Math.log(u);
        final List<VpTreePointINDArray> list = VpTreePointINDArray.dataPoints(d);
        final VpTreeNode<VpTreePointINDArray> tree = VpTreeNode.buildVpTree(list);
        log.info("Calculating probabilities of data similarities..");
        ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        for(int i = 0; i < n; i++) {
            if(i % 500 == 0)
                log.info("Handled " + i + " records");
            final int j = i;
            service.submit(new Runnable() {
                @Override
                public void run() {
                    double betaMin = Float.NEGATIVE_INFINITY;
                    double betaMax = Float.POSITIVE_INFINITY;
                    tree.findNearbyPoints(list.get(j),0.99);
                    NDArrayIndex[] range = new NDArrayIndex[]{
                            NDArrayIndex.concat(NDArrayIndex.interval(0, j),NDArrayIndex.interval(j + 1, d.columns()))};

                    INDArray row = d.slice(j).get(range);
                    Pair<INDArray,INDArray> pair =  hBeta(row,beta.getDouble(j));
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

                        pair = hBeta(row,beta.getDouble(j));
                        hDiff = pair.getFirst().subi(logU);
                        tries++;
                    }

                    p.slice(j).put(range,pair.getSecond());

                }
            });

        }


        try {
            service.shutdown();
            service.awaitTermination(1, TimeUnit.DAYS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        //dont need data in memory after
        d.data().flush();

        log.info("Mean value of sigma " + sqrt(beta.rdiv(1)).mean(Integer.MAX_VALUE));
        BooleanIndexing.applyWhere(p,Conditions.isNan(),new Value(realMin));

        //set 0 along the diagonal
        INDArray permute = p.transpose();



        INDArray pOut = p.add(permute);
        BooleanIndexing.applyWhere(pOut,Conditions.isNan(),new Value(realMin));

        pOut.divi(pOut.sum(Integer.MAX_VALUE));
        BooleanIndexing.applyWhere(pOut,Conditions.lessThan(1e-12),new Value(1e-12));
        //ensure no nans
        return pOut;

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
