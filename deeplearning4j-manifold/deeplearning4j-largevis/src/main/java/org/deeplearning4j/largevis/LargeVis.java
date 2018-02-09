package org.deeplearning4j.largevis;

import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.clustering.randomprojection.RPForest;
import org.deeplearning4j.clustering.randomprojection.RPTree;
import org.deeplearning4j.clustering.randomprojection.RPUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Counter;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;

@Data
public class LargeVis {

    private RPForest rpTree;
    @Builder.Default
    private int numWorkers = Runtime.getRuntime().availableProcessors();
    //vec.rows -> nVertices
    private INDArray vec,vis,prob;
    private INDArray knnVec,oldKnnVec;
    private int[] negTable;
    //may not need this?
    private long[] head,alias,next;
    private int numDims;
    private int maxSize;
    @Builder.Default
    private String distanceFunction = "euclidean";
    private INDArray edgeWeight;
    private int nEdges;
    private List<Long> reverse = new ArrayList<>();
    private ExecutorService threadExec;
    private int numTrees;
    @Builder.Default
    private int outDim = 2;
    @Builder.Default
    private double initialAlpha = 1.0;
    private int nSamples;
    @Builder.Default
    private int nNegatives = 5;
    @Builder.Default
    private int nNeighbors = 150;
    private int nTrees;
    @Builder.Default
    private int nPropagations = 3;
    @Builder.Default
    private double gamma = 7.0;
    @Builder.Default
    private double perplexity = 50.0;
    private int nVertices;
    @Builder.Default
    private long seed = 42;
    @Builder.Default
    private boolean normalize = true;

    @Builder
    public LargeVis(INDArray vec,
                    int numDims,
                    int maxSize,
                    String distanceFunction,
                    int numTrees,
                    int outDims,
                    int nNegatives,
                    double gamma,
                    double initialAlpha,
                    double perplexity,
                    int nPropagations,
                    long seed,int nNeighbors,
                    boolean normalize) {
        if(normalize) {
            NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
            normalizerStandardize.fit(new DataSet(vec,vec));
            normalizerStandardize.transform(vec);
        }

        this.vec = vec;
        this.numDims = numDims;
        this.distanceFunction = distanceFunction;
        this.outDim = outDims;
        this.nPropagations = nPropagations;
        this.initialAlpha = initialAlpha;
        this.nNeighbors = nNeighbors;
        this.nNegatives = nNegatives;
        this.gamma = gamma;
        this.perplexity = perplexity;
        this.seed = seed;
        this.numTrees = numTrees;
        rpTree = new RPForest(numTrees,maxSize,distanceFunction);
        rpTree.fit(vec);
        initNegTable();
        head = new long[vec.rows()];
        vis = Nd4j.create(vec.rows(),outDim);
        for(int i = 0; i < head.length; i++) {
            head[i] = -1;
        }

        edgeWeight = Nd4j.create(vec.rows());
        if(nSamples < 0) {
            if(nVertices < 10000) {
                nSamples = 1000;
            }
            else if(nVertices < 1000000) {
                nSamples = (nVertices - 10000) * 9000 / (1000000 - 10000) + 1000;
            }
            else
                nSamples = nVertices / 100;
        }

        nSamples *= 1000000;
        if(nTrees < 0) {
            if (nVertices < 100000)
                nTrees = 10;
            else if (nVertices < 1000000)
                nTrees = 20;

            else if (nVertices < 5000000)
                nTrees = 50;
            else nTrees = 100;

        }


        Nd4j.getRandom().setSeed(seed);

        knnVec = Nd4j.create(vec.rows());
    }


    private void initNegTable() {
        int negSize = (int) 1e8;
        reverse.clear();
        double sumWeights = 0.0;
        double dd = 0;
        INDArray weights = Nd4j.zeros(vec.rows());
        for(int i = 0; i < weights.length(); i++) {
            for(long p = head[i]; p >= 0; p = next[(int) p]) {
                weights.putScalar(i,
                        weights.getDouble(i) +
                                edgeWeight.getDouble(i));
            }

            sumWeights += weights.getDouble(i);
            weights.putScalar(i,Math.pow(weights.getDouble(i),0.75));

        }

        negTable = new int[negSize];
        for(int i = 0,x = 0; i < negTable.length; i++) {
            negTable[i] = x;
            if(i / (double) negSize  > dd / sumWeights && x < vec.rows() - 1) {
                dd += weights.getDouble(++x);
            }
        }

    }

    private void initAliasTable() {
        alias = new long[nEdges];
        prob = Nd4j.create(nEdges);
        INDArray normProb = Nd4j.create(nEdges);
        int[] largeBlock = new int[nEdges];
        int[] smallBlock = new int[nEdges];
        double sum = 0;
        int currSmallBlock,currLargeBlock;
        int numSmallBlock = 0;
        int numLargeBlock = 0;
        sum = edgeWeight.sumNumber().doubleValue();
        normProb = edgeWeight.muli(nEdges / sum);

        for(int k = nEdges - 1; k >= 0; --k)  {
            if (normProb.getDouble(k) < 1) {
                smallBlock[numSmallBlock++] = k;
            }
            else {
                largeBlock[numLargeBlock++] = k;
            }
        }

        while (numSmallBlock > 0 && numLargeBlock > 0) {
            currSmallBlock = smallBlock[--numSmallBlock];
            currLargeBlock = largeBlock[--numLargeBlock];
            prob.putScalar(currSmallBlock,normProb.getDouble(currSmallBlock));
            alias[currSmallBlock] = currLargeBlock;
            normProb.putScalar(currLargeBlock,normProb.getDouble(largeBlock) + normProb.getDouble(currSmallBlock));
            if(normProb.getDouble(currLargeBlock) < 1) {
                smallBlock[numSmallBlock++] = currLargeBlock;
            }
            else {
                largeBlock[numLargeBlock++] = currLargeBlock;
            }
        }

        while (numLargeBlock > 0) {
            prob.putScalar(largeBlock[--numLargeBlock],1);
        }

        while(numSmallBlock > 0) {
            prob.putScalar(smallBlock[--numSmallBlock],1);
        }


    }



    public void runKnn(INDArray query) {
        for(int i = 0; i < numWorkers; i++) {
            INDArray results = rpTree.queryAll(query,nNeighbors + 1 * nTrees);
            knnVec.put(new INDArrayIndex[]{NDArrayIndex.interval(i, results.length())},results);
        }

    }


    public void runPropagation() {

        for(int i = 0; i < numWorkers; i++) {
            runPropgation(i);
        }
    }


    public void runPropgation(int id) {
        int[] check = new int[vec.rows()];
        int low = id * vec.rows() / numWorkers;
        int hi = (id + 1) * vec.rows() / numWorkers;

        for(int i = 0; i < check.length; i++) {
            check[i] = -1;
        }

        int y = -1;


        Counter<Integer> distances = new Counter<>();
        for(int x = low; x < hi; x++) {
            check[x] = x;
            INDArray v1 = oldKnnVec.get(NDArrayIndex.interval(x,oldKnnVec.length()));
            for(int i = 0; i < v1.length(); i++) {
                y = v1.getInt(i);
                check[y] = x;
                distances.incrementCount(y, RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y)));
                if(distances.size() == nNeighbors + 1) {
                    //heap.pop() here which is it?
                    distances.removeKey(distances.argMax());
                }
            }

            for(int i = 0; i < v1.length(); i++) {
                INDArray v2 = oldKnnVec.get(NDArrayIndex.interval(v1.getInt(v1.getInt(i)),oldKnnVec.length()));
                for(int j = 0; j < v2.length(); j++) {
                    if(check[y = v2.getInt(j)] != x) {
                        check[y] = x;
                        distances.incrementCount(y, RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y)));
                    }
                }

                distances.incrementCount(y, RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y)));
                if(distances.size() == nNeighbors + 1) {
                    //heap.pop() here which is it?
                    distances.removeKey(distances.argMax());

                }
            }

            while(!distances.isEmpty()) {
                knnVec.putScalar(x,distances.removeKey(distances.argMax()));
            }
        }
    }


    public int sampleAnEdge(double rand1,double rand2) {
        int k = (int)( (nEdges - 0.1) * rand1);
        return rand2 <= prob.getDouble(k) ? k : (int) alias[k];
    }



    public void computeSimilarity(int id) {
        int low = id  * vec.rows() / numWorkers;
        int high = (id + 1) * vec.rows() / numWorkers;
        int loBeta = 0;
        int hiBeta = 0;
        int H = 0;
        double tmp = 0;
        double sumWeight = Double.MIN_VALUE;

        for(int x = low; x < high; x++) {
            int beta = 1;
            loBeta = hiBeta = -1;
            for(int iter = 0; iter < 200; iter++) {
                H = 0;
                for(int p = (int) head[x]; p >= 0; p = (int) next[p]) {
                    sumWeight  += tmp = Math.exp(- beta * edgeWeight.getDouble(p));
                    H += beta * (edgeWeight.getDouble(p) * tmp);
                }

                H = (int) ((H / sumWeight) + Math.log(sumWeight));
                if(Math.abs(H - Math.log(perplexity)) < 1e-5)
                    break;

                if(H > Math.log(perplexity)) {
                    loBeta = beta;
                    if(hiBeta < 0) {
                        beta *= 2;
                    }
                    else {
                        beta = (beta + hiBeta) / 2;
                    }

                    if(beta > Double.MAX_VALUE) {
                        beta = (int) Double.MAX_VALUE;
                    }
                }


                sumWeight = Double.MIN_VALUE;
                for(int p = (int) (head[x]); p >= 0; p = (int) next[p]) {
                    double newValue = Math.exp(-beta * edgeWeight.getDouble(p));
                    sumWeight +=  newValue;
                    edgeWeight.putScalar(p,newValue);
                }


                edgeWeight.divi(sumWeight);

                
            }
        }
    }


}
