package org.deeplearning4j.largevis;

import com.sun.istack.internal.NotNull;
import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.clustering.randomprojection.RPForest;
import org.deeplearning4j.clustering.randomprojection.RPUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.linalg.util.MathUtils;
import org.nd4j.list.IntNDArrayList;
import org.nd4j.list.NDArrayList;
import org.nd4j.list.matrix.IntMatrixNDArrayList;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

@Data
@Slf4j
public class LargeVis {

    private RPForest rpTree;
    @Builder.Default
    private int numWorkers = Runtime.getRuntime().availableProcessors();
    //vec.rows -> nVertices
    private INDArray vec,vis,prob;
    @Builder.Default
    private IUpdater updater = new Sgd(0.01);
    private GradientUpdater gradientUpdater;
    /**
     * KNNVec is a pointer to a vector.
     * This tends to be a list of vectors.
     *
     * Double indexing is actually just a get(i,j)
     * in a matrix.
     */
    private IntMatrixNDArrayList knnVec,oldKnnVec;
    private int[] negTable;
    //may not need this?
    @Builder.Default
    private IntNDArrayList head = new IntNDArrayList();
    @Builder.Default
    private int[] alias;
    @Builder.Default
    private IntNDArrayList next = new IntNDArrayList();
    private int maxSize;
    @Builder.Default
    private String distanceFunction = "euclidean";
    private NDArrayList edgeWeight;
    private int nEdges;
    @Builder.Default
    private IntNDArrayList reverse = new IntNDArrayList();
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

    private int negSize = (int) 1e8;

    private int edgeCountActual = 0;

    private IntNDArrayList edgeFrom = new IntNDArrayList();
    private IntNDArrayList edgeTo = new IntNDArrayList();
    private ExecutorService executorService;
    private Object lock = new Object();
    protected final AtomicInteger workerCounter = new AtomicInteger(0);
    // log uncaught exceptions
    Thread.UncaughtExceptionHandler handler = new Thread.UncaughtExceptionHandler() {
        public void uncaughtException(Thread th, Throwable ex) {
            log.error("Uncaught exception: " + ex);
            ex.printStackTrace();
        }
    };


    @Builder
    public LargeVis(INDArray vec,
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
                    boolean normalize,
                    IUpdater updater) {

        this.updater = updater;
        this.normalize = normalize;
        this.vec = vec;
        if(maxSize > 0)
            this.maxSize = maxSize;
        if(distanceFunction != null)
            this.distanceFunction = distanceFunction;
        if(outDims > 0)
            this.outDim = outDims;
        if(nPropagations > 0)
            this.nPropagations = nPropagations;
        if(initialAlpha > 0)
            this.initialAlpha = initialAlpha;
        if(nNeighbors > 0)
            this.nNeighbors = nNeighbors;
        if(nNegatives > 0)
            this.nNegatives = nNegatives;
        if(gamma > 0)
            this.gamma = gamma;
        if(perplexity > 0)
            this.perplexity = perplexity;
        if(seed > 0)
            this.seed = seed;
        if(numTrees > 0)
            this.numTrees = numTrees;
        if(outDims > 0)
            this.outDim = outDims;
        head = new IntNDArrayList();
        vis = Nd4j.create(vec.rows(),outDim);
        for(int i = 0; i < vec.rows(); i++) {
            head.add(-1);
        }

        edgeWeight = new NDArrayList();

        this.executorService = Executors.newFixedThreadPool(numWorkers, new ThreadFactory() {
            @Override
            public Thread newThread(@NotNull Runnable r) {
                Thread t = Executors.defaultThreadFactory().newThread(r);

                int cThread = workerCounter.getAndIncrement();

                t.setName("LargeVis thread " + cThread);
                t.setDaemon(true);
                t.setUncaughtExceptionHandler(handler);

                Nd4j.getAffinityManager().attachThreadToDevice(t,
                        cThread % Nd4j.getAffinityManager().getNumberOfDevices());

                return t;
            }
        });

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
        knnVec = new IntMatrixNDArrayList();
        //pre allocate up to vec.rows() (vertices) for the knn vectors
        for(int i = 0; i < vec.rows(); i++) {
            IntNDArrayList ndArrayList = new IntNDArrayList();
            ndArrayList.allocateWithSize(vec.rows());
            knnVec.add(ndArrayList);
        }

    }


    private void initNegTable() {
        log.info("Initializing neg table");
        reverse.clear();
        double sumWeights = 0.0;
        double dd = 0;
        INDArray weights = Nd4j.zeros(vec.rows());
        for(int i = 0; i < weights.length(); i++) {
            for(int p = head.get(i).intValue(); p >= 0; p = next.get(p).intValue()) {
                double result = weights.getDouble(i) +
                        edgeWeight.get(i).doubleValue();
                weights.putScalar(i, result);
            }

            sumWeights += weights.getDouble(i);
            weights.putScalar(i,Math.pow(weights.getDouble(i),0.75));

        }

        negTable = new int[negSize];
        for(int i = 0,x = 0; i < negSize; i++) {
            negTable[i] = x;
            if(i / (double) negSize  > dd / sumWeights && x < vec.rows() - 1) {
                dd += weights.getDouble(++x);
            }
        }

        log.info("Initialized neg table");

    }

    private void initAliasTable() {
        log.info("Initializing alias table");
        alias = new int[nEdges];
        prob = Nd4j.create(1,nEdges);
        INDArray normProb;
        int[] largeBlock = new int[nEdges];
        int[] smallBlock = new int[nEdges];
        double sum = 0;
        int currSmallBlock,currLargeBlock;
        int numSmallBlock = 0;
        int numLargeBlock = 0;
        sum = edgeWeight.array().sumNumber().doubleValue();
        normProb = edgeWeight.array().muli(nEdges / sum);
        int len = edgeWeight.array().length();

        for(int k = len - 1; k >= 0; k--)  {
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
            normProb.putScalar(currLargeBlock,normProb.getDouble(currLargeBlock) + normProb.getDouble(currSmallBlock));
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

        log.info("Initialized alias table");

    }




    public void runPropagation() {
        LinkedList<PropagationThread> list = new LinkedList<>();
        oldKnnVec = knnVec;
        knnVec = new IntMatrixNDArrayList();
        //pre allocate up to vec.rows() (vertices) for the knn vectors
        for(int i = 0; i < vec.rows(); i++) {
            IntNDArrayList ndArrayList = new IntNDArrayList();
            ndArrayList.allocateWithSize(vec.rows());
            knnVec.add(ndArrayList);
        }


        for(int i = 0; i < numWorkers; i++) {
            final int j = i;
            PropagationThread propagationThread = new PropagationThread(i);
            list.add(propagationThread);
            executorService.submit(propagationThread);
        }

        while(!list.isEmpty()) {
            PropagationThread curr = list.removeFirst();
            while(!curr.isDone()) {
                LockSupport.parkNanos(1000L);
            }
        }

    }




    private class PropagationThread implements Runnable {
        private int id;
        private AtomicBoolean isDone = new AtomicBoolean(false);

        public PropagationThread(int id) {
            this.id = id;
        }

        public boolean isDone() {
            return isDone.get();
        }

        @Override
        public void run() {
            log.info("Starting propagation thread " + id);
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
                List<Integer> v1 = oldKnnVec.get(x);
                for(int i = 0; i < v1.size(); i++) {
                    y = v1.get(i);
                    check[y] = x;
                    distances.incrementCount(y, RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y)));
                    if(distances.size() == nNeighbors + 1) {
                        //heap.pop() here which is it?
                        distances.removeKey(distances.argMax());
                    }
                }

                for(int i = 0; i < v1.size(); i++) {
                    List<Integer> v2 = oldKnnVec.get(v1.get(v1.get(i)));
                    for(int j = 0; j < v2.size(); j++) {
                        if(check[y = v2.get(j)] != x) {
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
                    knnVec.get(x).add(distances.argMax());
                    distances.removeKey(distances.argMax());
                }
            }



            isDone.set(true);
            log.info("Finished with propagation thread " + id);
        }
    }


    public int testAccuracy() {
        int testCase = 100;
        Counter<Integer> heap = new Counter<>();
        int hitCase = 0;
        for(int i = 0; i < testCase; i++) {
            int x = MathUtils.randomNumberBetween(0,vec.rows() - 1);
            for(int y = 0; y < vec.rows(); y++) {
                if(x != y) {
                    heap.incrementCount(y,RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y)));
                    if(heap.size() == nNeighbors + 1)  {
                        heap.removeKey(heap.argMax());
                    }
                }
            }

            while(!heap.isEmpty()) {
                int y = heap.argMax();
                IntNDArrayList ndArrayList = knnVec.get(x);
                for (int j = 0; j <ndArrayList.size(); j++) {
                    if (knnVec.get(x).get(j) == y) {
                        hitCase++;
                    }
                }


                heap.removeKey(heap.argMax());
            }
        }

        return hitCase;
    }



    public int sampleAnEdge(double rand1,double rand2) {
        int k = (int)( (nEdges - 0.1) * rand1);
        return rand2 <= prob.getDouble(k) ? k : alias[k];
    }



    private class SearchReverseThread implements Runnable {
        private AtomicBoolean done = new AtomicBoolean(false);
        private int id;

        public SearchReverseThread(int id) {
            this.id = id;
        }

        public boolean isDone() {
            return done.get();
        }

        @Override
        public void run() {
            log.info("Starting search reverse thread " + id);
            int low = id * nVertices / numWorkers;
            int high = (id + 1) * nVertices / numWorkers;
            for(int x = low; x < high; x++) {
                for(int p = head.get(x); p >= 0; p =  next.get(p)) {
                    int y = edgeTo.get(p);
                    for(int q = head.get(y); q >= 0; q = next.get(q)) {
                        if(edgeTo.get(q) == x) {
                            break;
                        }

                        reverse.set(p,q);
                    }
                }
            }

            done.set(true);
            log.info("Finishing Search reverse thread " + id);
        }
    }





    public void constructKnn() {
        if(normalize) {
            NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
            normalizerStandardize.fit(new DataSet(vec,vec));
            normalizerStandardize.transform(vec);
        }

        rpTree = new RPForest(numTrees,maxSize,distanceFunction);
        rpTree.fit(vec);

        runPropagation();
        testAccuracy();
        computeSimilarity();
    }

    public void computeSimilarity() {
        head = new IntNDArrayList();
        for(int i = 0; i < vec.rows(); i++) {
            head.add(-1);
        }

        for(int x = 0; x < vec.rows(); x++) {
            for(int  i = 0; i < knnVec.get(x).size(); i++) {
                edgeFrom.add(x);
                int y = knnVec.get(x).get(i);
                edgeTo.add(y);
                edgeWeight.add(RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y)));
                next.add(head.get(x));
                reverse.add(-1);
                head.set(x,nEdges++);
            }
        }

        LinkedList<ComputeSimilarityThread> computeSimilarityThreads = new LinkedList<>();
        for(int i = 0; i < numWorkers; i++) {
            ComputeSimilarityThread computeSimilarityThread = new ComputeSimilarityThread(i);
            computeSimilarityThreads.addAll(computeSimilarityThreads);
            executorService.submit(computeSimilarityThread);
        }

        while(!computeSimilarityThreads.isEmpty()) {
            ComputeSimilarityThread computeSimilarityThread = computeSimilarityThreads.removeFirst();
            while(!computeSimilarityThread.isDone()) {
                LockSupport.parkNanos(1000L);
            }
        }


        LinkedList<SearchReverseThread> searchReverseThreads = new LinkedList<>();

        for(int i = 0; i < numWorkers; i++) {
            SearchReverseThread searchReverseThread = new SearchReverseThread(i);
            searchReverseThreads.add(searchReverseThread);
            executorService.submit(searchReverseThread);
        }


        while(!searchReverseThreads.isEmpty()) {
            SearchReverseThread searchReverseThread = searchReverseThreads.removeFirst();
            while(!searchReverseThread.isDone()) {
                LockSupport.parkNanos(1000L);
            }
        }

        double sumWeight = 0.0;
        for(int x = 0; x < vec.rows(); x++) {
            for(int p = head.get(x); p >= 0; p = next.get(p)) {
                int y = edgeTo.get(p);
                int q = reverse.get(p);
                if(q == -1) {
                    edgeFrom.add(y);
                    edgeTo.add(x);
                    edgeWeight.add(0.0);
                    next.add(head.get(y));
                    reverse.add(p);
                    q = nEdges++;
                    reverse.set(p,q);
                    head.set(y,nEdges++);
                }

                if(x > y) {
                    double pWeight = p >= edgeWeight.size() ? 0.0 : edgeWeight.get(p);
                    double qWeight =  q >= edgeWeight.size() ? 0.0 : edgeWeight.get(q);
                    while(edgeWeight.size() < p + 1)
                        edgeWeight.add(0.0);
                    while(edgeWeight.size() < q + 1)
                        edgeWeight.add(0.0);
                    sumWeight += (pWeight + qWeight);
                    double assign = (pWeight + qWeight) / 2;
                    edgeWeight.set(p,assign);
                    edgeWeight.set(q,assign);
                }
            }
        }



    }


    private class ComputeSimilarityThread implements Runnable {
        private int id;
        private AtomicBoolean isDone = new AtomicBoolean(false);

        public ComputeSimilarityThread(int id) {
            this.id = id;
        }

        public boolean isDone() {
            return isDone.get();
        }
        @Override
        public void run() {
            log.info("Starting compute similarity thread " + id);
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
                    for(int p = head.get(x); p >= 0; p = next.get(p)) {
                        sumWeight  += tmp = Math.exp(- beta * edgeWeight.get(p));
                        H += beta * (edgeWeight.get(p) * tmp);
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
                    for(int p = head.get(x); p >= 0; p = next.get(p)) {
                        double newValue = Math.exp(-beta * edgeWeight.get(p));
                        sumWeight +=  newValue;
                        edgeWeight.set(p,newValue);
                    }


                    edgeWeight.array().divi(sumWeight);


                }
            }

            isDone.set(true);
            log.info("Finishing compute similarity thread " + id);
        }


    }


    private class VisualizeThread implements Runnable {
        private int id;
        private AtomicBoolean done = new AtomicBoolean(false);

        public VisualizeThread(int id) {
            this.id = id;
        }

        public boolean isDone() {
            return done.get();
        }

        @Override
        public void run() {
            log.info("Starting visualize thread " + id);
            double f = initialAlpha;
            double g = initialAlpha;
            INDArray gg;
            double currAlpha = initialAlpha;
            INDArray curr = Nd4j.create(outDim);
            INDArray err = Nd4j.create(outDim);
            double gradClip = 5.0;
            int edgeCount = 0;
            int lastEdgeCount = 0;
            int p,x,y;
            gradientUpdater = updater.instantiate(err,false);
            while(true) {
                if(edgeCount > nSamples / numWorkers + 2) {
                    break;
                }

                if(edgeCount - lastEdgeCount > 10000) {
                    edgeCountActual += edgeCount - lastEdgeCount;
                    lastEdgeCount = edgeCount;
                    currAlpha = initialAlpha * (1 - edgeCountActual / (nSamples + 1.0));
                }

                p = sampleAnEdge(Nd4j.getRandom().nextGaussian(),Nd4j.getRandom().nextDouble());
                x = edgeFrom.get(p);
                y = edgeTo.get(p);
                //lx = x * out_dim;
                //for (i = 0; i < out_dim; ++i) cur[i] = vis[lx + i], err[i] = 0;
                for(int i = 0; i < nNegatives + 1; i++) {
                    if(i > 0) {
                        y = negTable[(MathUtils.randomNumberBetween(0,negSize - 1))];
                        if (y == edgeTo.get(p)) continue;
                        f = RPUtils.computeDistance(distanceFunction,curr.slice(y),vis.slice(y));
                        if(i == 0) {
                            g = (-2 / (1 + f));
                        }
                        else {
                            g = 2 * gamma / (1 + f) / (0.1 + f);
                        }


                        /**
                         *
                         *
                         * 	gg = g * (cur[j] - vis[ly + j]);
                         err[j] += gg * cur_alpha;

                         gg = g * (vis[ly + j] - cur[j]);
                         vis[ly + j] += gg * cur_alpha;

                         */

                        //double check this
                        double currLr = updater.getLearningRate(0,i);

                        gg = curr.sub(vis.slice(y)).mul(g * currLr);
                        err.addi(gg);

                        gg = vis.slice(y).sub(curr);
                        vis.slice(y).addi(gg.mul(currLr));
                    }

                }

                vis.slice(x).addi(err);
                edgeCount++;
            }

            done.set(true);
            log.info("Finishing visualize thread " + id);
        }
    }


    public void visualize() {
        vis = Nd4j.create(vec.rows(),outDim);
        initNegTable();
        initAliasTable();
        LinkedList<VisualizeThread> visualizeThreads = new LinkedList<>();
        for(int i = 0; i < numWorkers; i++) {
            VisualizeThread visualizeThread = new VisualizeThread(i);
            visualizeThreads.add(visualizeThread);
            executorService.submit(visualizeThread);
        }

        while(!visualizeThreads.isEmpty()) {
            VisualizeThread next = visualizeThreads.removeFirst();
            while(!next.isDone()) {
                LockSupport.parkNanos(1000L);
            }
        }

    }


    public void fit() {
        constructKnn();
        visualize();
        log.info("Finished LargeVis training");
    }
}
