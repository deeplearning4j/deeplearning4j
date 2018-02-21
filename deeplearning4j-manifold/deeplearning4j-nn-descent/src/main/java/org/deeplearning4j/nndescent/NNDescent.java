package org.deeplearning4j.nndescent;

import com.google.common.primitives.Ints;
import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.clustering.randomprojection.RPForest;
import org.deeplearning4j.clustering.randomprojection.RPUtils;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Norm2;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldSubOp;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.MathUtils;
import org.nd4j.list.FloatNDArrayList;
import org.nd4j.list.IntNDArrayList;
import org.nd4j.list.matrix.IntMatrixNDArrayList;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.XavierFanInInitScheme;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;


/**
 * A port of the LargeVis algorithm:
 * https://github.com/lferry007/LargeVis
 *
 * to nd4j. This implementation uses
 * RPTrees rather than annoy as in the original implementation.
 *
 *
 * This algorithm also uses the nd4j updaters (to allow for more flexibility)
 * over static gradient clipping and the simpler learning rate schedule.
 *
 *
 * The algorithm has the following parameters:
 *      -fea: specify whether the input file is high-dimensional feature vectors (1) or networks (0). Default is 1.
 *      -input: Input file of feature vectors or networks
 *      -output: Output file of low-dimensional representations.
 *      -threads: Number of threads. Default is 8.
 *      -outdim: The lower dimensionality LargesVis learns for visualization (usually 2 or 3). Default is 2.
 *      -samples: Number of edge samples for graph layout (in millions). Default is set to data size / 100 (million).
 *      -prop: Number of times for neighbor propagations in the state of K      -NNG construction, usually less than 3. Default is 3.
 *      -alpha: Initial learning rate. Default is 1.0.
 *      -trees: Number of random-projection trees used for constructing K-NNG. 50 is sufficient for most cases.
 *      -neg: Number of negative samples used for negative sampling. Default is 5.
 *      -neigh: Number of neighbors (K) in K-NNG, which is usually set as three times of perplexity. Default is 150.
 *      -gamma: The weights assigned to negative edges. Default is 7.
 *      -perp: The perplexity used for deciding edge weights in K-NNG. Default is 50.
 *
 * @author Adam Gibson
 */
@Data
@Slf4j
public class NNDescent {

    private RPForest rpTree;
    @Builder.Default
    private int numWorkers = Runtime.getRuntime().availableProcessors();
    //vec.rows -> nVertices
    private INDArray vec,prob;
    private ThreadLocal<INDArray> scalars = new ThreadLocal<>();
    @Builder.Default
    private WorkspaceMode workspaceMode = WorkspaceMode.SINGLE;
    private ThreadLocal<MemoryWorkspace>  workspaceThread = new ThreadLocal<>();
    private WorkspaceConfiguration workspaceConfiguration;

    /**
     * KNNVec is a pointer to a vector.
     * This tends to be a list of vectors.
     *
     * Double indexing is actually just a get(i,j)
     * in a matrix.
     */
    private List<List<Integer>> knnVec,oldKnnVec;
    private int[] negTable;
    @Builder.Default
    private IntNDArrayList head = new IntNDArrayList();
    @Builder.Default
    private int[] alias;
    @Builder.Default
    private IntNDArrayList next = new IntNDArrayList();
    @Builder.Default
    private int maxSize = 1000;
    @Builder.Default
    private String distanceFunction = "euclidean";
    private FloatNDArrayList edgeWeight;
    private int nEdges;
    @Builder.Default
    private IntNDArrayList reverse = new IntNDArrayList();
    private ExecutorService threadExec;
    @Builder.Default
    private double initialAlpha = 1.0;
    private int nSamples;
    @Builder.Default
    private int nNegatives = 5;
    @Builder.Default
    private int nNeighbors = 150;
    @Builder.Default
    private int nTrees = 5;
    @Builder.Default
    private int nPropagations = 3;
    @Builder.Default
    private double gamma = 7.0;
    @Builder.Default
    private double perplexity = 50.0;
    private int nVertices;
    @Builder.Default
    private Boolean normalize = true;
    @Builder.Default
    private int iterationCount = 200;
    private int negSize = (int) 1e8;

    private AtomicInteger edgeCountActual = new AtomicInteger(0);
    private  MemoryWorkspace workspace;
    private IntNDArrayList edgeFrom = new IntNDArrayList();
    private IntNDArrayList edgeTo = new IntNDArrayList();
    private ExecutorService executorService;
    protected final AtomicInteger workerCounter = new AtomicInteger(0);

    @Builder.Default
    private Boolean sample = true;



    // log uncaught exceptions
    Thread.UncaughtExceptionHandler handler = new Thread.UncaughtExceptionHandler() {
        public void uncaughtException(Thread th, Throwable ex) {
            log.error("Uncaught exception: " + ex);
            ex.printStackTrace();
        }
    };





    @Builder
    public NNDescent(INDArray vec,
                     int maxSize,
                     String distanceFunction,
                     int nTrees,
                     int nNegatives,
                     double gamma,
                     double initialAlpha,
                     double perplexity,
                     int nPropagations,
                     long seed,
                     int nNeighbors,
                     Boolean normalize,
                     int iterationCount,
                     WorkspaceMode workspaceMode,
                     int numWorkers,
                     int nSamples,
                     Boolean sample,
                     ExecutorService executorService) {

        if(nTrees > 0) {
            this.nTrees = nTrees;
        }

        if(workspaceMode != null) {
            this.workspaceMode = workspaceMode;
        }

        if(numWorkers > 0) {
            this.numWorkers = numWorkers;
        }

        if(workspaceMode != WorkspaceMode.NONE)
            workspaceConfiguration = WorkspaceConfiguration.builder().cyclesBeforeInitialization(1)
                    .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.FIRST_LOOP)
                    .policyMirroring(MirroringPolicy.FULL).policyReset(ResetPolicy.BLOCK_LEFT)
                    .policySpill(SpillPolicy.REALLOCATE).build();

        if(maxSize > 0) {
            this.maxSize  = maxSize;
        }

        if(sample != null) {
            this.sample = sample;
        }

        if(iterationCount > 0) {
            this.iterationCount = iterationCount;
        }


        if(nSamples > 0) {
            this.nSamples = nSamples;
        }

        if(normalize != null) {
            this.normalize = normalize;
        }

        this.vec = vec;

        this.nVertices = vec.rows();
        if(maxSize > 0)
            this.maxSize = maxSize;
        if(distanceFunction != null)
            this.distanceFunction = distanceFunction;
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
        head = new IntNDArrayList();
        for(int i = 0; i < vec.rows(); i++) {
            head.add(-1);
        }



        edgeWeight = new FloatNDArrayList();

        if(executorService == null)
            this.executorService = Executors.newFixedThreadPool(this.numWorkers, new ThreadFactory() {
                @Override
                public Thread newThread( Runnable r) {
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
        else
            this.executorService = executorService;


        // opening workspace
        MemoryWorkspace workspace = getWorkspace();

        workspace.notifyScopeEntered();
        Nd4j.getRandom().setSeed(seed);
        knnVec = new ArrayList<>();
        //pre allocate up to vec.rows() (vertices) for the knn vectors
        for(int i = 0; i < vec.rows(); i++) {
            List<Integer> ndArrayList = new ArrayList<>();
            for(int j = 0; j < vec.rows(); j++) {
                ndArrayList.add(0);
            }
            knnVec.add(ndArrayList);
        }

        workspace.notifyScopeLeft();
        Nd4j.getMemoryManager().togglePeriodicGc(false);
    }

    public MemoryWorkspace getWorkspace() {
        if(this.workspaceThread.get() == null) {
            // opening workspace
            workspaceThread.set(workspaceMode == null || workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace() :
                    Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfiguration, "LargeVisWorkspace-" + Thread.currentThread().getName()));
        }

        return workspaceThread.get();
    }


    /**
     * Initializes the neg table
     */
    public void initNegTable() {
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

    /**
     * Initializes the alias table
     */
    public void initAliasTable() {
        log.info("Initializing alias table");
        alias = new int[nEdges];
        prob = Nd4j.create(1,nEdges);
        INDArray normProb;
        int[] largeBlock = new int[nEdges];
        int[] smallBlock = new int[nEdges];
        double sum;
        int currSmallBlock,currLargeBlock;
        int numSmallBlock = 0;
        int numLargeBlock = 0;
        sum = edgeWeight.array().sumNumber().doubleValue();
        normProb = edgeWeight.array().muli(nEdges / (sum + 1e-12));
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


    /**
     *
     */
    public void runPropagation() {
        LinkedList<PropagationThread> list = new LinkedList<>();
        for(int propagation = 0; propagation < nPropagations; propagation++) {
            oldKnnVec = knnVec;
            knnVec = new ArrayList<>();
            //pre allocate up to vec.rows() (vertices) for the knn vectors
            for(int i = 0; i < vec.rows(); i++) {
                List<Integer> ndArrayList = new ArrayList<>();
                knnVec.add(ndArrayList);
            }


            for(int i = 0; i < numWorkers; i++) {
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
            if (scalars.get() == null)
                scalars.set(Nd4j.scalar(0.0));


            log.info("Starting propagation thread " + id);
            int[] check = new int[vec.rows()];
            int low = id * vec.rows() / numWorkers;
            int hi = (id + 1) * vec.rows() / numWorkers;

            for(int i = 0; i < check.length; i++) {
                check[i] = -1;
            }

            int y = -1;


            Counter<Integer> distances = new Counter<>();
            PriorityQueue<Pair<Integer, Double>> distancesPq = distances.asPriorityQueue();

            INDArray result = null;
            /**
             * Find vectorization if possible
             */
            for(int x = low; x < hi; x++) {
                check[x] = x;
                int[] v1 = Ints.toArray(oldKnnVec.get(x));
                INDArray rows = Nd4j.pullRows(vec,1,v1);
                INDArray curr = vec.slice(x);
                if(result == null) {
                    result = Nd4j.create(v1.length);
                }

                INDArray yDistances = RPUtils.computeDistanceMulti(distanceFunction,curr,rows,result);
                for(int i = 0; i < v1.length; i++) {
                    y = v1[i];
                    check[y] = x;
                    distancesPq.add(Pair.of(y,yDistances.getDouble(i)));
                    if(distancesPq.size() == nNeighbors + 1) {
                        //heap.pop() here which is it?
                        distancesPq.poll();
                    }
                }
         /*       for(int i = 0; i < v1.size(); i++) {
                    y = v1.get(i);
                    check[y] = x;
                    double yDistance = RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y),scalars.get());
                    distancesPq.add(Pair.of(y,yDistance));
                    if(distancesPq.size() == nNeighbors + 1) {
                        //heap.pop() here which is it?
                        distancesPq.poll();
                    }
                }*/

                for(int i = 0; i < v1.length; i++) {
                    int[] v2 = Ints.toArray(oldKnnVec.get(v1[i]));
                    rows = Nd4j.pullRows(vec,1,v2);

                    yDistances = RPUtils.computeDistanceMulti(distanceFunction,curr,rows,result);
                    for(int j = 0; j < v2.length; j++) {
                        if(check[y = v2[j]] != x) {
                            check[y] = x;
                            distancesPq.add(Pair.of(y,yDistances.getDouble(j)));
                        }

                        if(distancesPq.size() == nNeighbors + 1) {
                            distancesPq.poll();
                        }
                    }

                }

                while(!distancesPq.isEmpty()) {
                    knnVec.get(x).add(distancesPq.peek().getFirst());
                    distancesPq.poll();
                }
            }



            isDone.set(true);
            log.info("Finished with propagation thread " + id);
        }
    }


    /**
     * Test accuracy of the parameters.
     * @return
     */
    public int testAccuracy() {
        if (scalars.get() == null)
            scalars.set(Nd4j.scalar(0.0));


        int testCase = 100;
        Counter<Integer> heap = new Counter<>();
        PriorityQueue<Pair<Integer, Double>> heapPq = heap.asPriorityQueue();
        int hitCase = 0;
        for(int i = 0; i < testCase; i++) {
            int x = MathUtils.randomNumberBetween(0,vec.rows() - 1,Nd4j.getRandom().nextDouble());
            for(int y = 0; y < vec.rows(); y++) {
                if(x != y) {
                    double yDistance =  RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y),scalars.get());
                    heapPq.add(Pair.of(y,yDistance));
                    if(heap.size() == nNeighbors + 1)  {
                        heapPq.poll();
                    }
                }
            }

            while(!heapPq.isEmpty()) {
                int y = heapPq.peek().getFirst();
                List<Integer> ndArrayList = knnVec.get(x);
                for (int j = 0; j <ndArrayList.size(); j++) {
                    if (knnVec.get(x).get(j) == y) {
                        hitCase++;
                    }
                }


                heapPq.poll();
            }
        }

        return hitCase;
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


    /**
     * Construct the k-nearest-neighbors tree including:
     * Building and running the {@link RPForest} algorithm
     *
     */

    public void constructKnn() {
        if(normalize) {
            NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
            normalizerStandardize.fit(new DataSet(vec,vec));
            normalizerStandardize.transform(vec);
        }

        rpTree = new RPForest(nTrees,maxSize,distanceFunction);
        rpTree.fit(vec);

        runPropagation();
        testAccuracy();
        computeSimilarity();
    }

    /**
     * Compute all the similarities
     * for each data point
     */
    public void computeSimilarity() {
        if (scalars.get() == null)
            scalars.set(Nd4j.scalar(0.0));

        head = new IntNDArrayList();
        for(int i = 0; i < vec.rows(); i++) {
            head.add(-1);
        }

        for(int x = 0; x < vec.rows(); x++) {
            for(int  i = 0; i < knnVec.get(x).size(); i++) {
                edgeFrom.add(x);
                int y = knnVec.get(x).get(i);
                edgeTo.add(y);
                float dist = (float) RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y),scalars.get());
                edgeWeight.add(dist);
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

        for(int x = 0; x < vec.rows(); x++) {
            for(int p = head.get(x); p >= 0; p = next.get(p)) {
                int y = edgeTo.get(p);
                int q = reverse.get(p);
                if(q == -1) {
                    edgeFrom.add(y);
                    edgeTo.add(x);
                    edgeWeight.add(1e-12f);
                    next.add(head.get(y));
                    reverse.add(p);
                    q = nEdges++;
                    reverse.set(p,q);
                    head.set(y,nEdges++);
                }

                if(x > y) {
                    float pWeight = p >= edgeWeight.size() ? 0.0f : edgeWeight.get(p);
                    float qWeight =  q >= edgeWeight.size() ? 0.0f : edgeWeight.get(q);
                    while(edgeWeight.size() < p + 1)
                        edgeWeight.add(0.0f);
                    while(edgeWeight.size() < q + 1)
                        edgeWeight.add(0.0f);
                    float assign = (pWeight + qWeight) / 2;

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
            double loBeta;
            double hiBeta;
            int H;
            double tmp;
            float sumWeight;

            for(int x = low; x < high; x++) {
                double beta = 1.0;
                loBeta = hiBeta = -1;
                for(int iter = 0; iter < iterationCount; iter++) {
                    H = 0;
                    sumWeight = Float.MIN_VALUE;
                    for(int p = head.get(x); p >= 0; p = next.get(p)) {
                        float currEdgeWeight = edgeWeight.get(p);
                        sumWeight  += tmp = Math.exp(- beta * currEdgeWeight);
                        H += beta * (currEdgeWeight * tmp);
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

                        if(beta > Float.MAX_VALUE) {
                            beta = (int) Float.MAX_VALUE;
                        }
                    }
                    else {
                        hiBeta = beta;
                        if(loBeta < 0) {
                            beta /= 2;
                        }
                        else {
                            beta = (loBeta + beta) / 2;
                        }
                    }

                    if(beta > Float.MAX_VALUE) {
                        beta = Float.MAX_VALUE;
                    }

                }


                sumWeight = Float.MIN_VALUE;
                for(int p = head.get(x); p >= 0; p = next.get(p)) {
                    float newValue = (float) Math.exp(-beta * edgeWeight.get(p));
                    sumWeight +=  newValue;
                }

                for (int p = head.get(x); p >= 0; p = next.get(p)) {
                    edgeWeight.set(p,edgeWeight.get(p) / sumWeight);
                }

            }

            isDone.set(true);
            log.info("Finishing compute similarity thread " + id);
        }


    }



    public void fit() {
        if(sample) {
            if (nSamples <= 0) {
                if (nVertices < 10000)
                    nSamples = 1000;
                else if (nVertices < 1000000)
                    nSamples = (nVertices - 10000) * 9000 / (1000000 - 10000) + 1000;
                else nSamples = nVertices / 100;
            }

            nSamples *= 1000000;
        }
        else {
            nSamples = 0;
        }


        if (nTrees <= 0) {
            if (nVertices < 100000)
                nTrees = 10;
            else if (nVertices < 1000000)
                nTrees = 20;
            else if (nVertices < 5000000)
                nTrees = 50;
            else nTrees = 100;
        }

        constructKnn();
        initNegTable();
        initAliasTable();
        log.info("Finished LargeVis training");
    }
}
