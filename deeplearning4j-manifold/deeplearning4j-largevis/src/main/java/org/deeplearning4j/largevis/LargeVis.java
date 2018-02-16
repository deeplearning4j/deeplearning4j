package org.deeplearning4j.largevis;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.clustering.randomprojection.RPForest;
import org.deeplearning4j.clustering.randomprojection.RPUtils;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.accum.Norm2;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldSubOp;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
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
public class LargeVis {

    private RPForest rpTree;
    @Builder.Default
    private int numWorkers = Runtime.getRuntime().availableProcessors();
    //vec.rows -> nVertices
    private INDArray vec,vis,prob;
    @Builder.Default
    private IUpdater updater = new Sgd(0.01);
    private WeightInitScheme weightInitScheme;
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
    private IntMatrixNDArrayList knnVec,oldKnnVec;
    private int[] negTable;
    @Builder.Default
    private IntNDArrayList head = new IntNDArrayList();
    @Builder.Default
    private int[] alias;
    @Builder.Default
    private IntNDArrayList next = new IntNDArrayList();
    private int maxSize;
    @Builder.Default
    private String distanceFunction = "euclidean";
    private FloatNDArrayList edgeWeight;
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
    private Boolean normalize;
    @Builder.Default
    private int iterationCount = 200;
    private int negSize = (int) 1e8;
    @Builder.Default
    private double gradClipValue = 5.0;
    @Builder.Default
    private GradientNormalization gradientNormalization = GradientNormalization.ClipElementWiseAbsoluteValue;
    private AtomicInteger edgeCountActual = new AtomicInteger(0);
    private  MemoryWorkspace workspace;
    private AtomicInteger updateCount = new AtomicInteger(0);
    private AtomicInteger epochCount = new AtomicInteger(0);
    private IntNDArrayList edgeFrom = new IntNDArrayList();
    private IntNDArrayList edgeTo = new IntNDArrayList();
    private ExecutorService executorService;
    protected final AtomicInteger workerCounter = new AtomicInteger(0);


    private ThreadLocal<INDArray> errors = new ThreadLocal<>();
    private ThreadLocal<INDArray> grads = new ThreadLocal<>();
    private ThreadLocal<INDArray> gradsFirstRow = new ThreadLocal<>();
    private ThreadLocal<INDArray> gradsSecondRow = new ThreadLocal<>();


    private ThreadLocal<Norm2> norm2 = new ThreadLocal<>();
    private ThreadLocal<ClipByValue> clip = new ThreadLocal<>();
    private ThreadLocal<OldSubOp> visXMinusVisY = new ThreadLocal<>();
    private ThreadLocal<OldSubOp> visYMinusVisX = new ThreadLocal<>();
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
                    Boolean normalize,
                    int iterationCount,
                    IUpdater updater,
                    WeightInitScheme weightInitScheme,
                    GradientNormalization gradientNormalization,
                    double gradClipValue,
                    WorkspaceMode workspaceMode,
                    int numWorkers) {

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


        if(iterationCount > 0) {
            this.iterationCount = iterationCount;
        }

        if(gradientNormalization != null) {
            this.gradientNormalization = gradientNormalization;
        }

        if(gradClipValue > 0) {
            this.gradClipValue = gradClipValue;
        }

        if(normalize != null) {
            this.normalize = normalize;
        }

        if(updater != null)
            this.updater = updater;
        this.normalize = normalize;
        this.vec = vec;

        if(weightInitScheme != null) {
            this.weightInitScheme = weightInitScheme;
        }

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



        edgeWeight = new FloatNDArrayList();

        this.executorService = Executors.newFixedThreadPool(numWorkers, new ThreadFactory() {
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

        // opening workspace
        MemoryWorkspace workspace = getWorkspace();

        workspace.notifyScopeEntered();
        Nd4j.getRandom().setSeed(seed);
        knnVec = new IntMatrixNDArrayList();
        //pre allocate up to vec.rows() (vertices) for the knn vectors
        for(int i = 0; i < vec.rows(); i++) {
            IntNDArrayList ndArrayList = new IntNDArrayList();
            ndArrayList.allocateWithSize(vec.rows());
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
        oldKnnVec = knnVec;
        knnVec = new IntMatrixNDArrayList();
        //pre allocate up to vec.rows() (vertices) for the knn vectors
        for(int i = 0; i < vec.rows(); i++) {
            IntNDArrayList ndArrayList = new IntNDArrayList();
            ndArrayList.allocateWithSize(vec.rows());
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
            PriorityQueue<Pair<Integer, Double>> distancesPq = distances.asReversedPriorityQueue();
            for(int x = low; x < hi; x++) {
                check[x] = x;
                List<Integer> v1 = oldKnnVec.get(x);
                for(int i = 0; i < v1.size(); i++) {
                    y = v1.get(i);
                    check[y] = x;
                    double yDistance = RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y),scalars.get());
                    distancesPq.add(Pair.of(y,yDistance));
                    if(distancesPq.size() == nNeighbors + 1) {
                        //heap.pop() here which is it?
                        distancesPq.poll();
                    }
                }

                for(int i = 0; i < v1.size(); i++) {
                    List<Integer> v2 = oldKnnVec.get(v1.get(v1.get(i)));
                    for(int j = 0; j < v2.size(); j++) {
                        if(check[y = v2.get(j)] != x) {
                            check[y] = x;
                            double yDistance =  RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y),scalars.get());
                            distancesPq.add(Pair.of(y,yDistance));
                        }
                    }

                    double yDistance = (RPUtils.computeDistance(distanceFunction,vec.slice(x),vec.slice(y),scalars.get()));
                    distancesPq.add(Pair.of(y,yDistance));
                    if(distancesPq.size() == nNeighbors + 1) {
                        //heap.pop() here which is it?
                        distancesPq.poll();
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
        PriorityQueue<Pair<Integer, Double>> heapPq = heap.asReversedPriorityQueue();
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
                IntNDArrayList ndArrayList = knnVec.get(x);
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



    public int sampleAnEdge(double rand1,double rand2) {
        int k = MathUtils.randomNumberBetween(0,nEdges - 1,Nd4j.getRandom().nextDouble());
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

        rpTree = new RPForest(numTrees,maxSize,distanceFunction);
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

    /**
     * Return the gradients wrt the distances of x and y
     * relative to each other
     * @param x the slice of vis to take the gradient of
     * @param y the slice of y to take the gradient of
     * @param i the current iteration
     * @param currLr the current learning rate
     * @return the gradients wrt x and y (in that order)
     */
    public INDArray gradientsFor(int x,int y,int i,double currLr,boolean normalize) {
        INDArray visY = vis.slice(y);
        INDArray visX = vis.slice(x);
        return gradientsFor(visX,y,i,currLr,normalize);
    }


    /**
     * Return the gradients wrt the distances of x and y
     * relative to each other
     * @param visX the slice of vis to take the gradient of
     * @param visY the slice of y to take the gradient of
     * @param i the current iteration
     * @param currLr the current learning rate
     * @param normalize whether to normalize the gradient or not
     * @return the gradients wrt x and y (in that order)
     */
    public INDArray gradientsFor(INDArray grads,INDArray visX,INDArray visY,int i,double currLr,boolean normalize) {
        MemoryWorkspace workspace = getWorkspace();
        try(MemoryWorkspace w2 = workspace.notifyScopeEntered()) {

            if (scalars.get() == null)
                scalars.set(Nd4j.scalar(0.0));

            double g;
            double f = RPUtils.computeDistance(distanceFunction,visX,visY,scalars.get());
            if(i == 0) {
                g = (-2 / (1 + f));
            }
            else {
                g = 2 * gamma / (1 + f) / (0.1 + f);
            }

            //gradient wrt distance to x and y
            Nd4j.getExecutioner().execAndReturn(getVisXMinusVisY(visX,visY,createGradFirstRow().muli(g * currLr)));
            Nd4j.getExecutioner().execAndReturn(getVisYMinusVisX(visY,visX,createGradSecondRow()));
            if(normalize) {
                normalizeBatch(grads);
            }


            return grads;
        }


    }

    /**
     * Return the gradients wrt the distances of x and y
     * relative to each other
     * @param visX the slice of vis to take the gradient of
     *  @param y the initial y for sampling
     * @param i the current iteration
     * @param currLr the current learning rate
     * @param normalize whether to normalize the gradient or not
     * @return the gradients wrt x and y (in that order)
     */
    public INDArray gradientsFor(INDArray visX,int y,int i,double currLr,boolean normalize) {
        return gradientsFor(Nd4j.create(2,visX.columns()),visX,vis.slice(y),i,currLr,normalize);
    }


    /**
     * Compute the error wrt the given parameters given a sampled edge(p)
     * and 2 vectors to compute the distance and error for
     *
     * Note that there is a side effect of updating y as a part of calling this method.
     * This method should mainly be used in gradient checking tests or internally within this class
     * not directly by a user.
     *
     * @param visX the x component to compute the error for
     * @param y the index of y
     * @param p the sample edge for random access
     * @param currLr the current learning rate for the gradient update
     * @param updateY
     * @return the error wrt the given parameters
     */
    public INDArray errorWrt(INDArray visX,  int y, int p, double currLr, boolean updateY,boolean normalize) {
        MemoryWorkspace workspace = getWorkspace();
        if(negTable == null) {
            initNegTable();
        }

        try(MemoryWorkspace w2 = workspace.notifyScopeEntered()) {
            INDArray err = createError();
            INDArray grads = createGrad();
            for(int i = 0; i < nNegatives + 1; i++) {
                if(y > 0) {
                    y = negTable[(MathUtils.randomNumberBetween(0, negSize - 1,Nd4j.getRandom().nextDouble()))];
                    if (y == edgeTo.get(p)) continue;
                }

                INDArray visY = vis.slice(y);
                //get the gradient wrt x and y
                gradientsFor(grads,visX,visY,i,currLr,normalize);
                INDArray gradsFirst = createGradFirstRow();
                INDArray gradsSecond = createGradSecondRow();
                err.addi(gradsFirst);
                if(updateY)
                    visY.addi(gradsSecond);


            }

            return err;

        }

    }


    public INDArray createGradSecondRow() {
        if(gradsSecondRow.get() == null) {
            gradsSecondRow.set(grads.get().slice(1));
        }

        return gradsSecondRow.get();
    }


    public OldSubOp getVisYMinusVisX(INDArray x,INDArray y,INDArray result) {
        return getSubOp(visYMinusVisX,x,y,result);

    }


    public OldSubOp getVisXMinusVisY(INDArray x,INDArray y,INDArray result) {
        return getSubOp(visXMinusVisY,x,y,result);
    }

    private OldSubOp getSubOp(ThreadLocal<OldSubOp> threadLocal,INDArray x,INDArray y,INDArray result) {
        if(threadLocal.get() == null) {
            OldSubOp clipByValue = new OldSubOp(x,y,result);
            threadLocal.set(clipByValue);
            return clipByValue;
        }

        OldSubOp clipByValue = threadLocal.get();
        clipByValue.setX(x);
        clipByValue.setY(y);
        return clipByValue;
    }


    public ClipByValue getClipByValue(INDArray input) {
        if(clip.get() == null) {
            ClipByValue clipByValue = new ClipByValue(new INDArray[] {input},null,-gradClipValue,gradClipValue,true);
            clip.set(clipByValue);
            return clipByValue;
        }

        ClipByValue clipByValue = clip.get();
        clipByValue.setInputArgument(0,input);
        return clipByValue;
    }


    public Norm2 getNorm2(INDArray input) {
        if(clip.get() == null) {
            Norm2 norm2 = new Norm2(input);
            this.norm2.set(norm2);
            return norm2;
        }

        Norm2 ret = norm2.get();
        ret.setX(input);
        return ret;
    }


    public INDArray createGradFirstRow() {
        if(gradsFirstRow.get() == null) {
            gradsFirstRow.set(grads.get().slice(0));
        }

        return gradsFirstRow.get();
    }

    public INDArray createGrad() {
        if(grads.get() == null) {
            grads.set(Nd4j.create(2,vis.columns()));
        }

        return grads.get();
    }

    public INDArray createError() {
        if(errors.get() == null) {
            errors.set(Nd4j.create(outDim));
        }

        return errors.get();
    }


    /**
     * Compute the error wrt the given parameters given a sampled edge(p)
     * and 2 vectors to compute the distance and error for
     *
     * Note that there is a side effect of updating y as a part of calling this method.
     * This method should mainly be used in gradient checking tests or internally within this class
     * not directly by a user.
     *
     * @param x the x component to compute the error for
     * @param y the initial index of y (changes with various negative sampling
     * @param p the sampled edge for random access
     * @param currLr the current learning rate for the gradient update
     * @param updateY whether to update y or not
     * @param normalize whether to apply gradient normalization or not
     * @return the error wrt the given parameters
     */
    public INDArray errorWrt(int x, int y, int p, double currLr, boolean updateY,boolean normalize) {
        return errorWrt(vis.slice(x),y,p,currLr,updateY,normalize);

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
            int edgeCount = 0;
            int lastEdgeCount = 0;
            int p,x,y;
            // opening workspace
            MemoryWorkspace workspace = getWorkspace();
            try(MemoryWorkspace workspace2 = workspace.notifyScopeEntered()) {
                double currLr = initialAlpha;
                while (true) {
                    if (edgeCount > nSamples / numWorkers + 2) {
                        break;
                    }

                    if (edgeCount - lastEdgeCount > 10000) {
                        int progress = edgeCountActual.addAndGet(edgeCount - lastEdgeCount);
                        lastEdgeCount = edgeCount;
                        currLr = updater.getLearningRate(updateCount.get(), epochCount.get());
                        epochCount.getAndIncrement();
                        //(real)edge_count_actual / (real)(n_samples + 1) * 100
                        double progress2 = (double) progress / (double) (nSamples + 1) * 100;
                        log.info("Progress " + String.valueOf(progress2));

                    }

                    p = sampleAnEdge(Nd4j.getRandom().nextGaussian(), Nd4j.getRandom().nextDouble());
                    x = edgeFrom.get(p);
                    y = edgeTo.get(p);
                    INDArray err = errorWrt(x, y, p, currLr, true, true);
                    //update the error for the given vector
                    INDArray visX = vis.slice(x);
                    visX.addi(err);
                    edgeCount++;
                    updateCount.getAndIncrement();

                }

            }
            done.set(true);
        }
    }


    /**
     * Save the model as a file with a csv format, adding the label as the last column.
     * @param labels
     * @param path the path to write
     * @throws IOException
     */
    public void saveAsFile(List<String> labels, String path) throws IOException {
        BufferedWriter write = null;
        try {
            write = new BufferedWriter(new FileWriter(new File(path)));
            for (int i = 0; i < vis.rows(); i++) {
                if (i >= labels.size())
                    break;
                String word = labels.get(i);
                if (word == null)
                    continue;
                StringBuilder sb = new StringBuilder();
                INDArray wordVector = vis.getRow(i);
                for (int j = 0; j < wordVector.length(); j++) {
                    sb.append(wordVector.getDouble(j));
                    if (j < wordVector.length() - 1)
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
        } finally {
            if (write != null)
                write.close();
        }
    }

    private void normalizeBatch(INDArray input) {
        switch (gradientNormalization) {
            case None: break;
            case RenormalizeL2PerParamType:
            case RenormalizeL2PerLayer:
                Norm2 norm2 = getNorm2(input);
                Nd4j.getExecutioner().exec(norm2,1);
                input.diviRowVector(norm2.z());
                break;
            case ClipElementWiseAbsoluteValue:
                ClipByValue clipByValue = getClipByValue(input);
                Nd4j.getExecutioner().exec(clipByValue);
                break;
            case ClipL2PerLayer:
                throw new UnsupportedOperationException("Clip l2 per layer not supported");
            case ClipL2PerParamType:
                throw new UnsupportedOperationException("Clip l2 per layer not supported");
            default:
                throw new RuntimeException(
                        "Unknown (or not implemented) gradient normalization strategy: " + gradientNormalization);
        }
    }


    private void normalize(INDArray input) {
        switch (gradientNormalization) {
            case None: break;
            case RenormalizeL2PerLayer:
                double l2 = input.norm2Number().doubleValue();
                input.divi(l2);

                break;
            case RenormalizeL2PerParamType:
                double l22 = Nd4j.getExecutioner().execAndReturn(new Norm2(input)).getFinalResult().doubleValue();
                input.divi(l22);

                break;
            case ClipElementWiseAbsoluteValue:
                ClipByValue clipByValue = new ClipByValue(new INDArray[] {input},null,-gradClipValue,gradClipValue,true);
                Nd4j.getExecutioner().exec(clipByValue);

                break;
            case ClipL2PerLayer:
                throw new UnsupportedOperationException("Clip l2 per layer not supported");
            case ClipL2PerParamType:
                throw new UnsupportedOperationException("Clip l2 per layer not supported");
            default:
                throw new RuntimeException(
                        "Unknown (or not implemented) gradient normalization strategy: " + gradientNormalization);
        }
    }

    /**
     * Init weights
     */
    public void initWeights() {
        if(this.weightInitScheme == null) {
            weightInitScheme = new XavierFanInInitScheme('c',outDim);
        }

        vis = weightInitScheme.create(new int[] {vec.rows(),outDim});

    }


    /**
     * Create the weight matrix for visualization.
     */
    public void visualize() {
        initWeights();
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

    /**
     * Get the result of the training
     * @return
     */
    public INDArray getResult() {
        return vis;
    }


    public void fit() {
        if(nSamples <= 0) {
            if (nVertices < 10000)
                nSamples = 1000;
            else if (nVertices < 1000000)
                nSamples = (nVertices - 10000) * 9000 / (1000000 - 10000) + 1000;
            else nSamples = nVertices / 100;
        }

        nSamples *= 1000000;
        if (nTrees < 0) {
            if (nVertices < 100000)
                nTrees = 10;
            else if (nVertices < 1000000)
                nTrees = 20;
            else if (nVertices < 5000000)
                nTrees = 50;
            else nTrees = 100;
        }

        constructKnn();
        visualize();
        log.info("Finished LargeVis training");
    }
}
