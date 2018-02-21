package org.deeplearning4j.umap;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.clustering.randomprojection.RPUtils;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nndescent.NNDescent;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Norm2;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldSubOp;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.util.MathUtils;
import org.nd4j.list.IntNDArrayList;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.XavierFanInInitScheme;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
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
public class UMap {


    private NNDescent nnDescent;
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

    @Builder.Default
    private String distanceFunction = "euclidean";
    private int nEdges;
    @Builder.Default
    private IntNDArrayList reverse = new IntNDArrayList();
    private ExecutorService threadExec;
    @Builder.Default
    private int outDim = 2;
    @Builder.Default
    private double initialAlpha = 1.0;
    @Builder.Default
    private int nNegatives = 5;
    @Builder.Default
    private double gamma = 7.0;
    @Builder.Default
    private double perplexity = 50.0;
    @Builder.Default
    private long seed = 42;
    @Builder.Default
    private Boolean normalize;
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

    @Builder.Default
    private Boolean sample = true;

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
    public UMap(INDArray vec,
                int maxSize,
                String distanceFunction,
                int numTrees,
                int outDims,
                int nNegatives,
                double gamma,
                double initialAlpha,
                double perplexity,
                int nPropagations,
                long seed,
                int nNeighbors,
                Boolean normalize,
                int iterationCount,
                IUpdater updater,
                int nTrees,
                WeightInitScheme weightInitScheme,
                GradientNormalization gradientNormalization,
                double gradClipValue,
                WorkspaceMode workspaceMode,
                int numWorkers,
                int nSamples,
                Boolean sample) {



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


        if(sample != null) {
            this.sample = sample;
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


        if(distanceFunction != null)
            this.distanceFunction = distanceFunction;
        if(outDims > 0)
            this.outDim = outDims;
        if(initialAlpha > 0)
            this.initialAlpha = initialAlpha;

        if(nNegatives > 0)
            this.nNegatives = nNegatives;
        if(gamma > 0)
            this.gamma = gamma;
        if(perplexity > 0)
            this.perplexity = perplexity;
        if(seed > 0)
            this.seed = seed;
        if(outDims > 0)
            this.outDim = outDims;


        vis = Nd4j.create(vec.rows(),outDim);





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

        this.nnDescent = NNDescent.builder()
                .distanceFunction(distanceFunction)
                .gamma(gamma)
                .initialAlpha(initialAlpha)
                .maxSize(maxSize)
                .iterationCount(iterationCount)
                .nNegatives(nNegatives)
                .normalize(normalize)
                .numTrees(nTrees)
                .nPropagations(nPropagations)
                .numWorkers(numWorkers)
                .perplexity(perplexity)
                .nSamples(nSamples)
                .sample(sample)
                .numTrees(numTrees)
                .nNeighbors(nNeighbors)
                .seed(seed)
                .workspaceMode(workspaceMode)
                .executorService(executorService)
                .build();
        nnDescent.fit();

        // opening workspace
        MemoryWorkspace workspace = nnDescent.getWorkspace();

        workspace.notifyScopeEntered();
        Nd4j.getRandom().setSeed(seed);

        workspace.notifyScopeLeft();
        Nd4j.getMemoryManager().togglePeriodicGc(false);
    }

    /**
     * Sample a random edge based on the 2
     * random numbers passed in.
     * The first number is used to compute a k
     * relative to the number of edges.
     * Depending on the probability of k
     * it returns the k or its alias.
     * @param rand1 the first number
     * @param rand2 the second number
     * @return the sampled edge
     */

    public int sampleAnEdge(double rand1,double rand2) {
        int k = (int) ((nEdges - 0.1) * rand1);
        return rand2 <= prob.getDouble(k) ? k : nnDescent.getAlias()[k];
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
        MemoryWorkspace workspace = nnDescent.getWorkspace();
        try(MemoryWorkspace w2 = workspace.notifyScopeEntered()) {

            if (scalars.get() == null)
                scalars.set(Nd4j.scalar(0.0));

            double g = grad(i,visX,visY);

            //gradient wrt distance to x and y
            Nd4j.getExecutioner().execAndReturn(getVisXMinusVisY(visX,visY,createGradFirstRow().muli(g * currLr)));
            Nd4j.getExecutioner().execAndReturn(getVisYMinusVisX(visY,visX,createGradSecondRow()));
            if(normalize) {
                normalizeBatch(grads);
            }


            return grads;
        }


    }

    public double grad(int i,INDArray visX,INDArray visY) {
        double g;
        if (scalars.get() == null)
            scalars.set(Nd4j.scalar(0.0));

        double f = RPUtils.computeDistance(distanceFunction,visX,visY,scalars.get());
        if(i == 0) {
            g = (-2 / (1 + f));
        }
        else {
            g = 2 * gamma / (1 + f) / (0.1 + f);
        }
        return g;
    }


    public double grad(int i,int x,int y) {
        return grad(i,vis.slice(x),vis.slice(y));
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
        MemoryWorkspace workspace = nnDescent.getWorkspace();
        int[] negTable = nnDescent.getNegTable();

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
            if(grads.get() == null) {
                INDArray grads = createGrad();
                gradsSecondRow.set(grads.slice(1));
            }
            else
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
            if(grads.get() == null) {
                INDArray grads = createGrad();
                gradsFirstRow.set(grads.slice(0));
            }
            else
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



    public double distance(int x,int y) {
        return RPUtils.computeDistance(distanceFunction,vis.slice(x),vis.slice(y));
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
            MemoryWorkspace workspace = nnDescent.getWorkspace();
            try(MemoryWorkspace workspace2 = workspace.notifyScopeEntered()) {
                double currLr = initialAlpha;
                while (true) {
                    if (edgeCount > nnDescent.getNSamples() / numWorkers + 2) {
                        break;
                    }

                    if (edgeCount - lastEdgeCount > 10000) {
                        int progress = edgeCountActual.addAndGet(edgeCount - lastEdgeCount);
                        lastEdgeCount = edgeCount;
                        currLr = updater.getLearningRate(updateCount.get(), epochCount.get());
                        epochCount.getAndIncrement();
                        double progress2 = (double) progress / (double) (nnDescent.getNSamples() + 1) * 100;
                        log.info("Progress " + String.valueOf(progress2));

                    }

                    p = sampleAnEdge(Nd4j.getRandom().nextDouble(), Nd4j.getRandom().nextDouble());
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
        visualize();
        log.info("Finished LargeVis training");
    }
}
