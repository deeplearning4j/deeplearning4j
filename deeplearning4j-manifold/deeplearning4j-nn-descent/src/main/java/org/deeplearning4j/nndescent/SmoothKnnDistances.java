package org.deeplearning4j.nndescent;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.GreaterThan;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;


/**
 * A port of the smooth knn algorithm to java
 * based on the work done with umap:
 * https://github.com/lmcinnes/umap/blob/f0c8761ef4be9bacf3976faf3f9a048d2a09f9ed/umap/umap_.py#L508
 *
 * Smoothed knn allows a continuous representation of knn over the discretized version
 * allowing for a smoother distribution of nearest neighbors for distance calculations.
 *
 * @author Adam Gibson
 */
@Slf4j
@Builder
public class SmoothKnnDistances {
    public final static double SMOOTH_K_TOLERANCE = 1e-5;
    public final static double MIN_K_DIST_SCALE = 1e-3;

    private ExecutorService executorService;
    @Builder.Default
    private int numWorkers = Runtime.getRuntime().availableProcessors();
    private INDArray vec;
    @Builder.Default
    private int knnIterations = 64;
    @Builder.Default
    private double bandwidth = 1.0;
    @Builder.Default
    private double localConnectivity = 1.0;
    @Builder.Default
    private int k = 1;
    @Builder.Default
    private ThreadLocal<MemoryWorkspace>  workspaceThread = new ThreadLocal<>();
    private WorkspaceConfiguration workspaceConfiguration;
    private WorkspaceMode workspaceMode;
    private INDArray rho,result;



    /**
     * Get and create the {@link MemoryWorkspace} used for nndescent
     * @return
     */
    public MemoryWorkspace getWorkspace() {
        if(this.workspaceThread.get() == null) {
            // opening workspace
            workspaceThread.set(workspaceMode == null || workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace() :
                    Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfiguration, "SmoothKnnDistances-" + Thread.currentThread().getName()));
        }

        return workspaceThread.get();
    }


    /**
     * Compute a continuous version of the distance to the kth nearest
     neighbor. That is, this is similar to knn-distance but allows continuous
     k values rather than requiring an integral k. In essence we are simply
     computing the distance such that the cardinality of fuzzy set we generate
     is k.
     * @param distances the input distances
     * @return
     */
    public INDArray[] smoothedDistances(
            INDArray distances) {
        if(workspaceMode != WorkspaceMode.NONE)
            workspaceConfiguration = WorkspaceConfiguration.builder().cyclesBeforeInitialization(1)
                    .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.FIRST_LOOP)
                    .policyMirroring(MirroringPolicy.FULL).policyReset(ResetPolicy.BLOCK_LEFT)
                    .policySpill(SpillPolicy.REALLOCATE).build();



        try (MemoryWorkspace workspace = getWorkspace().notifyScopeEntered()) {
            rho = Nd4j.zeros(distances.size(0));
            result = Nd4j.zeros(distances.size(0));
            LinkedList<SmoothKnnThread> list = new LinkedList<>();
            for(int i = 0; i < numWorkers; i++) {
                SmoothKnnThread propagationThread = SmoothKnnThread.builder()
                        .bandwidth(bandwidth).localConnectivity(localConnectivity)
                        .iterations(knnIterations).rho(rho).result(result)
                        .knnIterations(knnIterations).numWorkers(numWorkers)
                        .distances(distances).workspaceConfiguration(workspaceConfiguration)
                        .threadLocal(workspaceThread).workspaceMode(workspaceMode)
                        .vec(vec)
                        .id(i + 1).k(k).build();
                list.add(propagationThread);
                executorService.submit(propagationThread);
            }


            while(!list.isEmpty()) {
                SmoothKnnThread curr = list.removeFirst();
                while(!curr.isDone()) {
                    LockSupport.parkNanos(1000L);
                }
            }


            return new INDArray[] {rho,result};
        }

    }


    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    public  static class SmoothKnnThread implements Runnable {
        private int id;
        private INDArray rho,result;
        @Builder.Default
        private AtomicBoolean isDone = new AtomicBoolean(false);
        private INDArray distances;
        private int iterations;
        @Builder.Default
        private double localConnectivity = 1.0;
        @Builder.Default
        private double bandwidth = 1.0;
        private int k;
        @Builder.Default
        private int numWorkers = Runtime.getRuntime().availableProcessors();
        private INDArray vec;
        private int knnIterations;
        private ThreadLocal<MemoryWorkspace> threadLocal;
        private WorkspaceMode workspaceMode;
        private WorkspaceConfiguration workspaceConfiguration;
        public boolean isDone() {
            return isDone.get();
        }


        /**
         * Get and create the {@link MemoryWorkspace} used for nndescent
         * @return
         */
        public MemoryWorkspace getWorkspace() {
            if(this.threadLocal.get() == null) {
                // opening workspace
                threadLocal.set(workspaceMode == null || workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace() :
                        Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfiguration, "SmoothKnnDistances-" + Thread.currentThread().getName()));
            }

            return threadLocal.get();
        }

        @Override
        public void run() {
            double target = Math.log(k) * bandwidth;
            log.info("Starting smooth knn thread " + id);
            int low = numWorkers > 1 ? id * distances.rows() / numWorkers : 0;
            int hi = numWorkers > 1 ?  (id + 1) * distances.rows() / numWorkers : distances.rows();
            try (MemoryWorkspace workspace = getWorkspace().notifyScopeEntered()) {

                /**
                 * Find vectorization if possible
                 */

                for(int x = low; x < hi; x++) {
                    if(x >= distances.slices()) {
                        break;
                    }
                    double lowBound = 0.0;
                    double mid = 1.0;
                    double hiEnd = Double.POSITIVE_INFINITY;
                    INDArray ithDistances = distances.slice(x);
                    INDArray nonZeroDistances = BooleanIndexing.chooseFrom(new INDArray[]{ithDistances}, Arrays.asList(0.0), Collections.<Integer>emptyList(),new GreaterThan());
                    if(nonZeroDistances != null && nonZeroDistances.size(0) >= localConnectivity) {
                        int index = (int) Math.floor(localConnectivity);
                        double interpolation = localConnectivity - index;
                        if(index > 0) {
                            if(interpolation <= SMOOTH_K_TOLERANCE) {
                                double get = nonZeroDistances.getDouble(index - 1);
                                rho.putScalar(x,get);
                            }
                            else {
                                double get = nonZeroDistances.getDouble(index - 1) + interpolation * (nonZeroDistances.getDouble(index) - nonZeroDistances.getDouble(index - 1));
                                rho.putScalar(x,get);
                            }
                        }
                        else {
                            double get = interpolation * nonZeroDistances.getDouble(0);
                            rho.putScalar(x,get);
                        }

                    }
                    else if(nonZeroDistances != null && nonZeroDistances.size(0) > 0) {
                        double get = nonZeroDistances.maxNumber().doubleValue();
                        rho.putScalar(x,get);
                    }
                    else
                        rho.putScalar(x,0.0);

                    for(int iter = 0; iter < knnIterations; iter++) {
                        double pSum = 0.0;
                        for (int j = 1; j < distances.size(1); j++) {
                            double dist = Math.max(0, distances.getDouble(x, j) - rho.getDouble(x));
                            pSum += Math.exp(-dist / mid);
                        }

                        if (Math.abs(pSum - target) < SMOOTH_K_TOLERANCE) {
                            break;
                        }

                        if (pSum > target) {
                            hiEnd = mid;
                            mid = (lowBound + hiEnd) / 2.0;
                        } else {
                            lowBound = mid;
                            if (hiEnd == Double.POSITIVE_INFINITY) {
                                mid *= 2;
                            } else {
                                mid = (lowBound + hiEnd) / 2.0;
                            }
                        }
                    }

                    result.putScalar(x,mid);
                    if(rho.getDouble(x) > 0.0) {
                        double ithDistancesMean = ithDistances.meanNumber().doubleValue();
                        if(result.getDouble(x) < MIN_K_DIST_SCALE * ithDistancesMean) {
                            result.putScalar(x,MIN_K_DIST_SCALE * ithDistancesMean);

                        }
                    }
                    else {
                        double distanceMeanNumber = distances.meanNumber().doubleValue();
                        if(result.getDouble(x) < MIN_K_DIST_SCALE * distanceMeanNumber) {
                            result.putScalar(x,MIN_K_DIST_SCALE * distanceMeanNumber);
                        }

                    }
                }





                isDone.set(true);
                log.info("Finished with propagation thread " + id);
            }

        }
    }


}
