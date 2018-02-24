package org.deeplearning4j.nndescent;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.GreaterThan;

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

    /**
     * Compute a continuous version of the distance to the kth nearest
     neighbor. That is, this is similar to knn-distance but allows continuous
     k values rather than requiring an integral k. In essence we are simply
     computing the distance such that the cardinality of fuzzy set we generate
     is k.
     * @param distances the input distances
     * @param k The number of nearest neighbors to approximate for
     * @param iterations the number of iterations
     * @param localConnectivity  The local connectivity required -- i.e. the number of nearest
    neighbors that should be assumed to be connected at a local level.
    The higher this value the more connected the manifold becomes
    locally. In practice this should be not more than the local intrinsic
    dimension of the manifold.
     * @param bandwidth  The target bandwidth of the kernel, larger values will produce
    larger return values.
     * @return
     */
    public INDArray[] smoothedDistances(
            INDArray distances,
            int k,
            int iterations,
            double localConnectivity,
            double bandwidth) {
        INDArray rho = Nd4j.zeros(distances.size(0));
        INDArray result = Nd4j.zeros(distances.size(0));
        LinkedList<SmoothKnnThread> list = new LinkedList<>();
        for(int i = 0; i < numWorkers; i++) {
            SmoothKnnThread propagationThread = SmoothKnnThread.builder()
                    .bandwidth(bandwidth).localConnectivity(localConnectivity)
                    .iterations(iterations).rho(rho).result(result)
                    .knnIterations(knnIterations).numWorkers(numWorkers)
                    .distances(distances)
                    .vec(vec)
                    .id(i).k(k).build();
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
        private double localConnectivity;
        private double bandwidth;
        private int k;
        @Builder.Default
        private int numWorkers = Runtime.getRuntime().availableProcessors();
        private INDArray vec;
        private int knnIterations;

        public SmoothKnnThread(int id) {
            this.id = id;
        }

        public boolean isDone() {
            return isDone.get();
        }

        @Override
        public void run() {
            double target = Math.log(k) * bandwidth;
            log.info("Starting propagation thread " + id);
            int[] check = new int[vec.rows()];
            int low = id * distances.rows() / numWorkers;
            int hi = (id + 1) * distances.rows() / numWorkers;

            for(int i = 0; i < check.length; i++) {
                check[i] = -1;
            }

            int y = -1;

            double smoothKTolerance = 1e-12;
            /**
             * Find vectorization if possible
             */
            for(int x = low; x < hi; x++) {
                double lowBound = 0.0;
                double mid = 1.0;
                double hiEnd = Double.POSITIVE_INFINITY;
                INDArray ithDistances = distances.slice(x);
                INDArray nonZeroDistances = BooleanIndexing.chooseFrom(new INDArray[]{ithDistances}, Arrays.asList(0.0), Collections.<Integer>emptyList(),new GreaterThan());
                if(nonZeroDistances != null && nonZeroDistances.size(0) >= localConnectivity) {
                    int index = (int) Math.floor(localConnectivity);
                    double interpolation = localConnectivity - index;
                    if(index > 0) {
                        if(interpolation <= smoothKTolerance) {
                            rho.putScalar(x,nonZeroDistances.getDouble(index - 1));
                        }
                        else {
                            rho.putScalar(x,nonZeroDistances.getDouble(index - 1) + interpolation * (nonZeroDistances.getDouble(index) - nonZeroDistances.getDouble(index - 1)));
                        }
                    }
                    else {
                        rho.putScalar(x,interpolation * nonZeroDistances.getDouble(0));
                    }

                }
                else if(nonZeroDistances != null && nonZeroDistances.size(0) > 0) {
                    rho.putScalar(x,nonZeroDistances.maxNumber().doubleValue());
                }
                else
                    rho.putScalar(x,0.0);

                for(int iter = 0; iter < knnIterations; iter++) {
                    double pSum = 0.0;
                    for(int j = 1; j < distances.size(1); j++) {
                        double dist = Math.max(0,distances.getDouble(x,j) - rho.getDouble(x));
                        pSum += Math.exp(-dist / mid);
                    }

                    if(Math.abs(pSum - target) < smoothKTolerance) {
                        break;
                    }

                    if(pSum > target) {
                        hiEnd = mid;
                        mid = (lowBound + hiEnd) / 2.0;
                    }
                    else {
                        lowBound = mid;
                        if(hiEnd == Double.MAX_VALUE) {
                            mid  *= 2;
                        }
                        else {
                            mid = (lowBound + hiEnd) / 2.0;
                        }
                    }

                    result.putScalar(x,mid);

                    if(rho.getDouble(x) > 0.0) {
                        if(result.getDouble(x) < MIN_K_DIST_SCALE * ithDistances.meanNumber().doubleValue()) {
                            result.putScalar(x,MIN_K_DIST_SCALE * ithDistances.meanNumber().doubleValue());
                        }
                    }
                    else {
                        if(result.getDouble(x) < MIN_K_DIST_SCALE * distances.meanNumber().doubleValue()) {
                            result.putScalar(x,MIN_K_DIST_SCALE * distances.meanNumber().doubleValue());
                        }

                    }
                }

            }



            isDone.set(true);
            log.info("Finished with propagation thread " + id);
        }
    }


}
