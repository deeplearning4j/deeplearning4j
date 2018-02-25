package org.deeplearning4j.nndescent;

import lombok.Builder;
import lombok.val;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;
import java.util.concurrent.ExecutorService;

@Builder
public class FuzzySimplicalSet {
    @Builder.Default
    private ThreadLocal<MemoryWorkspace>  workspaceThread = new ThreadLocal<>();
    private WorkspaceConfiguration workspaceConfiguration;
    private WorkspaceMode workspaceMode;
    private ExecutorService executorService;
    @Builder.Default
    private int numWorkers = Runtime.getRuntime().availableProcessors();
    @Builder.Default
    private double bandwidth = 1.0;
    @Builder.Default
    private double localConnectivity = 1.0;

    private int nNeighbors;
    @Builder.Default
    private double setOpMixRatio = 1.0;

    @Builder.Default
    private String distanceFunction = "euclidean";

    private INDArray data;


    /**
     *
     * @return
     */
    public INDArray fit() {
        try(MemoryWorkspace workspace = getWorkspace().notifyScopeEntered()) {
            INDArray rows = Nd4j.create(data.size(0) * nNeighbors);
            INDArray columns = Nd4j.zeros(data.size(0) * nNeighbors);
            INDArray vals = Nd4j.zeros(data.size(0) * nNeighbors);
            int nTrees = 5 + (int) Math.round(Math.pow(data.size(0),0.5 / 20.0));
            int nIters = (int) Math.max(5,Math.round(Math.log(data.size(0))));
            NNDescent nnDescent = NNDescent.builder()
                    .nTrees(nTrees)
                    .iterationCount(nIters)
                    .numWorkers(numWorkers)
                    .vec(data)
                    .executorService(executorService)
                    .workspaceMode(workspaceMode)
                    .normalize(true)
                    .build();
            nnDescent.fit();

            Pair<List<List<Integer>>, INDArray> indicesAnddistancesForEachNearestNeighbors = nnDescent.indicesAnddistancesForEachNearestNeighbors();
            INDArray distances = indicesAnddistancesForEachNearestNeighbors.getRight();

            SmoothKnnDistances smoothKnnDistances = SmoothKnnDistances.builder()
                    .vec(data).executorService(executorService)
                    .localConnectivity(localConnectivity)
                    .bandwidth(bandwidth)
                    .build();
            INDArray[] smoothedDistances = smoothKnnDistances.smoothedDistances(indicesAnddistancesForEachNearestNeighbors.getRight()
            );
            INDArray rhos = smoothedDistances[0];
            INDArray sigmas = smoothedDistances[1];

            val indices = indicesAnddistancesForEachNearestNeighbors.getFirst();
            for(int i = 0; i < indices.size(); i++) {
                for(int j = 0; j < nNeighbors; j++) {
                    double val;
                    if(indices.get(i).get(j) == -1) {
                        continue;
                    }
                    if(indices.get(i).get(j) == i) {
                        val = 0.0;
                    }
                    else if(indices.get(i).get(j)- rhos.getDouble(i) <= 0.0) {
                        val = 1.0;
                    }
                    else {
                        val = Math.exp(-((distances.getDouble(i,j)- rhos.getDouble(i)) / sigmas.getDouble(i) * bandwidth ));
                    }

                    rows.putScalar(i * nNeighbors + j,i);
                    columns.putScalar(i * nNeighbors + j,indices.get(i).get(j));
                    vals.putScalar(i * nNeighbors + j,val);
                }
            }

            INDArray sparse = Nd4j.createSparseCOO(vals.data(),rows.data(),columns.data(),new int[] {data.size(0),data.size(0)});
            INDArray sparseTranspose = sparse.transpose();
            INDArray prodMatrix = sparse.mmul(sparseTranspose);
            sparse.addi(sparse.add(sparseTranspose).subi(prodMatrix)).addi(1 - setOpMixRatio).muli(prodMatrix).muli(setOpMixRatio);
            return sparse;
        }


    }
    /**
     * Get and create the {@link MemoryWorkspace} used for nndescent
     * @return
     */
    public MemoryWorkspace getWorkspace() {
        if(this.workspaceThread.get() == null) {
            // opening workspace
            workspaceThread.set(workspaceMode == null || workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace() :
                    Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfiguration, "FuzzySimplicalSet-" + Thread.currentThread().getName()));
        }

        return workspaceThread.get();
    }





}
