package org.deeplearning4j.nndescent;

import org.junit.Test;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import static org.junit.Assert.assertEquals;

public class SmoothKnnDistancesTest {

    @Test
    public void testDistances() throws Exception {
        INDArray mnist = Nd4j.createFromNpyFile(new ClassPathResource("nn_data.npz.npy").getFile());
        NNDescent nnDescent = NNDescent.builder()
                .vec(mnist).nTrees(1000)
                .numWorkers(1)
                .build();
        nnDescent.fit();
        try (MemoryWorkspace workspace = nnDescent.getWorkspace().notifyScopeEntered()) {
            SmoothKnnDistances smoothKnnDistances = SmoothKnnDistances.builder()
                    .knnIterations(64)
                    .executorService(nnDescent.getExecutorService())
                    .numWorkers(8)
                    .vec(mnist).build();
            INDArray arr = nnDescent.distancesForEachNearestNeighbors();
            INDArray[] smoothedDistances = smoothKnnDistances.smoothedDistances(arr);
            INDArray resultAssertions = Nd4j.createFromNpyFile(new ClassPathResource("result_save.npz.npy").getFile());
            INDArray rhoAssertions = Nd4j.createFromNpyFile(new ClassPathResource("rho_save.npz.npy").getFile());
           /* assertEquals(resultAssertions,smoothedDistances[1]);
            assertEquals(rhoAssertions,smoothedDistances[0]);*/

        }


    }

}
