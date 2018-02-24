package org.deeplearning4j.nndescent;

import com.google.common.primitives.Doubles;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;
import java.util.stream.Collectors;

public class SmoothKnnDistancesTest {

    @Test
    public void testDistances() throws Exception {
        INDArray mnist = Nd4j.createFromNpyFile(new ClassPathResource("nn_data.npz.npy").getFile());
        NNDescent nnDescent = NNDescent.builder()
                .vec(mnist).nTrees(1000)
                .numWorkers(1)
                .build();
        nnDescent.fit();
        List<Pair<Double, Integer>> queryResult = nnDescent.getRpTree().queryWithDistances(mnist.slice(0),10);
        SmoothKnnDistances smoothKnnDistances = SmoothKnnDistances.builder()
                .knnIterations(64)
                .executorService(nnDescent.getExecutorService())
                .numWorkers(1)
                .vec(mnist)
                .vec(mnist).build();
        INDArray arr = Nd4j.create(Doubles.toArray(queryResult.stream().map(pair -> pair.getFirst()).collect(Collectors.toList())));
        smoothKnnDistances.smoothedDistances(arr,10,64,1.0,1.0);
    }

}
