package org.deeplearning4j.nndescent;

import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class NNDescentTest {


    @Test
    public void testEuclideanVectorWise() {
        INDArray  x = Nd4j.linspace(1,4,4);
        INDArray y = Nd4j.linspace(1,16,16).reshape(4,4);
        for(int i = 0; i < y.rows(); i++) {
            EuclideanDistance op = new EuclideanDistance(y.slice(i),x);
            Nd4j.getExecutioner().exec(op);
            System.out.println(op.getFinalResult());
        }
    }


    @Test
    public void testTad() {
        INDArray linspace = Nd4j.linspace(1,16,16).reshape(4,4);
        for(int i = 0; i < linspace.tensorssAlongDimension(1); i++) {
            System.out.println(Arrays.toString(linspace.javaTensorAlongDimension(i,1).stride()));
        }
    }


    @Test
    public void testBasicNNDescent() throws Exception {
        /**
         * Write comparison tests with rptree in python,
         * as well as the nn_descent lib within umap using
         * mnist
         */
        Nd4j.getRandom().setSeed(42);
        INDArray mnist = Nd4j.createFromNpyFile(new ClassPathResource("nn_data.npz.npy").getFile());

        NNDescent nnDescent = NNDescent.builder()
                .vec(mnist).nTrees(1000)
                .numWorkers(8)
                .build();
        nnDescent.fit();
        List<Pair<Double, Integer>> queryResult = nnDescent.getRpTree().queryWithDistances(mnist.slice(0),10);
        List<Integer> rpTreeIndices = new ArrayList<>();
        for(int i = 0; i < queryResult.size(); i++) {
            rpTreeIndices.add(queryResult.get(i).getRight());
        }
        List<DataPoint> dataPointList = new ArrayList<>();
        List<Double> distances = new ArrayList<>();
        VPTree vpTree = new VPTree(mnist);

        vpTree.search(mnist.slice(0),10,dataPointList,distances);
        List<Integer> vpTreeIndices = new ArrayList<>();

        for(DataPoint dataPoint : dataPointList) {
            vpTreeIndices.add(dataPoint.getIndex());
        }

        Set<Integer> rpTreeIndicesSet = new HashSet<>(rpTreeIndices);
        Set<Integer> vpTreeIndicesSet = new HashSet<>(vpTreeIndices);
        assertEquals(rpTreeIndicesSet,vpTreeIndicesSet);
        System.out.println(rpTreeIndices);
        System.out.println(vpTreeIndices);

        INDArray allDistances = nnDescent.distancesForEachNearestNeighbors();
        assertArrayEquals(new int[] {mnist.rows(),nnDescent.getNNeighbors()},allDistances.shape());

    }


}
