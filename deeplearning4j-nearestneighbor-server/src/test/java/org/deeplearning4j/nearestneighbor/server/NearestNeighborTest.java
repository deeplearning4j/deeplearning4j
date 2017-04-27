package org.deeplearning4j.nearestneighbor.server;

import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.nearestneighbor.model.NearestNeighborRequest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 4/27/17.
 */
public class NearestNeighborTest {

    @Test
    public void testNearestNeighbor() {
        INDArray arr = Nd4j.create(new double[][]{
                {1,2,3,4},
                {1,2,3,5},
                {3,4,5,6}
        });

        VPTree vpTree = new VPTree(arr,false);
        NearestNeighborRequest request = new NearestNeighborRequest();
        request.setK(2);
        request.setInputIndex(0);
        NearestNeighbor nearestNeighbor = NearestNeighbor.builder().tree(vpTree)
                .points(arr).record(request).build();
        assertEquals(1,nearestNeighbor.search().get(0).getIndex());
    }

}
