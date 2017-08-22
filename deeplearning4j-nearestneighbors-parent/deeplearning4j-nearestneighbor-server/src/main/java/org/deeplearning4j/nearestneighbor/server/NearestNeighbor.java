package org.deeplearning4j.nearestneighbor.server;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.nearestneighbor.model.NearestNeighborRequest;
import org.deeplearning4j.nearestneighbor.model.NearestNeighborsResult;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 4/27/17.
 */
@AllArgsConstructor
@Builder
public class NearestNeighbor {
    private NearestNeighborRequest record;
    private VPTree tree;
    private INDArray points;

    public List<NearestNeighborsResult> search() {
        INDArray input = points.slice(record.getInputIndex());
        List<NearestNeighborsResult> results = new ArrayList<>();
        if (input.isVector()) {
            List<DataPoint> add = new ArrayList<>();
            List<Double> distances = new ArrayList<>();
            tree.search(input, record.getK(), add, distances);

            if (add.size() != distances.size()) {
                throw new IllegalStateException(
                        String.format("add.size == %d != %d == distances.size",
                                add.size(), distances.size()));
            }

            for (int i=0; i<add.size(); i++) {
                results.add(new NearestNeighborsResult(add.get(i).getIndex(), distances.get(i)));
            }
        }



        return results;

    }


}
