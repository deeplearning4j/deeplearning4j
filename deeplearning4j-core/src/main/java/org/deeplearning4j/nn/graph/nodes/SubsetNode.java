package org.deeplearning4j.nn.graph.nodes;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/**
 * A GraphNode to select a subset of the activations from the input.
 * Assumes only a single array as input.
 * Subset is along dimension 1 (activation for FF/RNN, depth for CNN)
 */
public class SubsetNode implements GraphNode {

    private int from;
    private int to;
    private int[] forwardShape;

    public SubsetNode(@JsonProperty("from") int fromInclusive, @JsonProperty("to") int toInclusive) {
        this.from = fromInclusive;
        this.to = toInclusive;
    }

    public int getFrom(){
        return from;
    }

    public void setFrom(int from){
        this.from = from;
    }

    public int getTo(){
        return to;
    }

    public void setTo(int to){
        this.to = to;
    }

    @Override
    public boolean equals(Object o){
        if(!(o instanceof SubsetNode)) return false;
        SubsetNode s = (SubsetNode)o;
        return from == s.from && to == s.to;
    }

    @Override
    public int hashCode(){
        return Integer.hashCode(from) ^ Integer.hashCode(to);
    }

    @Override
    public INDArray forward(INDArray... activations) {
        if (activations == null || activations.length != 1)
            throw new IllegalArgumentException("Invalid input: expect exactly 1 input for forward pass (got: " +
                    (activations == null ? null : activations.length) + ")");
        forwardShape = Arrays.copyOf(activations[0].shape(), activations[0].rank());

        switch (activations[0].rank()) {
            case 2:
                return activations[0].get(NDArrayIndex.all(), NDArrayIndex.interval(from, to, true));
            case 3:
                return activations[0].get(NDArrayIndex.all(), NDArrayIndex.interval(from, to, true), NDArrayIndex.all());
            case 4:
                return activations[0].get(NDArrayIndex.all(), NDArrayIndex.interval(from, to, true), NDArrayIndex.all(), NDArrayIndex.all());
            default:
                throw new UnsupportedOperationException("Cannot get subset for activations of rank " + activations[0].rank());
        }
    }

    @Override
    public INDArray[] backward(INDArray epsilon) {
        INDArray out = Nd4j.zeros(forwardShape);
        switch (forwardShape.length) {
            case 2:
                out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(from, to, true)}, epsilon);
                break;
            case 3:
                out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(from, to, true), NDArrayIndex.all()}, epsilon);
                break;
            case 4:
                out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(from, to, true), NDArrayIndex.all(), NDArrayIndex.all()}, epsilon);
                break;
            default:
                throw new RuntimeException("Invalid activation rank");  //Should never happen
        }
        return new INDArray[]{out};
    }

    @Override
    public SubsetNode clone(){
        return new SubsetNode(from,to);
    }
}
