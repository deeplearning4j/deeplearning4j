package org.nd4j.autodiff.opstate;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;

/**
 * Created by agibsonccc on 4/6/17.
 */
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
public class NDArrayVertex extends Vertex<SDVariable>  {
    private OpState opState;
    private SameDiff sameDiff;



    public NDArrayVertex(SameDiff sameDiff, int idx, int depth, int[] shape) {
        this(sameDiff,idx,depth,SDVariable.builder()
        .sameDiff(sameDiff).shape(shape).vertexId(new int[]{idx}).build());
    }

    /**
     *
     * @param sameDiff
     * @param idx
     * @param depth the depth of the vertex
     * @param value
     */
    public NDArrayVertex(
            SameDiff sameDiff,
            int idx,
            int depth,
            SDVariable value) {
        super(idx, depth,value);
        this.sameDiff = sameDiff;


    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NDArrayVertex vertex = (NDArrayVertex) o;
        if(vertex.depth != this.depth)
            return false;
        return opState != null ? opState.equals(vertex.opState) : vertex.opState == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result *= 31 * depth;
        result *= 31 * idx;
        result = 31 * result + (opState != null ? opState.hashCode() : 0);
        return result;
    }

    @Override
    public String toString() {
        return "NDArrayVertex{" +
                "idx=" + idx +
                ", depth=" + depth +
                ", value=" + value +
                '}';
    }
}
