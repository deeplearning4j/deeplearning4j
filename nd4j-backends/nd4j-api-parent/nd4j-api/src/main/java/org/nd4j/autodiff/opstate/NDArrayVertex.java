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
    private SameDiff sameDiff;



    public NDArrayVertex(SameDiff sameDiff, int idx, int depth, int[] shape) {
        this(sameDiff,idx,depth,SDVariable.builder()
        .sameDiff(sameDiff).shape(shape).build());
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
        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result *= 31 * depth;
        result *= 31 * idx;
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
