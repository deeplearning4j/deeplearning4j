package org.nd4j.autodiff.opstate;

import com.google.common.base.Preconditions;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Created by agibsonccc on 4/6/17.
 */
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
public class NDArrayVertex extends Vertex<NDArrayInformation>  {
    private OpState opState;
    private SameDiff sameDiff;
    public NDArrayVertex(SameDiff sameDiff,int idx, int[] shape) {
        this(sameDiff,idx,
                NDArrayInformation.builder().shape(shape)
                        .id(String.valueOf(idx))
                        .build());
    }

    /**
     *
     * @param sameDiff
     * @param idx
     * @param value
     */
    public NDArrayVertex(SameDiff sameDiff,int idx, NDArrayInformation value) {
        super(idx, value);
        this.sameDiff = sameDiff;
        if(value.getOwner() != null) {
            if (value.getOwner().getArrayField() != null) {
                Preconditions.checkState(sameDiff == value.getOwner().getArrayField().getOps(), "Invalid same diff instance passed in.");
                if (value.getOwner().getArrayField().getInput().getOwner().getDifferentialFunction() != null)
                    Preconditions.checkState(sameDiff == value.getOwner().getArrayField().getInput().getOwner().getDifferentialFunction().getSameDiff(), "Invalid same diff instance passe din.");
                Preconditions.checkState(sameDiff == value.getOwner().getArrayField().getInput().getOwner().getArrayField().getOps(),"Invalid same diff instance passe din.");

            }
        }

    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        NDArrayVertex vertex = (NDArrayVertex) o;

        return opState != null ? opState.equals(vertex.opState) : vertex.opState == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (opState != null ? opState.hashCode() : 0);
        return result;
    }

    @Override
    public String toString() {
        return "NDArrayVertex{" +
                "idx=" + idx +
                ", value=" + value +
                '}';
    }
}
