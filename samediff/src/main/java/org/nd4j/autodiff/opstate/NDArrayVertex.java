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

    public NDArrayVertex(SameDiff sameDiff,int idx, NDArrayInformation value) {
        super(idx, value);
        this.sameDiff = sameDiff;
        if(value.getOwner() != null) {
            if (value.getOwner().getArrayField() != null) {
                Preconditions.checkState(sameDiff == value.getOwner().getArrayField().getOps(), "Invalid same diff instance passe din.");
                if (value.getOwner().getArrayField().getInput().getOwner().getDifferentialFunction() != null)
                    Preconditions.checkState(sameDiff == value.getOwner().getArrayField().getInput().getOwner().getDifferentialFunction().getSameDiff(), "Invalid same diff instance passe din.");
                Preconditions.checkState(sameDiff == value.getOwner().getArrayField().getInput().getOwner().getArrayField().getOps(),"Invalid same diff instance passe din.");

            }
        }

    }

    @Override
    public String toString() {
        return "NDArrayVertex{" +
                "idx=" + idx +
                ", value=" + value +
                '}';
    }
}
