package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.ArrayField;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by agibsonccc on 4/6/17.
 */
@Data()
@EqualsAndHashCode(exclude = "owner")
@Builder
public class NDArrayInformation implements Serializable {
    private int[] shape;
    private String id;
    private OpState owner;
    private Number scalarValue;


    public Number scalar() {
        if(scalarValue != null)
            return scalarValue;

        if(owner == null)
            throw new IllegalStateException("No owner set.");
        return owner.getScalarValue();
    }
    @Override
    public String toString() {
        return "NDArrayInformation{" +
                "shape=" + Arrays.toString(shape) +
                ", id='" + id + '\'' +
                '}';
    }
}
