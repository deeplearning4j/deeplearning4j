package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Arrays;
import java.util.UUID;

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
    private String arrId;

    public static NDArrayInformation newInfo(INDArray arr) {
        return NDArrayInformation.builder()
                .shape(arr.shape())
                .arrId(UUID.randomUUID().toString())
                .scalarValue(arr.isScalar() ? arr.getDouble(0) : null)
                .build();
    }

    public String getArrId() {
        if(arrId == null)
            arrId = UUID.randomUUID().toString();
        return arrId;

    }

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
