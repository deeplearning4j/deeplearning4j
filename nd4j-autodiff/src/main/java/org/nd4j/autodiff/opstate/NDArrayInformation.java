package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by agibsonccc on 4/6/17.
 */
@Data
@Builder
public class NDArrayInformation implements Serializable {
    private int[] shape;
    private String id;
    private OpState owner;

    @Override
    public String toString() {
        return "NDArrayInformation{" +
                "shape=" + Arrays.toString(shape) +
                ", id='" + id + '\'' +
                '}';
    }
}
