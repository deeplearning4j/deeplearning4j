package org.nd4j.autodiff.opstate;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.nd4j.autodiff.graph.api.Vertex;

/**
 * Created by agibsonccc on 4/6/17.
 */
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
public class NDArrayVertex extends Vertex<NDArrayInformation>  {

    public NDArrayVertex(int idx,int[] shape) {
        this(idx,
                NDArrayInformation.builder().shape(shape)
                        .id(String.valueOf(idx))
                        .build());
    }

    public NDArrayVertex(int idx, NDArrayInformation value) {
        super(idx, value);
    }

}
