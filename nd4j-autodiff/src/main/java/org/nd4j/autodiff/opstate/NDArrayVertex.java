package org.nd4j.autodiff.opstate;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.graph.api.Vertex;

/**
 * Created by agibsonccc on 4/6/17.
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class NDArrayVertex extends Vertex<NDArrayInformation> {

    public NDArrayVertex(int idx,int[] shape) {
        this(idx,NDArrayInformation.builder().shape(shape).build());
    }

    public NDArrayVertex(int idx, NDArrayInformation value) {
        super(idx, value);
    }
}
