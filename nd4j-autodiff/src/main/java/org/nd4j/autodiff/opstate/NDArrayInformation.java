package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

/**
 * Created by agibsonccc on 4/6/17.
 */
@Data
@Builder
public class NDArrayInformation implements Serializable {
    private int[] shape;
    private String id;
    private OpState owner;
}
