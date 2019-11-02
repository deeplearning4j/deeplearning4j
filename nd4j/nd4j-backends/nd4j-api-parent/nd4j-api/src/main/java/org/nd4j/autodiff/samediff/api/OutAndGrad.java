package org.nd4j.autodiff.samediff.api;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * A simple object holding two maps - one of output arrays, another of gradient arrays
 */
@AllArgsConstructor
@Data
public class OutAndGrad {

    private final Map<String, INDArray> outputs;
    private final Map<String, INDArray> gradients;

}
