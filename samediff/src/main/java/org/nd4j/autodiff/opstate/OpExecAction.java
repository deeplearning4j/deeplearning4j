package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

/**
 * An action in the graph
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class OpExecAction implements Serializable {
    private OpState opState;
    private NDArrayInformation[] inputs;
    private NDArrayInformation output;
    private int[] inputsIds;
    private int outputId;

}
