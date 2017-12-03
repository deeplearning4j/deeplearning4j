package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Created by agibsonccc on 5/1/17.
 */
@Data
@Builder
public class OpExecOrder {
    private List<OpExecAction> actions;





}
