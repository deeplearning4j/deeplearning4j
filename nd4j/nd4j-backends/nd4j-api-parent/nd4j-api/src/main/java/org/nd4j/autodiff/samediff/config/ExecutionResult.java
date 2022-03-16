package org.nd4j.autodiff.samediff.config;

import lombok.Builder;
import lombok.Data;

import java.util.Map;

@Builder
@Data
public class ExecutionResult<T> {

    private Map<String,T> outputs;
    private Map<String,SDValue> valueOutputs;

}
