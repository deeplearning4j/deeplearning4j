package org.nd4j.autodiff.samediff;

import lombok.Data;
import org.nd4j.autodiff.samediff.config.SDValueType;

import java.util.HashMap;
import java.util.Map;

@Data
public class VariableStateInfo {
    private String variableName;
    private String frame;
    private int iteration;
    private SDValueType valueType;
    private Object value;
    private String shape;
    private String dataType;
    private long length;
    private long memoryUsage;
    private String numericalHealth;
    private Map<String, Object> metadata = new HashMap<>();
}
