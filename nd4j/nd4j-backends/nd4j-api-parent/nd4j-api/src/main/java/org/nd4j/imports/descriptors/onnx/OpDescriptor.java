package org.nd4j.imports.descriptors.onnx;

import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

@Data
@NoArgsConstructor
public class OpDescriptor implements Serializable {
    private String name;
    private List<TensorDescriptor> inputs;
    private List<TensorDescriptor> outputs;
    private Map<String,String> attrs;
}
