package org.nd4j.imports.descriptors.onnx;

import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
public class TensorDescriptor {
    private List<String> types;
    private String typeStr;
    private String name;
}
