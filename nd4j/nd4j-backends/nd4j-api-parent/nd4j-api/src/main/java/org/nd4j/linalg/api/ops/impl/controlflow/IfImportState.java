package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.Builder;
import lombok.Data;
import org.tensorflow.framework.NodeDef;

import java.util.List;

@Builder
@Data
public class IfImportState {
    private List<NodeDef> condNodes;
    private List<NodeDef> trueNodes;
    private List<NodeDef> falseNodes;
    private String falseBodyScopeName,trueBodyScopeName,conditionBodyScopeName;
}
