package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class FrameContextInfo {
    private String frameName;
    private int iteration;
    private String parentFrame;
    private int nestingDepth;
    private List<String> relatedFrames = new ArrayList<>();
    private Map<String, List<String>> crossFrameReferences = new HashMap<>();
}
