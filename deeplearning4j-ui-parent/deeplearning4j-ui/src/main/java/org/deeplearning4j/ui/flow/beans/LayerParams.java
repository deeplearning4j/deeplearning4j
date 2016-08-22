package org.deeplearning4j.ui.flow.beans;

import lombok.Data;

import java.util.Map;

/**
 *
 * @author raver119@gmail.com
 */
@Data
public class LayerParams {
    private Map W;
    private Map RW;
    private Map RWF;
    private Map B;
}
