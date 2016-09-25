package org.deeplearning4j.ui.activation;

import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * @author Adam Gibson
 */
public @Data @NoArgsConstructor class PathUpdate implements Serializable {
    private String path;


}
