package org.deeplearning4j.ui.api;

import org.nd4j.shade.dwjackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author Adam Gibson
 */
public @Data @AllArgsConstructor @NoArgsConstructor
 class UrlResource {
    @JsonProperty
    private String url;

}
