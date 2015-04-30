package org.deeplearning4j.ui;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.dropwizard.Configuration;

import javax.validation.Valid;
import javax.validation.constraints.NotNull;

/**
 * @author Adam Gibson
 */
public class UIConfiguration extends Configuration {
    @JsonProperty
    @NotNull
    @Valid
    private String uploadPath = System.getProperty("java.io.tmpdir");

    public String getUploadPath() {
        if(uploadPath == null || uploadPath.isEmpty())
            uploadPath = System.getProperty("java.io.tmpdir");
        return uploadPath;
    }

    public void setUploadPath(String uploadPath) {
        this.uploadPath = uploadPath;
    }
}
