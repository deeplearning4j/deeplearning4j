package org.arbiter.optimize.ui;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.dropwizard.Configuration;
import org.hibernate.validator.constraints.NotEmpty;

/**
 * Created by Alex on 20/12/2015.
 */
public class ArbiterUIConfig extends Configuration {

    @NotEmpty
    private String template;

    @NotEmpty
    private String defaultName = "ArbiterConf";

    @JsonProperty
    public String getTemplate() {
        return template;
    }

    @JsonProperty
    public void setTemplate(String template) {
        this.template = template;
    }

    @JsonProperty
    public String getDefaultName() {
        return defaultName;
    }

    @JsonProperty
    public void setDefaultName(String name) {
        this.defaultName = name;
    }

}
