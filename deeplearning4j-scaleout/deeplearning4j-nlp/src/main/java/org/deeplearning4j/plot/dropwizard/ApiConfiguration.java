package org.deeplearning4j.plot.dropwizard;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.dropwizard.Configuration;
import org.hibernate.validator.constraints.NotEmpty;

/**
 * Api Configuration
 *
 * @author Adam Gibson
 */
public class ApiConfiguration extends Configuration {


    private String coordPath = "coords.csv";
    @JsonProperty
    public String getCoordPath() {
        return coordPath;
    }

    public void setCoordPath(String coordPath) {
        this.coordPath = coordPath;
    }
}
