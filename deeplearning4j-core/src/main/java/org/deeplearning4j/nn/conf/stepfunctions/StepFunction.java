package org.deeplearning4j.nn.conf.stepfunctions;

import java.io.Serializable;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.annotation.JsonTypeInfo.As;
import com.fasterxml.jackson.annotation.JsonTypeInfo.Id;

/**
 * Custom step function for line search.
 */
@JsonTypeInfo(use=Id.NAME, include=As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = DefaultStepFunction.class, name = "default"),
        @JsonSubTypes.Type(value = GradientStepFunction.class, name = "gradient"),
        @JsonSubTypes.Type(value = NegativeDefaultStepFunction.class, name = "negativeDefault"),
        @JsonSubTypes.Type(value = NegativeGradientStepFunction.class, name = "negativeGradient"),
        })
public class StepFunction implements Serializable {

    private static final long serialVersionUID = -1884835867123371330L;

}
