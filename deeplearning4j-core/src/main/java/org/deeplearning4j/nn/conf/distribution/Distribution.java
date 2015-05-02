package org.deeplearning4j.nn.conf.distribution;

import java.io.Serializable;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.annotation.JsonTypeInfo.As;
import com.fasterxml.jackson.annotation.JsonTypeInfo.Id;

/**
 * An abstract distribution.
 *
 */
@JsonTypeInfo(use=Id.NAME, include=As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = BinomialDistribution.class, name = "binomial"),
        @JsonSubTypes.Type(value = NormalDistribution.class, name = "normal"),
        @JsonSubTypes.Type(value = UniformDistribution.class, name = "uniform"),
        })
public abstract class Distribution implements Serializable {

    private static final long serialVersionUID = 5401741214954998498L;
}
