package org.deeplearning4j.nn.conf.graph;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = ElementWiseVertex.class, name = "ElementWiseVertex"),
        @JsonSubTypes.Type(value = MergeVertex.class, name = "MergeVertex"),
        @JsonSubTypes.Type(value = SubsetVertex.class, name = "SubsetVertex")
})
public abstract class GraphVertex implements Cloneable {

    @Override
    public abstract GraphVertex clone();
}
