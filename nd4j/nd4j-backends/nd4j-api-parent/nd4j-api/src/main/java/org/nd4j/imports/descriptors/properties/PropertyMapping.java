package org.nd4j.imports.descriptors.properties;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

@Data
@Builder
public class PropertyMapping implements Serializable {

    private String[] propertyNames;
    private Integer tfInputPosition;
    private String onnxAttrName;
    private String tfAttrName;
    private Integer shapePosition;

}
