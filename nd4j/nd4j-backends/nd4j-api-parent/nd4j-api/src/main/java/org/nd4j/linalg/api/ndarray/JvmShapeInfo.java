package org.nd4j.linalg.api.ndarray;


import lombok.Getter;
import lombok.NonNull;
import org.nd4j.linalg.api.shape.Shape;

public class JvmShapeInfo {
    @Getter protected long[] javaShapeInformation;
    @Getter protected long[] shape;
    @Getter protected long[] stride;
    @Getter protected long length;
    @Getter protected long ews;
    @Getter protected long extras;
    @Getter protected char order;
    @Getter protected int rank;

    public JvmShapeInfo(@NonNull long[] javaShapeInformation) {
        this.javaShapeInformation = javaShapeInformation;
        this.shape = Shape.shape(javaShapeInformation);
        this.stride = Shape.stride(javaShapeInformation);
        this.length = Shape.length(javaShapeInformation);
        this.ews = Shape.elementWiseStride(javaShapeInformation);
        this.extras = Shape.extras(javaShapeInformation);
        this.order = Shape.order(javaShapeInformation);
        this.rank = Shape.rank(javaShapeInformation);
    }
}
