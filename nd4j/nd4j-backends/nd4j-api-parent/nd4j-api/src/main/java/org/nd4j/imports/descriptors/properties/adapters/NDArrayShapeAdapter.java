package org.nd4j.imports.descriptors.properties.adapters;

import lombok.AllArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.lang.reflect.Field;

@AllArgsConstructor
public class NDArrayShapeAdapter implements AttributeAdapter {
    private int index;

    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        INDArray input = (INDArray) inputAttributeValue;
        on.setValueFor(fieldFor,input.size(index));
    }
}
