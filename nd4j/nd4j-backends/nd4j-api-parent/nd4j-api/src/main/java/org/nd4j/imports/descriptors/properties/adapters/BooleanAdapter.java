package org.nd4j.imports.descriptors.properties.adapters;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;

import java.lang.reflect.Field;

public class BooleanAdapter implements AttributeAdapter {

    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        on.setValueFor(fieldFor, (boolean) inputAttributeValue);
    }
}
