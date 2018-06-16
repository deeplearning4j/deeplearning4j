package org.nd4j.imports.descriptors.properties.adapters;

import lombok.AllArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;

import java.lang.reflect.Field;

/**
 * Comparison for whether a string equals a target string
 * returning a boolean
 */
@AllArgsConstructor
public class StringNotEqualsAdapter implements AttributeAdapter {
    private String compString;

    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        on.setValueFor(fieldFor, !inputAttributeValue.toString().equals(compString));
    }
}
