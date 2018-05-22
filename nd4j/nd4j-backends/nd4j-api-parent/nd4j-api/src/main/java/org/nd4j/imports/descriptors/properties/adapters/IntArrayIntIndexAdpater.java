package org.nd4j.imports.descriptors.properties.adapters;

import lombok.AllArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;

import java.lang.reflect.Field;

@AllArgsConstructor
public class IntArrayIntIndexAdpater implements AttributeAdapter {
    private int index;


    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        int[] value = (int[]) inputAttributeValue;
        on.setValueFor(fieldFor,value[index]);
    }
}
