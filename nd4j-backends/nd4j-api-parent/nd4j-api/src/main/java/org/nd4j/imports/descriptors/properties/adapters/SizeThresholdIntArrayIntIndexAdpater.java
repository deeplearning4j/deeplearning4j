package org.nd4j.imports.descriptors.properties.adapters;

import lombok.AllArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;

import java.lang.reflect.Field;

@AllArgsConstructor
public class SizeThresholdIntArrayIntIndexAdpater implements AttributeAdapter {
    private int index;
    private int sizeThreshold;
    private int fallbackIndex;


    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        int[] value = (int[]) inputAttributeValue;
        if(value.length < sizeThreshold)
            on.setValueFor(fieldFor,value[fallbackIndex]);
        else
            on.setValueFor(fieldFor,value[index]);
    }
}
