package org.nd4j.imports.descriptors.properties.adapters;

import lombok.AllArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;

import java.lang.reflect.Field;

@AllArgsConstructor
public class ConditionalFieldValueIntIndexArrayAdapter implements AttributeAdapter {
    private Object targetValue;
    private int trueIndex,falseIndex;
    private Field fieldName;

    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        int[] inputValue = (int[]) inputAttributeValue;
        Object comp = on.getValue(fieldName);
        if(targetValue.equals(comp)) {
            on.setValueFor(fieldFor,inputValue[trueIndex]);
        }
        else {
            on.setValueFor(fieldFor,inputValue[falseIndex]);
        }
    }
}
