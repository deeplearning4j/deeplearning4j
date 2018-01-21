package org.nd4j.imports.descriptors.properties.adapters;

import lombok.AllArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.lang.reflect.Field;

@AllArgsConstructor
public class ConditionalFieldValueNDArrayShapeAdapter  implements AttributeAdapter {
    private Object targetValue;
    private int trueIndex,falseIndex;
    private Field fieldName;

    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        INDArray inputValue = (INDArray) inputAttributeValue;
        Object compProperty = on.getValue(fieldName);
        if(targetValue.equals(compProperty)) {
            on.setValueFor(fieldFor,inputValue.size(trueIndex));
        }
        else {
            on.setValueFor(fieldFor,inputValue.size(falseIndex));
        }
    }
}
