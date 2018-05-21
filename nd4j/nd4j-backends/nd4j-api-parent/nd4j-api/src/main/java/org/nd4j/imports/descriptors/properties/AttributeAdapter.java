package org.nd4j.imports.descriptors.properties;

import org.nd4j.autodiff.functions.DifferentialFunction;

import java.lang.reflect.Field;

/**
 * Attribute adapter for taking an attribute with an input value
 * and mapping it to the proper output.
 * This is for a case where the output attribute type needs to be
 * adapted in some form to the field in samediff.
 *
 * A common example of this would be an array input trying to map to individual
 * fields in samediff.
 *
 * For example you have:
 * [1,2,3,4,5]
 *
 * A possible implementation of {@link #mapAttributeFor(Object, Field, DifferentialFunction)} could be:
 *  void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
 *      int[] inputArr = (int[]) inputAttributeValue;
 *      on.setValueFor(fieldFor,inputArr[1]);
 *  }
 *
 *  on.setValueFor(conversions.get(fieldFor),
 */
public interface AttributeAdapter {

    /**
     * Map the attribute using the specified field
     * on the specified function on
     * adapting the given input type to
     * the type of the field for the specified function.
     * @param inputAttributeValue the evaluate to adapt
     * @param fieldFor the field for
     * @param on the function to map on
     */
    void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on);

}
