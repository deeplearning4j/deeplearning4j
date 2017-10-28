package org.nd4j.imports.converters.tf;


/**
 * All op converter
 *
 * @author raver119@gmail.com
 */
public class Any extends Sum {

    @Override
    public String opName() {
        return "any";
    }
}
