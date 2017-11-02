package org.nd4j.imports.converters.tf;


/**
 * Min op converter
 *
 * @author raver119@gmail.com
 */
public class Min extends Sum {

    @Override
    public String opName() {
        return "min";
    }
}
