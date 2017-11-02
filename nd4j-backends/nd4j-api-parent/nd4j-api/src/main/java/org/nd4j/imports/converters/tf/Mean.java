package org.nd4j.imports.converters.tf;


/**
 * Mean op converter
 *
 * @author raver119@gmail.com
 */
public class Mean extends Sum {

    @Override
    public String opName() {
        return "mean";
    }
}
