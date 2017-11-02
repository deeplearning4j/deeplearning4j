package org.nd4j.imports.converters.tf;


/**
 * All op converter
 *
 * @author raver119@gmail.com
 */
public class Prod extends Sum {

    @Override
    public String opName() {
        return "prod";
    }
}
