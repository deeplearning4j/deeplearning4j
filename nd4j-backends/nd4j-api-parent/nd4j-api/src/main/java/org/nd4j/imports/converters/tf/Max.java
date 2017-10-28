package org.nd4j.imports.converters.tf;


/**
 * Max op converter
 *
 * @author raver119@gmail.com
 */
public class Max extends Sum {

    @Override
    public String opName() {
        return "max";
    }
}
