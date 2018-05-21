package org.deeplearning4j.streaming.embedded;

import scala.Option;

/**
 * Created by agibsonccc on 6/9/16.
 */
public class StringOption extends Option<String> {
    private String value;

    public StringOption(String value) {
        this.value = value;
    }

    @Override
    public boolean isEmpty() {
        return value == null || value.isEmpty();
    }

    @Override
    public String get() {
        return value;
    }

    @Override
    public Object productElement(int n) {
        return value.charAt(n);
    }

    @Override
    public int productArity() {
        return value.length();
    }

    @Override
    public boolean canEqual(Object that) {
        return that instanceof String;
    }

    @Override
    public boolean equals(Object that) {
        return that.equals(value);
    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }
}
