package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.io.Closeable;

/**
 * Used with {@link SameDiff#withNameScope(String)}
 *
 * @author Alex Black
 */
@Data
public class NameScope implements Closeable {
    private final SameDiff sameDiff;
    private final String name;

    public NameScope(SameDiff sameDiff, String name){
        this.sameDiff = sameDiff;
        this.name = name;
    }

    @Override
    public void close() {
        sameDiff.closeNameScope(this);
    }

    @Override
    public String toString(){
        return "NameScope(" + name + ")";
    }
}
