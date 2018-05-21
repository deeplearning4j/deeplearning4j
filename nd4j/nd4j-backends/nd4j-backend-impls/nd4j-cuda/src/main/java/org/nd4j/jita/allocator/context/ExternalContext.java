package org.nd4j.jita.allocator.context;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * This is simple class-independant storage for device contexts.
 *
 * TODO: Something better then typecast required here
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ExternalContext {
    private Object context;
}
