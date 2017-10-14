package org.nd4j.imports.intermediate;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This class is intermediate representation of Variable
 *
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TVariable {
    private String name;
    private int id;
    private INDArray array;
    private int[] shape;
    private boolean isPlaceholder;
}
