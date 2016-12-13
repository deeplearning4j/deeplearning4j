package org.deeplearning4j.spark.models.sequencevectors.export;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ExportContainer<T extends SequenceElement> implements Serializable{
    private T element;
    private INDArray array;

    // TODO: implement B64 optional compression here?
    @Override
    public String toString() {
        // TODO: we need proper string cleansing here
        return element.getLabel() + " " + (Arrays.toString(array.data().asFloat()).replaceAll("(\\[\\],)"," "));
    }
}
