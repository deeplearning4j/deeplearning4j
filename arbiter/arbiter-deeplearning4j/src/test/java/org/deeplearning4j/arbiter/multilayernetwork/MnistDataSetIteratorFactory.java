package org.deeplearning4j.arbiter.multilayernetwork;

import lombok.Data;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;

import java.io.IOException;

/**
 * Created by agibsonccc on 3/13/17.
 */
@Data
public class MnistDataSetIteratorFactory implements DataSetIteratorFactory {
    /**
     * @return
     */
    @Override
    public DataSetIterator create() {
        try {
            return new MnistDataSetIterator(1000, 1000);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
