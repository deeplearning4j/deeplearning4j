package org.deeplearning4j.api.loader;

import org.nd4j.api.loader.Loader;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * An interface for loading MultiDataSets from a {@link org.nd4j.api.loader.Source}
 *
 * @author Alex Black
 */
public interface MultiDataSetLoader extends Loader<MultiDataSet> {

}
