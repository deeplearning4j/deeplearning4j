package org.deeplearning4j.api.loader;

import org.nd4j.api.loader.Loader;
import org.nd4j.linalg.dataset.DataSet;

/**
 * An interface for loading DataSets from a {@link org.nd4j.api.loader.Source}
 *
 * @author Alex Black
 */
public interface DataSetLoader extends Loader<DataSet> {

}
