package org.deeplearning4j.api.loader.impl;

import org.deeplearning4j.api.loader.DataSetLoader;
import org.nd4j.api.loader.Source;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.io.InputStream;

/**
 * Loads DataSets using {@link DataSet#load(InputStream)}
 *
 * @author Alex Black
 */
public class SerializedDataSetLoader implements DataSetLoader {
    @Override
    public DataSet load(Source source) throws IOException {
        DataSet ds = new DataSet();
        try(InputStream is = source.getInputStream()){
            ds.load(is);
        }
        return ds;
    }
}
