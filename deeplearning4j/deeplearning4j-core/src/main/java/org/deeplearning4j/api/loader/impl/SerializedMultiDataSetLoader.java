package org.deeplearning4j.api.loader.impl;

import org.deeplearning4j.api.loader.MultiDataSetLoader;
import org.nd4j.api.loader.Source;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.io.IOException;
import java.io.InputStream;

public class SerializedMultiDataSetLoader implements MultiDataSetLoader {
    @Override
    public MultiDataSet load(Source source) throws IOException {
        org.nd4j.linalg.dataset.MultiDataSet ds = new org.nd4j.linalg.dataset.MultiDataSet();
        try(InputStream is = source.getInputStream()){
            ds.load(is);
        }
        return ds;
    }
}
