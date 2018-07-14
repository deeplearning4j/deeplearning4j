package org.deeplearning4j.api.loader.impl;

import org.deeplearning4j.api.loader.DataSetLoader;
import org.nd4j.linalg.dataset.DataSet;

import java.io.InputStream;

public class SerializedDataSetLoader implements DataSetLoader {
    @Override
    public DataSet apply(Source loader) {
        try(InputStream is = loader.getInputStream()){

        }
    }
}
