package org.deeplearning4j.datasets.iterator.file;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FileMultiDataSetIterator extends BaseFileIterator<MultiDataSet, MultiDataSetPreProcessor> {


    protected FileMultiDataSetIterator(File rootDir, String... validExtensions) {
        super(rootDir, validExtensions);
    }

    @Override
    protected MultiDataSet load(File f) throws IOException {
        MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet();
        mds.load(f);
        return mds;
    }

    @Override
    protected int sizeOf(MultiDataSet of) {
        return of.getFeatures(0).size(0);
    }

    @Override
    protected List<MultiDataSet> split(MultiDataSet toSplit) {
        return toSplit.asList();
    }

    @Override
    public MultiDataSet merge(List<MultiDataSet> toMerge){
        return org.nd4j.linalg.dataset.MultiDataSet.merge(toMerge);
    }
}
