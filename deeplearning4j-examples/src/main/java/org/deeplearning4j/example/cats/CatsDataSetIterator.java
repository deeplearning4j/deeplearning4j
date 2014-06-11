package org.deeplearning4j.example.cats;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

import java.io.File;

/**
 * Created by agibsonccc on 6/3/14.
 */
public class CatsDataSetIterator extends BaseDatasetIterator {

    public CatsDataSetIterator(File rootDir,int batch, int numExamples) {
        super(batch, numExamples, new CatDataFetcher(rootDir,56,56));
    }


}
