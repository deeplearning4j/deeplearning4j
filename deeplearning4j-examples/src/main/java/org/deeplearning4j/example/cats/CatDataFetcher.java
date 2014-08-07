package org.deeplearning4j.example.cats;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.util.ImageLoader;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by agibsonccc on 6/3/14.
 */
public class CatDataFetcher extends BaseDataFetcher {
    /**
     *
     */
    private static final long serialVersionUID = -7473748140401804666L;
    private ImageLoader loader;
    public final static int NUM_IMAGES = 13233;
    private int imageWidth,imageHeight;
    protected Iterator<File> fileIterator;



    public CatDataFetcher(File rootDir,int imageWidth,int imageHeight) {
        try {
            fileIterator = FileUtils.iterateFiles(rootDir,org.apache.commons.io.filefilter.FileFileFilter.FILE, org.apache.commons.io.filefilter.DirectoryFileFilter.DIRECTORY);
            loader = new ImageLoader(imageWidth,imageHeight);
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            inputColumns = imageWidth * imageHeight;
            numOutcomes = 1;
            totalExamples = NUM_IMAGES;
        } catch (Exception e) {
            throw new IllegalStateException("Unable to fetch images",e);
        }
    }


    public CatDataFetcher(File rootDir) {
        this(rootDir,200,200);
    }

    @Override
    public boolean hasMore() {
        return fileIterator.hasNext();
    }

    @Override
    public void fetch(int numExamples) {
        if(!hasMore())
            throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");



        //we need to ensure that we don't overshoot the number of examples total
        List<DataSet> toConvert = new ArrayList<>();

        for(int i = 0; i < numExamples; i++,cursor++) {
            if(!hasMore())
                break;
            try {
                DoubleMatrix load = loader.asMatrix(fileIterator.next());
                load = load.reshape(1,load.length);
                toConvert.add(new DataSet(load, MatrixUtil.toOutcomeVector(1, 2)));

            }catch(Exception e) {
               throw new RuntimeException(e);
            }
        }

        initializeCurrFromList(toConvert);
    }




    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }

}
