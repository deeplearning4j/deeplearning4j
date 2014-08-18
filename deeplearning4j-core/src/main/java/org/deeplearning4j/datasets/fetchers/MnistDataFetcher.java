package org.deeplearning4j.datasets.fetchers;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.datasets.mnist.MnistManager;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.util.ArrayUtil;


/**
 * Data fetcher for the MNIST dataset
 * @author Adam Gibson
 *
 */
public class MnistDataFetcher extends BaseDataFetcher {

    /**
     *
     */
    private static final long serialVersionUID = -3218754671561789818L;
    private transient MnistManager man;
    public final static int NUM_EXAMPLES = 60000;
    private String tempRoot = System.getProperty("user.home");
    private String rootMnist = tempRoot + File.separator + "MNIST" + File.separator;
    private boolean binarize = true;


    /**
     * Constructor telling whether to binarize the dataset or not
     * @param binarize whether to binarize the dataset or not
     * @throws IOException
     */
    public MnistDataFetcher(boolean binarize) throws IOException {
        if(!new File(rootMnist).exists())
            new MnistFetcher().downloadAndUntar();
        man = new MnistManager(rootMnist+ MnistFetcher.trainingFilesFilename_unzipped,rootMnist + MnistFetcher.trainingFileLabelsFilename_unzipped);
        numOutcomes = 10;
        this.binarize = binarize;
        totalExamples = NUM_EXAMPLES;
        //1 based cursor
        cursor = 1;
        man.setCurrent(cursor);
        int[][] image;
        try {
            image = man.readImage();
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read image");
        }
        inputColumns = ArrayUtil.flatten(image).length;


    }

    public MnistDataFetcher() throws IOException {
        this(true);
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
            if(man == null) {
                try {
                    man = new MnistManager(rootMnist + MnistFetcher.trainingFilesFilename_unzipped,rootMnist + MnistFetcher.trainingFileLabelsFilename_unzipped);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            man.setCurrent(cursor);
            //note data normalization
            try {
                INDArray in = ArrayUtil.toNDArray(ArrayUtil.flatten(man.readImage()));
                if(binarize)
                    for(int d = 0; d < in.length(); d++) {
                        if(binarize) {
                            if((double) in.getScalar(d).element() > 30) {
                                in.putScalar(d,1);
                            }
                            else
                                in.putScalar(d,0);

                        }


                    }
                 else
                      in.divi(255);


                INDArray out = createOutputVector(man.readLabel());
                boolean found = false;
                for(int col = 0; col < out.length(); col++) {
                    if((double) out.getScalar(col).element() > 0) {
                        found = true;
                        break;
                    }
                }
                if(!found)
                    throw new IllegalStateException("Found a matrix without an outcome");

                toConvert.add(new DataSet(in,out));
            } catch (IOException e) {
                throw new IllegalStateException("Unable to read image");

            }
        }


        initializeCurrFromList(toConvert);



    }

    @Override
    public void reset() {
        cursor = 1;
    }

    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }





}
