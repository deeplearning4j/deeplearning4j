package org.deeplearning4j.datasets.fetchers;

import java.io.File;
import java.io.IOException;
import java.util.List;

import com.google.common.collect.Lists;
import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.datasets.mnist.MnistManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.util.ArrayUtil;


/**
 * Data fetcher for the MNIST dataset
 * @author Adam Gibson
 *
 */
public class MnistDataFetcher extends BaseDataFetcher {

    private static final long serialVersionUID = -3218754671561789818L;
    private transient MnistManager mnistManager;
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
        if(!new File(rootMnist).exists()) {
            new MnistFetcher().downloadAndUntar();
        }
        mnistManager = new MnistManager(rootMnist+ MnistFetcher.trainingFilesFilename_unzipped,
            rootMnist + MnistFetcher.trainingFileLabelsFilename_unzipped);
        numOutcomes = 10;
        this.binarize = binarize;
        totalExamples = NUM_EXAMPLES;
        //1 based cursor

        cursor = 1;
        mnistManager.setCurrent(cursor);
        int[][] image;
        try {
            image = mnistManager.readImage();
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
        List<DataSet> toConvert = Lists.newArrayList();

        for(int i = 0; i < numExamples && hasMore(); i++,cursor++) {
            if(mnistManager == null) {
                try {
                    mnistManager =
                        new MnistManager(rootMnist + MnistFetcher.trainingFilesFilename_unzipped,
                                         rootMnist + MnistFetcher.trainingFileLabelsFilename_unzipped);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            mnistManager.setCurrent(cursor);
            //note data normalization
            try {
                INDArray in = ArrayUtil.toNDArray(ArrayUtil.flatten(mnistManager.readImage()));
                if(binarize)
                    for(int d = 0; d < in.length(); d++) {
                        if(binarize) {
                            if(in.getDouble(d) > 30) {
                                in.putScalar(d, 1);
                            }
                            else {
                                in.putScalar(d, 0);
                            }
                        }
                    }
                 else {
                    in.divi(255);
                }

                INDArray out = createOutputVector(mnistManager.readLabel());
                boolean found = false;
                for(int col = 0; col < out.length(); col++) {
                    if(out.getDouble(col) > 0) {
                        found = true;
                        break;
                    }
                }
                if(!found) {
                    throw new IllegalStateException("Found a matrix without an outcome");
                }
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
