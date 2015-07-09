/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.datasets.fetchers;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
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

    /**
     *
     */
    private static final long serialVersionUID = -3218754671561789818L;
    private transient MnistManager man;
    public final static int NUM_EXAMPLES = 60000;
    private static final String TEMP_ROOT = System.getProperty("user.home");
    private static final String MNIST_ROOT = TEMP_ROOT + File.separator + "MNIST" + File.separator;
    private boolean binarize = true;


    /**
     * Constructor telling whether to binarize the dataset or not
     * @param binarize whether to binarize the dataset or not
     * @throws IOException
     */
    public MnistDataFetcher(boolean binarize) throws IOException {
        if(!new File(MNIST_ROOT).exists()) {
            new MnistFetcher().downloadAndUntar();
        }
        try {
            man = new MnistManager(MNIST_ROOT + MnistFetcher.trainingFilesFilename_unzipped, MNIST_ROOT + MnistFetcher.trainingFileLabelsFilename_unzipped);
        }catch(Exception e) {
            FileUtils.deleteDirectory(new File(MNIST_ROOT));
            new MnistFetcher().downloadAndUntar();
            man = new MnistManager(MNIST_ROOT + MnistFetcher.trainingFilesFilename_unzipped, MNIST_ROOT + MnistFetcher.trainingFileLabelsFilename_unzipped);

        }
        numOutcomes = 10;
        this.binarize = binarize;
        totalExamples = NUM_EXAMPLES;
        cursor = 0;
        inputColumns = man.getImages().getEntryLength();

    }

    public MnistDataFetcher() throws IOException {
        this(true);
    }

    @Override
    public void fetch(int numExamples) {
        if(!hasMore()) {
            throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");
        }

        //we need to ensure that we don't overshoot the number of examples total
        List<DataSet> toConvert = new ArrayList<>();

        for(int i = 0; i < numExamples; i++,cursor++) {
            if(!hasMore()) {
                break;
            }
            if(man == null) {
                try {
                    man = new MnistManager(MNIST_ROOT + MnistFetcher.trainingFilesFilename_unzipped,MNIST_ROOT + MnistFetcher.trainingFileLabelsFilename_unzipped);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            man.setCurrent(cursor);
            //note data normalization
            try {
                INDArray in = ArrayUtil.toNDArray(ArrayUtil.flatten(man.readImage()));
                if(binarize) {
                    for(int d = 0; d < in.length(); d++) {
                        if(in.getDouble(d) > 30) {
                            in.putScalar(d,1);
                        }
                        else {
                            in.putScalar(d,0);
                        }
                    }
                } else {
                    in.divi(255);
                }


                INDArray out = createOutputVector(man.readLabel());
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
        cursor = 0;
        curr = null;
    }

    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }





}
