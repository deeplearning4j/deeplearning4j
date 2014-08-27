package org.deeplearning4j.example.mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.plot.MultiLayerNetworkReconstructionRender;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by agibsonccc on 6/1/14.
 */
public class MnistReconstructions {

    private static Logger log = LoggerFactory.getLogger(MnistReconstructions.class);


    public static void main(String[] args)  throws Exception {
        DBN d = SerializationUtils.readObject(new File(args[0]));
        MultiLayerNetworkReconstructionRender r = new MultiLayerNetworkReconstructionRender(new MnistDataSetIterator(10,1000,false),d,1);
        r.draw();

    }

}
