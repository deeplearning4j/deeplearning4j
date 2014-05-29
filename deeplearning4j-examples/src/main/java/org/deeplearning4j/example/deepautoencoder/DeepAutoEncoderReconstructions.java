package org.deeplearning4j.example.deepautoencoder;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by agibsonccc on 5/12/14.
 */
public class DeepAutoEncoderReconstructions {

    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderReconstructions.class);


    public static void main(String[] args) throws Exception {
        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(80,1000);

        DeepAutoEncoder encoder = SerializationUtils.readObject(new File(args[0]));
        //Iterate over the data set after done training and show the 2 side by side (you have to drag the test image over to the right)
        while(iter.hasNext()) {
            DataSet first = iter.next();
            log.info("Encoded "  + encoder.encode(first.getFirst()));
            DoubleMatrix reconstruct = encoder.reconstruct(first.getFirst());
            for(int j = 0; j < first.numExamples(); j++) {

                DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
                DoubleMatrix reconstructed2 = reconstruct.getRow(j);
                DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2, 1, new MersenneTwister(123)).mul(255);

                DrawReconstruction d = new DrawReconstruction(draw1);
                d.title = "REAL";
                d.draw();
                DrawReconstruction d2 = new DrawReconstruction(draw2,1000,1000);
                d2.title = "TEST";
                d2.draw();
                Thread.sleep(10000);
                d.frame.dispose();
                d2.frame.dispose();
            }


        }

    }

}
