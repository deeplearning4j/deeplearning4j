package org.deeplearning4j.example.deepautoencoder;

import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collections;
import java.util.List;

/**
 * Read a DBN from a file and use that as the basis for the deep autoencoder
 */
public class DeepAutoEncoderFromFile {
    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderFromFile.class);

    public static void main(String[] args) throws Exception {
        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(100,60000);


        DBN dbn = SerializationUtils.readObject(new File(args[0]));



        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);


        while(iter.hasNext()) {
            DataSet next = iter.next();
            if(next == null)
                break;
            log.info("Training on " + next.numExamples());
            encoder.finetune(next.getFirst(),1e-3,1000);

            //Iterate over the data set after done training and show the 2 side by side (you have to drag the test image over to the right)
            while(iter.hasNext()) {
                DataSet first = iter.next();
                DoubleMatrix reconstruct = encoder.reconstruct(first.getFirst());
                for(int j = 0; j < first.numExamples(); j++) {

                    DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
                    DoubleMatrix reconstructed2 = reconstruct.getRow(j);
                    DoubleMatrix draw2 = reconstructed2.mul(255);

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


        SerializationUtils.saveObject(encoder,new File("deepautoencoder.ser"));

        iter.reset();

        //Iterate over the data set after done training and show the 2 side by side (you have to drag the test image over to the right)
        while(iter.hasNext()) {
            DataSet first = iter.next();
            DoubleMatrix reconstruct = encoder.reconstruct(first.getFirst());
            for(int j = 0; j < first.numExamples(); j++) {

                DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
                DoubleMatrix reconstructed2 = reconstruct.getRow(j);
                DoubleMatrix draw2 = reconstructed2.mul(255);

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
