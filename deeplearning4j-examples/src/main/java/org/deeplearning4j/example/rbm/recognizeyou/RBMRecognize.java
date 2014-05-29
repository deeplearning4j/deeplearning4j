package org.deeplearning4j.example.rbm.recognizeyou;

import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.ImageLoader;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by agibsonccc on 5/7/14.
 */
public class RBMRecognize {
    private static Logger log = LoggerFactory.getLogger(RBMRecognize.class);


    public static void main(String[] args) throws Exception {
        ImageLoader loader = new ImageLoader(300,300);

        DoubleMatrix d2 = loader.asRowVector(new File(args[0]));
        MatrixUtil.scaleByMax(d2);
        //MatrixUtil.binarize(0.01,d2);

        RBM r = new RBM.Builder().withHidden(RBM.HiddenUnit.RECTIFIED)
                .withVisible(RBM.VisibleUnit.GAUSSIAN)
                .numberOfVisible(300 * 300)
                .numHidden(1200)
                .build();


        r.trainTillConvergence(d2,1e-2, new Object[]{1,1e-2,10000});


        DoubleMatrix draw1 = r.reconstruct(d2).mul(255);

        DrawReconstruction d = new DrawReconstruction(draw1);
        d.title = "REAL";
        d.draw();


        Thread.sleep(10000);
        d.frame.dispose();




    }

}
