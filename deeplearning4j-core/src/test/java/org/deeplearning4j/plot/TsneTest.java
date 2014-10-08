package org.deeplearning4j.plot;

import com.google.common.reflect.ClassPath;
import org.apache.commons.io.IOUtils;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by agibsonccc on 10/1/14.
 */
public class TsneTest {

    @Test
    public void testTsne() throws Exception {
        Tsne calculation = new Tsne.Builder().setMaxIter(10000)
                .normalize(true).useAdaGrad(true).learningRate(500)
                .build();
        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getFile();
        INDArray data = Nd4j.readTxt(f.getAbsolutePath(),"   ");
        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream());

        calculation.plot(data,2,28, labelsList);
    }


}
