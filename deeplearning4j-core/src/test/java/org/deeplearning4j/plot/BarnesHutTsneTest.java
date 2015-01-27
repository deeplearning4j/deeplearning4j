package org.deeplearning4j.plot;

import org.apache.commons.io.IOUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.List;

/**
 * Created by agibsonccc on 10/1/14.
 */
public class BarnesHutTsneTest {

    @Test
    public void testTsne() throws Exception {
        BarnesHutTsne b = new BarnesHutTsne.Builder().theta(0.3).learningRate(500)
                .build();
        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getFile();
        INDArray data = Nd4j.readTxt(f.getAbsolutePath(),"   ");
        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream());
        b.fit(data);
    }


}
