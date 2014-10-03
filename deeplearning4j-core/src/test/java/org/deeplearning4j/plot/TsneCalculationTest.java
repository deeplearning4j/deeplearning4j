package org.deeplearning4j.plot;

import static org.junit.Assert.*;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.Arrays;

/**
 * Created by agibsonccc on 10/1/14.
 */
public class TsneCalculationTest {

    @Test
    public void testTsne() throws Exception {
        TsneCalculation calculation = new TsneCalculation.Builder().normallize(true)
                .build();
        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getFile();
        INDArray data = Nd4j.readTxt(f.getAbsolutePath(),"   ");
        data = data.getRow(0);
        //assertTrue(Arrays.equals(new int[]{2500,784},data.shape()));
        calculation.calculate(data,2,50,20);
    }


}
