package org.deeplearning4j.ui;

import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class ApiTest {
    @Test
    @Ignore
    public void testUpdateCoords() throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        Nd4j.factory().setDType(DataBuffer.Type.DOUBLE);
        Nd4j.getRandom().setSeed(123);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(250).theta(0.5).learningRate(500)
                        .useAdaGrad(false).numDimension(2).build();

        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getFile();
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(), "   ").get(NDArrayIndex.interval(0, 100),
                        NDArrayIndex.interval(0, 784));



        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream()).subList(0, 100);
        b.fit(data);
        b.saveAsFile(labelsList, "coords.csv");
        //        String coords =  client.target("http://localhost:8080").path("api").path("update")
        //                .request().accept(MediaType.APPLICATION_JSON)
        ////                .post(Entity.entity(new UrlResource("http://localhost:8080/api/coords.csv"), MediaType.APPLICATION_JSON))
        //                .readEntity(String.class);
        //        ObjectMapper mapper = new ObjectMapper();
        //        List<String> testLines = mapper.readValue(coords,List.class);
        //        List<String> lines = IOUtils.readLines(new FileInputStream("coords.csv"));
        //        assertEquals(testLines,lines);

        throw new RuntimeException("Not implemented");
    }

}
