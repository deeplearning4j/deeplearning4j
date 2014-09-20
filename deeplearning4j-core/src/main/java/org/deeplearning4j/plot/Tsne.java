package org.deeplearning4j.plot;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.UUID;

/**
 * Stochastic neighbor embedding
 *
 * @author Adam Gibson
 */
public class Tsne {




    private String commandTemplate = "python /tmp/tsne.py --path %s --ndims %d --perplexity %.3f --initialdims %s";

    private static Logger log = LoggerFactory.getLogger(Tsne.class);


    private static ClassPathResource r = new ClassPathResource("/scripts/tsne.py");


    static {
        loadIntoTmp();
    }

    private static void loadIntoTmp() {

        File script = new File("/tmp/tsne.py");


        try {
            List<String> lines = IOUtils.readLines(r.getInputStream());
            FileUtils.writeLines(script, lines);

        } catch (IOException e) {
            throw new IllegalStateException("Unable to load python file");

        }

    }

    public void plot(INDArray matrix,int nDims,float perplexity,int initialDims) throws IOException {

        String path = writeMatrix(matrix);
        String command = String.format(commandTemplate,path,nDims,perplexity,initialDims);
        Process is = Runtime.getRuntime().exec(command);

        log.info("Std out " + IOUtils.readLines(is.getInputStream()).toString());
        log.error(IOUtils.readLines(is.getErrorStream()).toString());

    }




    protected String writeMatrix(INDArray matrix) throws IOException {
        String filePath = System.getProperty("java.io.tmpdir") + File.separator +  UUID.randomUUID().toString();
        File write = new File(filePath);
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
        write.deleteOnExit();
        for(int i = 0; i < matrix.rows(); i++) {
            INDArray row = matrix.getRow(i);
            StringBuffer sb = new StringBuffer();
            for(int j = 0; j < row.length(); j++) {
                sb.append(String.format("%.10f", row.get(j)));
                if(j < row.length() - 1)
                    sb.append(",");
            }
            sb.append("\n");
            String line = sb.toString();
            bos.write(line.getBytes());
            bos.flush();
        }

        bos.close();
        return filePath;
    }

}
