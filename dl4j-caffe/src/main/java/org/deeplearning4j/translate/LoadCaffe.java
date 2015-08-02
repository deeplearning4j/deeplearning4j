package org.deeplearning4j.translate;

import lombok.Data;
import org.deeplearning4j.caffe.Caffe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * @author jeffreytang
 */
@Data
public class LoadCaffe {

    protected static Logger log = LoggerFactory.getLogger(TranslateCaffe.class);




    protected static SolverNetBuilderContainer load(String netPath,
                                                    String solverPath,
                                                    boolean binaryFile) throws IOException {
        Caffe.NetParameter net;
        Caffe.SolverParameter solver;


        if (binaryFile) {
            log.info("Reading in binary format Caffe netowrk configurations");
            try {
                net = ReadCaffe.readBinaryNet(netPath, 10000);
            } catch (OutOfMemoryError e) {
                throw new OutOfMemoryError("Model is bigger than 10GB. If you want o raise the limit, " +
                        "specify the sizeLimitMb");
            }
        } else {
            net = ReadCaffe.readTextFormatNet(netPath);
        }

        solver = ReadCaffe.readTextFormatSolver(solverPath);

        return new SolverNetBuilderContainer(solver, net);
    }
}
