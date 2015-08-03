package org.deeplearning4j.translate;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.caffe.Caffe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * @author jeffreytang
 */
@Data
@NoArgsConstructor
public class CaffeLoader {
    protected String binaryNetPath;
    protected File binaryNetFile;
    protected String textFormatNetPath;
    protected File textFormatNetFile;
    protected String textFormatSolverPath;
    protected File textFormatSolverFile;

    protected static Logger log = LoggerFactory.getLogger(CaffeLoader.class);

    public CaffeLoader(Builder builder) {
        this.binaryNetPath = builder.binaryNetPath;
        this.binaryNetFile = builder.binaryNetFile;
        this.textFormatNetPath = builder.textFormatNetPath;
        this.textFormatNetFile = builder.textFormatNetFile;
        this.textFormatSolverPath = builder.textFormatSolverPath;
        this.textFormatSolverFile = builder.textFormatSolverFile;
    }

    public static class Builder {
        private String binaryNetPath;
        private File binaryNetFile;
        private String textFormatNetPath;
        private File textFormatNetFile;
        private String textFormatSolverPath;
        private File textFormatSolverFile;

        public Builder textFormatSolver(File textFormatSolverFile) {
            this.textFormatSolverFile = textFormatSolverFile;
            return this;
        }

        public Builder textFormatSolver(String textFormatSolverPath) {
            this.textFormatSolverPath = textFormatSolverPath;
            return this;
        }

        public Builder textFormatNet(File textFormatNetFile) {
            this.textFormatNetFile = textFormatNetFile;
            return this;
        }

        public Builder textFormatNet(String textFormatNetPath) {
            this.textFormatNetPath = textFormatNetPath;
            return this;
        }

        public Builder binaryNet(File binaryNetFile) {
            this.binaryNetFile = binaryNetFile;
            return this;
        }

        public Builder binaryNet(String binaryNetPath) {
            this.binaryNetPath = binaryNetPath;
            return this;
        }

        public CaffeLoader load() {
            // Check at least there is one source of solver file
            if (textFormatSolverPath == null && textFormatSolverFile == null)
                throw new UnsupportedOperationException("Must input solver file path or file object (TextFormat).");
            // If both file and path of solver file provided, read in path
            if (textFormatSolverPath != null && textFormatSolverFile != null)
                log.info("Detected 2 sources (path and file) of solver file. Reading in the path provided.");

            // Check at least there is one source for net file
            if (textFormatNetPath == null && textFormatNetFile == null &&
                    binaryNetPath == null && binaryNetFile == null)
                throw new UnsupportedOperationException("Must input net file path or file object (TextFormat or Binary).");
            // If textFormat and binary are provided, use binary instead of textFormat
            if ((textFormatNetPath != null || textFormatNetFile != null) &&
                    (binaryNetPath != null || binaryNetFile != null))
                log.info("Detected 2 sources (binary and text) of net file. Reading in the binary provided.");


        }
    }


    protected static SolverNetBuilderContainer load(String netPath,
                                                    String solverPath,
                                                    boolean binaryFile) throws IOException {
        Caffe.NetParameter net;
        Caffe.SolverParameter solver;


        if (binaryFile) {
            log.info("Reading in binary format Caffe netowrk configurations");
            try {
                net = CaffeReader.readBinaryNet(netPath, 10000);
            } catch (OutOfMemoryError e) {
                throw new OutOfMemoryError("Model is bigger than 10GB. If you want o raise the limit, " +
                        "specify the sizeLimitMb");
            }
        } else {
            net = CaffeReader.readTextFormatNet(netPath);
        }

        solver = CaffeReader.readTextFormatSolver(solverPath);

        return new SolverNetBuilderContainer(solver, net);
    }
}
