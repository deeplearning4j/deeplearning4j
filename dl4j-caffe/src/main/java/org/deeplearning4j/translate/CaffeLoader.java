package org.deeplearning4j.translate;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.caffe.Caffe.SolverParameter;
import org.deeplearning4j.caffe.Caffe.NetParameter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

/**
 * @author jeffreytang
 */
@Data
@NoArgsConstructor
public class CaffeLoader<T> {
    protected T binaryNet;
    protected T textFormatNet;
    protected T textFormatSolver;
    protected Integer sizeLimitMb = 10000;

    protected static Logger log = LoggerFactory.getLogger(CaffeLoader.class);

    /**
     *
     * @param textFormatSolver Path of file of the solver file
     * @return CaffeLoader Class
     */
    public CaffeLoader textFormatSolver(T textFormatSolver) {
        this.textFormatSolver = textFormatSolver;
        return this;
    }

    /**
     *
     * @param textFormatNet Path or file of the TextFormat net file
     * @return CaffeLoader Class
     */
    public CaffeLoader textFormatNet(T textFormatNet) {
        this.textFormatNet = textFormatNet;
        return this;
    }


    /**
     *
     * @param binaryNet Path or InputStream of Binary net file
     * @return CaffeLoader Class
     */
    public CaffeLoader binaryNet(T binaryNet) {
        this.binaryNet = binaryNet;
        return this;
    }

    /**
     *
     * @param sizeLimitMb Size limit (MB) of how big the Binary net is. Defaults to 10,000
     * @return CaffeLoader Class
     */
    public CaffeLoader sizeLimit(int sizeLimitMb) {
        this.sizeLimitMb = sizeLimitMb;
        return this;
    }

    /**
     *
     * @return SolverParameter Class
     * @throws IOException
     */
    private SolverParameter loadSolver() throws IOException {
        SolverParameter solver;
        // Check the solver file is a path or file
        if (textFormatSolver instanceof String) {
            solver = CaffeReader.readTextFormatSolver((String)textFormatSolver);
        } else if (textFormatSolver instanceof File) {
            solver = CaffeReader.readTextFormatSolver((File)textFormatSolver);
        } else {
            throw new UnsupportedOperationException("textFormatSolver must be of type String (file path) or File.");
        }
        return solver;
    }

    /**
     *
     * @return NetParameter Class
     * @throws IOException
     */
    private NetParameter loadNet() throws IOException {
        NetParameter net;
        // Check at least there is one source for net file
        if(binaryNet != null) {
            if (binaryNet instanceof String) {
                net = CaffeReader.readBinaryNet((String)binaryNet, sizeLimitMb);
            } else if (binaryNet instanceof InputStream) {
                net = CaffeReader.readBinaryNet((InputStream)binaryNet, sizeLimitMb);
            } else {
                throw new UnsupportedOperationException("binaryNet must be of type String (file path) or InputStream.");
            }
        } else if(textFormatNet != null) {
            if (textFormatNet instanceof String) {
                net = CaffeReader.readTextFormatNet((String)textFormatNet);
            } else if (textFormatNet instanceof File) {
                net = CaffeReader.readTextFormatNet((File)textFormatNet);
            } else {
                throw new UnsupportedOperationException("textFormatNet must be of type String (file path) or File.");
            }
        } else {
            throw new UnsupportedOperationException("Must provide input for either textFormatNet or binaryNet.");
        }
        return net;
    }


    /**
     *
     * @return SolverNetContainer Class
     * @throws IOException
     */
    public SolverNetContainer load() throws IOException{
        SolverParameter solver = loadSolver();
        NetParameter net = loadNet();
        return new SolverNetContainer(solver, net);
    }
}
