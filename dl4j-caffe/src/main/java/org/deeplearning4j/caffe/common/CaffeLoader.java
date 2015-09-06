package org.deeplearning4j.caffe.common;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.caffe.proto.Caffe.SolverParameter;
import org.deeplearning4j.caffe.proto.Caffe.NetParameter;
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
public class CaffeLoader {
    protected Object binaryNet;
    protected Object textFormatNet;
    protected Object textFormatSolver;
    protected Integer sizeLimitMb = 10000;

    protected static Logger log = LoggerFactory.getLogger(CaffeLoader.class);
    CaffeReader reader = new CaffeReader();

    /**
     *
     * @param textFormatSolver Path of file of the solver file
     * @return CaffeLoader Class
     */
    @SuppressWarnings("unchecked")
    public CaffeLoader textFormatSolver(Object textFormatSolver) {
        this.textFormatSolver = textFormatSolver;
        return this;
    }

    /**
     *
     * @param textFormatNet Path or file of the TextFormat net file
     * @return CaffeLoader Class
     */
    @SuppressWarnings("unchecked")
    public CaffeLoader textFormatNet(Object textFormatNet) {
        this.textFormatNet = textFormatNet;
        return this;
    }


    /**
     *
     * @param binaryNet Path or InputStream of Binary net file
     * @return CaffeLoader Class
     */
    @SuppressWarnings("unchecked")
    public CaffeLoader binaryNet(Object binaryNet) {
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
            solver = reader.readTextFormatSolver((String) textFormatSolver);
        } else if (textFormatSolver instanceof File) {
            solver = reader.readTextFormatSolver((File) textFormatSolver);
        } else {
            throw new IllegalStateException("textFormatSolver must be of type String (file path) or File.");
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
                net = reader.readBinaryNet((String)binaryNet, sizeLimitMb);
            } else if (binaryNet instanceof InputStream) {
                net = reader.readBinaryNet((InputStream)binaryNet, sizeLimitMb);
            } else {
                throw new IllegalStateException("binaryNet must be of type String (file path) or InputStream.");
            }
        } else if(textFormatNet != null) {
            if (textFormatNet instanceof String) {
                net = reader.readTextFormatNet((String)textFormatNet);
            } else if (textFormatNet instanceof File) {
                net = reader.readTextFormatNet((File)textFormatNet);
            } else {
                throw new IllegalStateException("textFormatNet must be of type String (file path) or File.");
            }
        } else {
            throw new IllegalStateException("Must provide input for either textFormatNet or binaryNet.");
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
