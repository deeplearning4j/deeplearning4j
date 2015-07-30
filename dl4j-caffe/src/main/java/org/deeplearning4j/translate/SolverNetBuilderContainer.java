package org.deeplearning4j.translate;

import lombok.Data;
import org.deeplearning4j.caffe.Caffe;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;

/**
 * @author jeffreytang
 *
 * Wrapper for Caffe SolverParameter and NetParamter classes
 * And NeuralNetConfiguration.Builder
 *
 * When SolverParameter and NetParameter are parsed into the Builder,
 * build() is called on Builder to make NeuralNetConfiguration class
 *
 */

@Data
public class SolverNetBuilderContainer {
    public Caffe.NetParameter net;
    public Caffe.SolverParameter solver;
    public Builder builder;
    public ListBuilder listBuilder;

    public SolverNetBuilderContainer(Caffe.SolverParameter solver, Caffe.NetParameter net) {
        this.solver = solver;
        this.net = net;
    }

    // If Solver is parsed, then the builder field will be populated
    public Boolean getParsedSolver() {
        if (builder != null) {
            return true;
        } else {
            return false;
        }
    }


    // If Net is parsed, then the listBuilder field will be populated
    public Boolean getParsedNet() {
        if (listBuilder != null) {
            return true;
        } else {
            return false;
        }
    }

    // See if both Net and Solver are parsed
    public Boolean getParsedState() {
        if (getParsedSolver() && getParsedNet()) {
            return true;
        } else {
            return false;
        }
    }
}
