package org.deeplearning4j.caffe.common;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;

/**
 * @author jeffreytang
 */
@Data
@NoArgsConstructor
public class NNConfigBuilderContainer {
    public Builder builder;
    public ListBuilder listBuilder;

    // If Solver is parsed, then the builder field will be populated
    public Boolean parsedSolver() {
        return builder != null;
    }

    // If Net is parsed, then the listBuilder field will be populated
    public Boolean parsedNet() {
        return listBuilder != null;
    }

    // See if both Net and Solver are parsed
    public Boolean parsedAll() {
        return parsedSolver() && parsedNet();
    }
}
