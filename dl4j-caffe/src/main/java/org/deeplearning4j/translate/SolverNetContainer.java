package org.deeplearning4j.translate;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.caffe.Caffe;

/**
 * @author jeffreytang
 *
 * Wrapper for Caffe SolverParameter and NetParamter classes
 */

@Data
@AllArgsConstructor
public class SolverNetContainer {
    public Caffe.NetParameter net;
    public Caffe.SolverParameter solver;
}
