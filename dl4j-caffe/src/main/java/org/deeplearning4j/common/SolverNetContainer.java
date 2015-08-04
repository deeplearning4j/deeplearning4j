package org.deeplearning4j.common;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.caffeprojo.Caffe;
import org.deeplearning4j.caffeprojo.Caffe.NetParameter;

/**
 * @author jeffreytang
 *
 * Wrapper for Caffe SolverParameter and NetParamter classes
 */

@Data
@AllArgsConstructor
public class SolverNetContainer {
    protected Caffe.SolverParameter solver;
    protected NetParameter net;
}
