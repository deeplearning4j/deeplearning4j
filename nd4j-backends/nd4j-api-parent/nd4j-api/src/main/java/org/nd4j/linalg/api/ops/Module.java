package org.nd4j.linalg.api.ops;


/**
 * A Module is a {@link CustomOp}
 * with varying input arguments
 * and automatically calculated outputs
 * defined at a higher level than c++.
 *
 * A Module is meant to be a way of implementing custom operations
 * in straight java/nd4j.
 */
public interface Module extends CustomOp {

    /**
     *
     */
    void exec();


    Module[] subModules();


    void addModule(Module module);


    void execSameDiff();




}
