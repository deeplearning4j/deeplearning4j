package org.deeplearning4j.caffe.translate;

import org.deeplearning4j.caffe.CaffeTestUtil;
import org.deeplearning4j.caffe.proto.Caffe;
import org.deeplearning4j.caffe.common.NNCofigBuilderContainer;
import org.deeplearning4j.caffe.common.SolverNetContainer;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertTrue;

/**
 * @author jeffreytang
 */
public class CaffeSolverTranslatorTest {

    @Test
    public void testSolverTranslator() throws IOException, NoSuchFieldException, IllegalAccessException{
        // Get SolverParamter
        SolverNetContainer container = CaffeTestUtil.getLogisticSolverNet();
        Caffe.SolverParameter solver = container.getSolver();

        // Instantiate new BuilderContainer
        NNCofigBuilderContainer builderContainer = new NNCofigBuilderContainer();

        // Instantiate translator to translate and populate the container
        CaffeSolverTranslator solverTranslator = new CaffeSolverTranslator();
        solverTranslator.translate(solver, builderContainer);

        assertTrue(builderContainer.getBuilder() != null);
    }
}
