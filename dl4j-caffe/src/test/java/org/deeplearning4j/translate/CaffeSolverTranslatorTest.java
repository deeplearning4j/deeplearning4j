package org.deeplearning4j.translate;

import org.deeplearning4j.caffe.Caffe;
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
        SolverNetContainer container = CaffeTestUtil.getSolverNet();
        Caffe.SolverParameter solver = container.getSolver();

        // Instantiate new BuilderContainer
        NNCofigBuilderContainer builderContainer = new NNCofigBuilderContainer();

        // Instantiate translator to translate and populate the container
        CaffeSolverTranslator solverTranslator = new CaffeSolverTranslator();
        solverTranslator.translate(solver, builderContainer);

        assertTrue(builderContainer.getBuilder() != null);
    }
}
