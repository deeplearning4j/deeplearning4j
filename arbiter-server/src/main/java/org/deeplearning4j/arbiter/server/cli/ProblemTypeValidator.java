package org.deeplearning4j.arbiter.server.cli;

import com.beust.jcommander.IParameterValidator;
import com.beust.jcommander.ParameterException;
import org.deeplearning4j.arbiter.server.ArbiterCliGenerator;

/**
 * Created by agibsonccc on 3/13/17.
 */
public class ProblemTypeValidator implements IParameterValidator {
    /**
     * Validate the parameter.
     *
     * @param name  The name of the parameter (e.g. "-host").
     * @param value The value of the parameter that we need to validate
     * @throws ParameterException Thrown if the value of the parameter is invalid.
     */
    @Override
    public void validate(String name, String value) throws ParameterException {
        if(!value.equals(ArbiterCliGenerator.REGRESSION) || value.equals(ArbiterCliGenerator.CLASSIFICIATION)) {
            throw new ParameterException("Problem type can only be " + ArbiterCliGenerator.REGRESSION + " or " + ArbiterCliGenerator.CLASSIFICIATION);

        }
    }
}
