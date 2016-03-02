package org.nd4j.linalg;

import junit.framework.Protectable;
import junit.framework.TestCase;
import junit.framework.TestResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A wrapper test result that hides
 * Unsupported OperationException.
 *
 * The reason for this is certain backends
 * don't support certain features.
 *
 * This will allow us to test only the
 * features that are implemented.
 *
 * @author Adam Gibson
 */
public class Nd4jTestResult extends TestResult {
    private static Logger log = LoggerFactory.getLogger(Nd4jTestResult.class);
    @Override
    protected void run(final TestCase test) {
        startTest(test);
        try {
            Protectable p = new Protectable() {
                public void protect() throws Throwable {
                    test.runBare();
                }
            };
            runProtected(test, p);
            endTest(test);

        }catch(UnsupportedOperationException e) {
            log.warn("Feature not supported " + test.getName());
        }

    }
}
