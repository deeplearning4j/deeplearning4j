package org.deeplearning4j.malmo;

/**
 * Exception thrown when Malmo cannot connect to a client after multiple retries
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public class MalmoConnectionError extends RuntimeException {
    private static final long serialVersionUID = -9034754802977073358L;

    public MalmoConnectionError(String string) {
        super(string);
    }
}
