package org.deeplearning4j.nn.modelimport.keras;


/**
 * Indicates that user is attempting to import a Keras model configuration that
 * is not currently supported by DL4J model import.
 *
 * See https://deeplearning4j.org/model-import-keras for more information.
 *
 * @author davekale
 */
public class IncompatibleKerasConfigurationException extends RuntimeException {

    public IncompatibleKerasConfigurationException(String message) { super(appendDocumentationURL(message)); }

    public IncompatibleKerasConfigurationException(String message, Throwable cause) {
        super(appendDocumentationURL(message), cause);
    }

    public IncompatibleKerasConfigurationException(Throwable cause) { super(cause); }

    private static String appendDocumentationURL(String message) {
        return message + " For more information, see https://deeplearning4j.org/model-import-keras.";
    }
}
