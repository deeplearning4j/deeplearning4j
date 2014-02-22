package org.deeplearning4j.exception;

public class DeepLearningException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7973589163269627293L;

	public DeepLearningException() {
		super();
	}

	public DeepLearningException(String message, Throwable cause,
			boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}

	public DeepLearningException(String message, Throwable cause) {
		super(message, cause);
	}

	public DeepLearningException(String message) {
		super(message);
	}

	public DeepLearningException(Throwable cause) {
		super(cause);
	}

	

}
