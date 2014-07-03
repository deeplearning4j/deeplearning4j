package au.com.bytecode.opencsv;

public class CSVRuntimeException extends RuntimeException {

	private static final long serialVersionUID = 1L;
	
	public CSVRuntimeException(Throwable cause) {
		super(cause);
	}
	
	public CSVRuntimeException(String message, Throwable cause) {
		super(message, cause);
	}
}
