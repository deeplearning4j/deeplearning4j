package org.ansj.exception;

public class LibraryException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public LibraryException(Exception e) {
        super(e);
    }

    public LibraryException(String message) {
        super(message);
    }

}
