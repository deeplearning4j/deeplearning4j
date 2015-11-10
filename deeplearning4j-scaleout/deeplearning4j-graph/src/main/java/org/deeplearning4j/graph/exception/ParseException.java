package org.deeplearning4j.graph.exception;

public class ParseException extends RuntimeException {
    public ParseException(){
        super();
    }

    public ParseException(String s){
        super(s);
    }

    public ParseException(String s, Exception e){
        super(s,e);
    }
}
