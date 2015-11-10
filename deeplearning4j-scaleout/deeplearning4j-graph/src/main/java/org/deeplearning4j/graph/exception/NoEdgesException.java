package org.deeplearning4j.graph.exception;

/**Unchecked exception
 */
public class NoEdgesException extends RuntimeException {

    public NoEdgesException(){
        super();
    }

    public NoEdgesException(String s){
        super(s);
    }

    public NoEdgesException(String s, Exception e ){
        super(s,e);
    }

}
