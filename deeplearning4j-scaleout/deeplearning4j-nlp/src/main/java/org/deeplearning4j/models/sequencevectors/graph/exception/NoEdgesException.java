package org.deeplearning4j.models.sequencevectors.graph.exception;

/**Unchecked exception, thrown to signify that an operation (usually on a vertex) cannot be completed
 * because there are no edges for that vertex.
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
