package org.deeplearning4j;

/**
 * Base pipeline class
 *
 * @author Adam Gibson
 */
public abstract  class BasePipeline implements Pipeline {
    protected String[] inputUris,outputUris,canovaUris;

    @Override
    public String[] inputUris() {
        return inputUris;
    }

    @Override
    public String[] outputUris() {
        return outputUris;
    }

    @Override
    public String[] canovaUris() {
        return canovaUris;
    }

    public static class Builder {

    }

}
