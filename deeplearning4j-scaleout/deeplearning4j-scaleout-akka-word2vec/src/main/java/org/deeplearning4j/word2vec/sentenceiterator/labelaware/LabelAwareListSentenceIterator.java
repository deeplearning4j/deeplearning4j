package org.deeplearning4j.word2vec.sentenceiterator.labelaware;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class LabelAwareListSentenceIterator implements LabelAwareSentenceIterator {


    private String delimiter;
    private int labelPosition;
    private int textPosition;
    private List<String> lines;
    private int currPosition;
    private List<String> labels;
    private List<String> text;
    private SentencePreProcessor sentencePreProcessor;

    /**
     * Initializes the sentence iterator with the given args, note that this will close the input stream for you
     * @param is the input stream (this will be closed)
     * @param delimiter the delimiter (tab comma,...)
     * @param labelPosition the position of the label on each line
     * @param textPosition the position of the text on each line
     * @throws IOException
     */
    public LabelAwareListSentenceIterator(InputStream is,String delimiter,int labelPosition,int textPosition) throws IOException {
        this.delimiter = delimiter;
        this.labelPosition = labelPosition;
        this.textPosition = textPosition;
        lines = IOUtils.readLines(is);
        initLists();

        is.close();

    }

    /**
     * Same as calling (is,\t,0,1)
     * @param is the input stream to read lines from
     * @throws IOException
     */
    public LabelAwareListSentenceIterator(InputStream is) throws IOException {
        this(is,"\t",0,1);
    }

    private void initLists() {
        labels = new ArrayList<>(lines.size());
        text = new ArrayList<>(lines.size());

        for(String line : lines) {
            String[] split = line.split(delimiter);
            String label = split[labelPosition];
            String text = split[textPosition];
            labels.add(label);
            this.text.add(text);
        }
    }

    /**
     * Returns the current label for nextSentence()
     *
     * @return the label for nextSentence()
     */
    @Override
    public synchronized String currentLabel() {
        return labels.get(currPosition);
    }

    /**
     * Gets the next sentence or null
     * if there's nothing left (Do yourself a favor and
     * check hasNext() )
     *
     * @return the next sentence in the iterator
     */
    @Override
    public synchronized String nextSentence() {
        String ret = text.get(currPosition);

        if(sentencePreProcessor != null)
            ret = sentencePreProcessor.preProcess(ret);
        currPosition++;

        return ret;
    }

    /**
     * Same idea as {@link java.util.Iterator}
     *
     * @return whether there's anymore sentences left
     */
    @Override
    public synchronized  boolean hasNext() {
        return currPosition < text.size();
    }

    /**
     * Resets the iterator to the beginning
     */
    @Override
    public void reset() {
        currPosition = 0;

    }

    /**
     * Allows for any finishing (closing of input streams or the like)
     */
    @Override
    public void finish() {
         //no op
    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return sentencePreProcessor;
    }

    @Override
    public void setPreProcessor(SentencePreProcessor preProcessor) {
        this.sentencePreProcessor = preProcessor;
    }
}
