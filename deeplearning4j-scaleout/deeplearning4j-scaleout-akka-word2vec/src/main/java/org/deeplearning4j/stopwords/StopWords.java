package org.deeplearning4j.stopwords;

import java.io.IOException;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.springframework.core.io.ClassPathResource;

import akka.event.slf4j.Logger;

/**
 * Loads stop words from the class path
 *
 * @author Adam Gibson
 *
 */
public class StopWords
{

    private static List<String> stopWords;

    @Deprecated
    @SuppressWarnings("unchecked")
    public static List<String> getStopWords()
    {

        try {
            if (stopWords == null) {
                stopWords = IOUtils.readLines(new ClassPathResource("/stopwords").getInputStream());
            }
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
        return stopWords;
    }

    @SuppressWarnings("unchecked")
    public static List<String> getStopWords(String stopwordfile)
    {
        if (stopWords == null) {
            try {
                stopWords = IOUtils
                        .readLines(new ClassPathResource(stopwordfile).getInputStream());
            }
            catch (IOException e) {
                Logger.root().error(
                        String.format("Unable to read stopword file '%s'.", stopwordfile));
                throw new RuntimeException(e);
            }
        }
        return stopWords;
    }

}
