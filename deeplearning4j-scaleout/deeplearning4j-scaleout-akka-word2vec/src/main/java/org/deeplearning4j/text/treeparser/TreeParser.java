package org.deeplearning4j.text.treeparser;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngine;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.cas.CAS;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.util.CasPool;
import org.cleartk.opennlp.tools.ParserAnnotator;
import org.cleartk.opennlp.tools.parser.DefaultOutputTypesHelper;
import org.cleartk.syntax.constituent.type.TopTreebankNode;
import org.cleartk.token.type.Sentence;
import org.cleartk.token.type.Token;
import org.cleartk.util.ParamUtil;
import org.deeplearning4j.rntn.Tree;
import org.deeplearning4j.text.annotator.PoStagger;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.StemmerAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;

import java.util.ArrayList;
import java.util.List;

/**
 * Tree parser
 * @author Adam Gibson
 */
public class TreeParser {

    private  AnalysisEngine parser;
    private AnalysisEngine tokenizer;
    private CasPool pool;

    public TreeParser(AnalysisEngine parser,AnalysisEngine tokenizer,CasPool pool) {
        this.parser = parser;
        this.tokenizer = tokenizer;
        this.pool = pool;
    }


    public TreeParser() throws Exception {
        if(parser == null) {
            parser = getParser();
        }
        if(tokenizer == null)
            tokenizer = getTokenizer();
        if(pool == null)
            pool = new CasPool(Runtime.getRuntime().availableProcessors() * 10,parser);


    }

    /**
     * Gets trees from text.
     * First a sentence segmenter is used to segment the training examples in to sentences.
     * Sentences are then turned in to trees and returned.
     * @param text the text to process
     * @param preProcessor the pre processor to use for pre processing sentences
     * @return the list of trees
     * @throws Exception
     */
    public List<Tree> getTrees(String text,SentencePreProcessor preProcessor)  throws Exception {
        CAS c = pool.getCas();
        c.setDocumentText(text);
        tokenizer.process(c);
        List<Tree> ret = new ArrayList<>();
        CAS c2 = pool.getCas();
        for(Sentence sentence : JCasUtil.select(c.getJCas(),Sentence.class)) {
            List<String> tokens = new ArrayList<>();
            for(Token t : JCasUtil.selectCovered(Token.class,sentence))
                tokens.add(t.getCoveredText());


            c2.setDocumentText(sentence.getCoveredText());
            tokenizer.process(c2);
            parser.process(c2);

            //build the tree based on this
            TopTreebankNode node = JCasUtil.selectSingle(c.getJCas(),TopTreebankNode.class);
            ret.add(TreeFactory.buildTree(node));


        }

        return ret;


    }
    /**
     * Gets trees from text.
     * First a sentence segmenter is used to segment the training examples in to sentences.
     * Sentences are then turned in to trees and returned.
     * @param text the text to process
     * @return the list of trees
     * @throws Exception
     */
    public List<Tree> getTrees(String text)  throws Exception {
        CAS c = pool.getCas();
        c.setDocumentText(text);
        tokenizer.process(c);
        List<Tree> ret = new ArrayList<>();
        CAS c2 = pool.getCas();
        for(Sentence sentence : JCasUtil.select(c.getJCas(),Sentence.class)) {
            List<String> tokens = new ArrayList<>();
            for(Token t : JCasUtil.selectCovered(Token.class,sentence))
                tokens.add(t.getCoveredText());


            c2.setDocumentText(sentence.getCoveredText());
            tokenizer.process(c2);
            parser.process(c2);

            //build the tree based on this
            TopTreebankNode node = JCasUtil.selectSingle(c.getJCas(),TopTreebankNode.class);
            ret.add(TreeFactory.buildTree(node));


        }

        return ret;


    }


    public static AnalysisEngine getTokenizer() throws Exception {
        return createEngine(
                createEngineDescription(
                        SentenceAnnotator.getDescription(),
                        TokenizerAnnotator.getDescription(),
                        PoStagger.getDescription("en"),
                        StemmerAnnotator.getDescription("English")

                )
        );
    }

    public static AnalysisEngine getParser() throws Exception {
        /*

         */
        return createEngine(
                createEngineDescription(
                        createEngineDescription(
                                ParserAnnotator.class,
                                ParserAnnotator.PARAM_USE_TAGS_FROM_CAS,
                                true,
                                ParserAnnotator.PARAM_PARSER_MODEL_PATH,
                                ParamUtil.getParameterValue(ParserAnnotator.PARAM_PARSER_MODEL_PATH, "/models/en-parser-chunking.bin"),
                                ParserAnnotator.PARAM_OUTPUT_TYPES_HELPER_CLASS_NAME,
                                DefaultOutputTypesHelper.class.getName())));


    }

}
