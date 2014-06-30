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
import org.cleartk.util.ParamUtil;
import org.deeplearning4j.text.annotator.PoStagger;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.StemmerAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;

/**
 * Tree parser
 * @author Adam Gibson
 */
public class TreeParser {

    private  AnalysisEngine parser;
    private CasPool pool;

    public TreeParser(AnalysisEngine parser,CasPool pool) {
        this.parser = parser;
        this.pool = pool;
    }


    public TreeParser() throws Exception {
        if(parser == null) {
            parser = getParser();
            pool = new CasPool(Runtime.getRuntime().availableProcessors() * 10,parser);
        }

    }



    public TopTreebankNode getTree(String text)  throws Exception {
        CAS c = pool.getCas();
        c.setDocumentText(text);
        parser.process(c.getJCas());
        TopTreebankNode node = JCasUtil.selectSingle(c.getJCas(),TopTreebankNode.class);
        return node;


    }



    public static AnalysisEngine getParser() throws Exception {
        /*

         */
        return createEngine(
                createEngineDescription(
                SentenceAnnotator.getDescription(),
                TokenizerAnnotator.getDescription(),
                PoStagger.getDescription("en"),
                StemmerAnnotator.getDescription("English"),
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
