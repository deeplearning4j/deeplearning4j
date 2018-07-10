/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.text.uima;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CAS;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.CasPool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Resource holder for uima
 * @author Adam Gibson
 *
 */
public class UimaResource {

    private AnalysisEngine analysisEngine;
    private CasPool casPool;
    private static final Logger log = LoggerFactory.getLogger(UimaResource.class);

    public UimaResource(AnalysisEngine analysisEngine) throws ResourceInitializationException {
        this.analysisEngine = analysisEngine;
        this.casPool = new CasPool(Runtime.getRuntime().availableProcessors(), analysisEngine);

    }

    public UimaResource(AnalysisEngine analysisEngine, CasPool casPool) {
        this.analysisEngine = analysisEngine;
        this.casPool = casPool;

    }


    public AnalysisEngine getAnalysisEngine() {
        return analysisEngine;
    }


    public void setAnalysisEngine(AnalysisEngine analysisEngine) {
        this.analysisEngine = analysisEngine;
    }


    public CasPool getCasPool() {
        return casPool;
    }


    public void setCasPool(CasPool casPool) {
        this.casPool = casPool;
    }


    /**
     * Use the given analysis engine and process the given text
     * You must release the return cas yourself
     * @param text the text to process
     * @return the processed cas
     */
    public CAS process(String text) {
        CAS cas = retrieve();
        if (cas == null)
            return null;

        cas.setDocumentText(text);
        try {
            analysisEngine.process(cas);
        } catch (AnalysisEngineProcessException e) {
            log.warn("Unable to process text " + text, e);
        }

        return cas;


    }


    public CAS retrieve() {
        CAS ret = casPool.getCas();
        try {
            return ret == null ? analysisEngine.newCAS() : ret;
        } catch (ResourceInitializationException e) {
            throw new RuntimeException(e);
        }
    }


    public void release(CAS cas) {
        casPool.releaseCas(cas);
    }



}
