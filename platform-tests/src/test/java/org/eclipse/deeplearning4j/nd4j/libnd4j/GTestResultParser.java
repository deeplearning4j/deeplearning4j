/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.libnd4j;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Parser for Google Test XML output format.
 * Converts GTest XML results into a format that can be used with JUnit reporting.
 */
public class GTestResultParser {
    private static final Logger log = LoggerFactory.getLogger(GTestResultParser.class);

    /**
     * Parse a Google Test XML result file.
     *
     * @param xmlPath Path to the XML file
     * @return GTestResults containing all test suites and cases
     */
    public static GTestResults parseXmlFile(Path xmlPath) {
        if (!Files.exists(xmlPath)) {
            log.warn("XML result file not found: {}", xmlPath);
            return new GTestResults();
        }

        try (InputStream is = Files.newInputStream(xmlPath)) {
            return parseXml(is);
        } catch (Exception e) {
            log.error("Failed to parse XML result file: {}", xmlPath, e);
            return new GTestResults();
        }
    }

    /**
     * Parse Google Test XML from an input stream.
     */
    public static GTestResults parseXml(InputStream inputStream) {
        GTestResults results = new GTestResults();

        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            factory.setFeature("http://xml.org/sax/features/external-general-entities", false);
            factory.setFeature("http://xml.org/sax/features/external-parameter-entities", false);

            DocumentBuilder builder = factory.newDocumentBuilder();
            Document doc = builder.parse(inputStream);

            Element testSuitesElement = doc.getDocumentElement();
            if (testSuitesElement == null) {
                return results;
            }

            // Parse top-level attributes
            results.setName(testSuitesElement.getAttribute("name"));
            results.setTests(parseIntAttribute(testSuitesElement, "tests", 0));
            results.setFailures(parseIntAttribute(testSuitesElement, "failures", 0));
            results.setDisabled(parseIntAttribute(testSuitesElement, "disabled", 0));
            results.setErrors(parseIntAttribute(testSuitesElement, "errors", 0));
            results.setTime(parseDoubleAttribute(testSuitesElement, "time", 0.0));

            // Parse test suites
            NodeList suiteNodes = testSuitesElement.getElementsByTagName("testsuite");
            for (int i = 0; i < suiteNodes.getLength(); i++) {
                Element suiteElement = (Element) suiteNodes.item(i);
                GTestSuite suite = parseTestSuite(suiteElement);
                results.addTestSuite(suite);
            }

        } catch (Exception e) {
            log.error("Error parsing GTest XML", e);
        }

        return results;
    }

    /**
     * Parse a test suite element.
     */
    private static GTestSuite parseTestSuite(Element suiteElement) {
        GTestSuite suite = new GTestSuite();
        suite.setName(suiteElement.getAttribute("name"));
        suite.setTests(parseIntAttribute(suiteElement, "tests", 0));
        suite.setFailures(parseIntAttribute(suiteElement, "failures", 0));
        suite.setDisabled(parseIntAttribute(suiteElement, "disabled", 0));
        suite.setErrors(parseIntAttribute(suiteElement, "errors", 0));
        suite.setTime(parseDoubleAttribute(suiteElement, "time", 0.0));

        // Parse test cases
        NodeList caseNodes = suiteElement.getElementsByTagName("testcase");
        for (int i = 0; i < caseNodes.getLength(); i++) {
            Element caseElement = (Element) caseNodes.item(i);
            // Only parse direct children (not nested)
            if (caseElement.getParentNode() == suiteElement) {
                GTestCase testCase = parseTestCase(caseElement);
                suite.addTestCase(testCase);
            }
        }

        return suite;
    }

    /**
     * Parse a test case element.
     */
    private static GTestCase parseTestCase(Element caseElement) {
        GTestCase testCase = new GTestCase();
        testCase.setName(caseElement.getAttribute("name"));
        testCase.setClassName(caseElement.getAttribute("classname"));
        testCase.setStatus(caseElement.getAttribute("status"));
        testCase.setTime(parseDoubleAttribute(caseElement, "time", 0.0));
        testCase.setFile(caseElement.getAttribute("file"));
        testCase.setLine(parseIntAttribute(caseElement, "line", 0));

        // Check for failure
        NodeList failureNodes = caseElement.getElementsByTagName("failure");
        if (failureNodes.getLength() > 0) {
            Element failureElement = (Element) failureNodes.item(0);
            testCase.setFailure(true);
            testCase.setFailureMessage(failureElement.getAttribute("message"));
            testCase.setFailureType(failureElement.getAttribute("type"));
            testCase.setFailureDetails(failureElement.getTextContent());
        }

        // Check for skipped
        NodeList skippedNodes = caseElement.getElementsByTagName("skipped");
        if (skippedNodes.getLength() > 0) {
            testCase.setSkipped(true);
            Element skippedElement = (Element) skippedNodes.item(0);
            testCase.setSkipReason(skippedElement.getAttribute("message"));
        }

        return testCase;
    }

    /**
     * Parse an integer attribute with a default value.
     */
    private static int parseIntAttribute(Element element, String attrName, int defaultValue) {
        String value = element.getAttribute(attrName);
        if (value == null || value.isEmpty()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    /**
     * Parse a double attribute with a default value.
     */
    private static double parseDoubleAttribute(Element element, String attrName, double defaultValue) {
        String value = element.getAttribute(attrName);
        if (value == null || value.isEmpty()) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

}
