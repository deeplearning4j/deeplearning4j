/*
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.arbiter.optimize;

import org.arbiter.optimize.api.CandidateGenerator;
import org.arbiter.optimize.candidategenerator.GridSearchCandidateGenerator;
import org.junit.Test;

import static org.junit.Assert.*;

public class TestGridSearch {

    @Test
    public void testIndexing(){

        int[] nValues = {2,3};
        int prod = 2*3;
        double[][] expVals = new double[][]{
                {0.0,0.0},
                {1.0,0.0},
                {0.0,0.5},
                {1.0,0.5},
                {0.0,1.0},
                {1.0,1.0}
        };
        for( int i=0; i<prod; i++ ){
            double[] out = GridSearchCandidateGenerator.indexToValues(nValues,i,prod);
            double[] exp = expVals[i];
            assertArrayEquals(exp,out,1e-4);
        }
    }

    @Test
    public void testGeneration() throws Exception {

        //Define configuration:
        CandidateGenerator<TestRandomSearch.BraninConfig> candidateGenerator =
                new GridSearchCandidateGenerator<>(new TestRandomSearch.BraninSpace(), 4,
                        GridSearchCandidateGenerator.Mode.Sequential);

        //Check sequential:
        double[] expValuesFirst = {-5,0,5,10};  //Range: -5 to +10, with 4 values
        double[] expValuesSecond = {0,5,10,15}; //Range: 0 to +15, with 4 values
        for( int i=0; i<4*4; i++ ){
            TestRandomSearch.BraninConfig conf = candidateGenerator.getCandidate().getValue();
            double expF = expValuesFirst[i%4];  //Changes most rapidly
            double expS = expValuesSecond[i/4];

            double actF = conf.getX1();
            double actS = conf.getX2();

            assertEquals(expF,actF,1e-4);
            assertEquals(expS,actS,1e-4);
        }

        //Check random order. specifically: check that all values are generated, in some order
        double[][] orderedOutput = new double[16][2];
        for( int i=0; i<expValuesFirst.length; i++ ){
            for( int j=0; j<expValuesSecond.length; j++ ){
                orderedOutput[4*j+i][0] = expValuesFirst[i];
                orderedOutput[4*j+i][1] = expValuesSecond[j];
            }
        }

        candidateGenerator = new GridSearchCandidateGenerator<>(new TestRandomSearch.BraninSpace(), 4,
                GridSearchCandidateGenerator.Mode.RandomOrder);
        boolean[] seen = new boolean[16];
        int seenCount = 0;
        for( int i=0; i<4*4; i++ ){
            assertTrue(candidateGenerator.hasMoreCandidates());
            TestRandomSearch.BraninConfig config = candidateGenerator.getCandidate().getValue();
            double x1 = config.getX1();
            double x2 = config.getX2();
            //Work out which of the values this is...
            boolean matched = false;
            for( int j=0; j<16; j++ ){
                if(Math.abs(orderedOutput[j][0] - x1)<1e-5 && Math.abs(orderedOutput[j][1]-x2) < 1e-5){
                    matched = true;
                    if(seen[j]) fail("Same candidate generated multiple times");
                    seen[j] = true;
                    seenCount++;
                    break;
                }
            }
            assertTrue("Candidate " + x1 + ", " + x2 + " not found; invalid?",matched);
        }
        assertFalse(candidateGenerator.hasMoreCandidates());
        assertEquals(16,seenCount);
    }

}
