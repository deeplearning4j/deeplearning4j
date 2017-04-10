/*-*
 * Copyright © 2010-2015 Atilika Inc. and contributors (see CONTRIBUTORS.md)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.  A copy of the
 * License is distributed with this work in the LICENSE.md file.  You may
 * also obtain a copy of the License from
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.atilika.kuromoji.compile;

import com.atilika.kuromoji.buffer.BufferEntry;
import com.atilika.kuromoji.buffer.FeatureInfoMap;
import com.atilika.kuromoji.buffer.StringValueMapBuffer;
import com.atilika.kuromoji.buffer.WordIdMap;
import com.atilika.kuromoji.dict.DictionaryEntryBase;
import com.atilika.kuromoji.dict.GenericDictionaryEntry;
import com.atilika.kuromoji.dict.TokenInfoDictionary;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.TreeMap;

public abstract class TokenInfoDictionaryCompilerBase<T extends DictionaryEntryBase> implements Compiler {

    protected List<BufferEntry> bufferEntries = new ArrayList<>();
    protected FeatureInfoMap posInfo = new FeatureInfoMap();
    protected FeatureInfoMap otherInfo = new FeatureInfoMap();
    protected WordIdMapCompiler wordIdsCompiler = new WordIdMapCompiler();

    // optional list to collect the generic dictionary entries
    protected List<GenericDictionaryEntry> dictionaryEntries = null;

    private String encoding;
    private List<String> surfaces = new ArrayList<>();

    public TokenInfoDictionaryCompilerBase(String encoding) {
        this.encoding = encoding;
    }

    public void analyzeTokenInfo(InputStream input) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(input, encoding));
        String line;

        while ((line = reader.readLine()) != null) {
            T entry = parse(line);

            GenericDictionaryEntry dictionaryEntry = generateGenericDictionaryEntry(entry);

            posInfo.mapFeatures(dictionaryEntry.getPosFeatures());
        }
    }

    public void readTokenInfo(InputStream input) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(input, encoding));
        String line;
        int entryCount = posInfo.getEntryCount();

        while ((line = reader.readLine()) != null) {
            T entry = parse(line);

            GenericDictionaryEntry dictionaryEntry = generateGenericDictionaryEntry(entry);

            short leftId = dictionaryEntry.getLeftId();
            short rightId = dictionaryEntry.getRightId();
            short wordCost = dictionaryEntry.getWordCost();

            List<String> allPosFeatures = dictionaryEntry.getPosFeatures();

            List<Integer> posFeatureIds = posInfo.mapFeatures(allPosFeatures);

            List<String> featureList = dictionaryEntry.getFeatures();
            List<Integer> otherFeatureIds = otherInfo.mapFeatures(featureList);

            BufferEntry bufferEntry = new BufferEntry();
            bufferEntry.tokenInfo.add(leftId);
            bufferEntry.tokenInfo.add(rightId);
            bufferEntry.tokenInfo.add(wordCost);

            if (entriesFitInAByte(entryCount)) {
                List<Byte> posFeatureIdBytes = createPosFeatureIds(posFeatureIds);
                bufferEntry.posInfo.addAll(posFeatureIdBytes);
            } else {
                for (Integer posFeatureId : posFeatureIds) {
                    bufferEntry.tokenInfo.add(posFeatureId.shortValue());
                }
            }

            bufferEntry.features.addAll(otherFeatureIds);

            bufferEntries.add(bufferEntry);
            surfaces.add(dictionaryEntry.getSurface());

            if (dictionaryEntries != null) {
                dictionaryEntries.add(dictionaryEntry);
            }
        }
    }

    protected abstract GenericDictionaryEntry generateGenericDictionaryEntry(T entry);

    protected abstract T parse(String line);

    @Override
    public void compile() throws IOException {
        // TODO: Should call this method instead of write()
    }

    private boolean entriesFitInAByte(int entryCount) {
        return entryCount <= 0xff;
    }

    private List<Byte> createPosFeatureIds(List<Integer> posFeatureIds) {
        List<Byte> posFeatureIdBytes = new ArrayList<>();
        for (Integer posFeatureId : posFeatureIds) {
            posFeatureIdBytes.add(posFeatureId.byteValue());
        }
        return posFeatureIdBytes;
    }


    public InputStream combinedSequentialFileInputStream(File dir) throws FileNotFoundException {
        List<FileInputStream> fileInputStreams = new ArrayList<>();
        List<File> files = getCsvFiles(dir);

        for (File file : files) {
            fileInputStreams.add(new FileInputStream(file));
        }

        return new SequenceInputStream(Collections.enumeration(fileInputStreams));
    }

    public List<File> getCsvFiles(File dir) {
        FilenameFilter filter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".csv");
            }
        };

        ArrayList<File> files = new ArrayList<>();
        Collections.addAll(files, dir.listFiles(filter));
        Collections.sort(files);
        return files;
    }

    public void addMapping(int sourceId, int wordId) {
        wordIdsCompiler.addMapping(sourceId, wordId);
    }

    public List<String> getSurfaces() {
        return surfaces;
    }

    public void write(String directoryName) throws IOException {
        writeDictionary(directoryName + File.separator + TokenInfoDictionary.TOKEN_INFO_DICTIONARY_FILENAME);
        writeMap(directoryName + File.separator + TokenInfoDictionary.POS_MAP_FILENAME, posInfo);
        writeMap(directoryName + File.separator + TokenInfoDictionary.FEATURE_MAP_FILENAME, otherInfo);
        writeWordIds(directoryName + File.separator + TokenInfoDictionary.TARGETMAP_FILENAME);
    }


    protected void writeMap(String filename, FeatureInfoMap map) throws IOException {
        TreeMap<Integer, String> features = map.invert();

        StringValueMapBuffer mapBuffer = new StringValueMapBuffer(features);
        FileOutputStream fos = new FileOutputStream(filename);
        mapBuffer.write(fos);
    }

    protected void writeDictionary(String filename) throws IOException {
        TokenInfoBufferCompiler tokenInfoBufferCompiler =
                        new TokenInfoBufferCompiler(new FileOutputStream(filename), bufferEntries);
        tokenInfoBufferCompiler.compile();
    }

    protected void writeWordIds(String filename) throws IOException {
        wordIdsCompiler.write(new FileOutputStream(filename));
    }

    @Deprecated
    public WordIdMap getWordIdMap() throws IOException {
        File file = File.createTempFile("kuromoji-wordid-", ".bin");
        file.deleteOnExit();

        OutputStream output = new BufferedOutputStream(new FileOutputStream(file));
        wordIdsCompiler.write(output);
        output.close();

        InputStream input = new BufferedInputStream(new FileInputStream(file));
        WordIdMap wordIds = new WordIdMap(input);
        input.close();

        return wordIds;
    }

    @Deprecated
    public List<BufferEntry> getBufferEntries() {
        return bufferEntries;
    }

    @Deprecated
    public List<GenericDictionaryEntry> getDictionaryEntries() {
        return dictionaryEntries;
    }

    @Deprecated
    public void setDictionaryEntries(List<GenericDictionaryEntry> dictionaryEntries) {
        this.dictionaryEntries = dictionaryEntries;
    }
}
