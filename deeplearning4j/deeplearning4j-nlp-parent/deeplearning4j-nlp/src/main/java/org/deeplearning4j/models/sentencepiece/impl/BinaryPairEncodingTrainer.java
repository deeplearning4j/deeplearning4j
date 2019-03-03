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

package org.deeplearning4j.models.sentencepiece.impl;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import lombok.var;
import org.deeplearning4j.models.sentencepiece.SubwordVocabulary;
import org.deeplearning4j.models.sentencepiece.impl.bpe.Position;
import org.deeplearning4j.models.sentencepiece.impl.bpe.Symbol;
import org.deeplearning4j.models.sentencepiece.interfaces.Trainer;
import org.nd4j.base.Preconditions;

import java.util.*;

@Slf4j
public class BinaryPairEncodingTrainer extends AbstractTrainer implements Trainer {
    protected Map<Long, Symbol> symbolsCache = new HashMap<>();
    protected Set<Symbol> activeSymbols = new HashSet<>();
    protected List<List<Symbol>> symbols = new ArrayList<>();

    // probably not needed in Java impl
    protected List<Symbol> allocated = new ArrayList<>();

    @Override
    public SubwordVocabulary buildVocabulary(Iterator<String> iterator) {
        return null;
    }

    protected void computeFrequency(Symbol symbol) {
        if (symbol.getFrequency() > 0) {  // if freq == 0, re-computation is required.
            return;
        }
        // Avoids double-count. ("AAA" => only count the first "AA").
        var prev_pos = new Position(-1, 0, 0);

        for (val it : symbol.getPositions()) {
            val pos = Position.decodePosition(it);
            // There are two same bigrams in "AAA", [AA] [AA], and we want to
            // remove the second one to avoid double counts.
            // If the right symbol in the first bigram and the left symbol in the
            // second bigram have the same position, (pos.left == prev_pos.right),
            // duplicated bigram exisit.
            // Also, symbols_[sid][left] and symbols_[sid]right] must store
            // the same symbols in symbol->left and symbols->right.
            if ((pos.getSid() == prev_pos.getSid() && pos.getLeft() == prev_pos.getRight()) ||
                    symbol.getLeft() != symbols.get((int) pos.getSid()).get((int) pos.getLeft()) ||
                            symbol.getRight() != symbols.get((int) pos.getSid()).get((int) pos.getRight())) {
                symbol.getPositions().remove(it);
                // Initializes prev_pos.
                // In "AAAA", the last "AA" can be counted.
                prev_pos.setSid(-1);
                prev_pos.setLeft(0);
            } else {
                symbol.incrementFrequency(sentences.get((int) pos.getSid()).getSecond());
                prev_pos = pos;
                ++it;
            }
        }
    }

    protected int getNextIndex(int sid, int index) {
        for (int i = index + 1; i < symbols.get(sid).size(); ++i) {
            if (symbols.get(sid).get(i) == null)
                continue;
            return i;
        }
        return -1;
    }

    protected int getPrevIndex(int sid, int index) {
        for (int i = index - 1; i >= 0; --i) {
            if (symbols.get(sid).get(i) == null)
                continue;
            return i;
        }
        return -1;
    }

    protected void addNewPair(int sid, int left, int right) {
        if (left == -1 || right == -1) return;
        val symbol = getPairSymbol(symbols.get(sid).get(left), symbols.get(sid).get(right));
        if (symbol != null) {
            activeSymbols.add(symbol);
            symbol.getPositions().add(Position.encodePosition(sid, left, right));
        }
    }

    protected void resetFreq(int sid, int left, int right, Symbol best) {
        if (left == -1 || right == -1)
            return;

        val symbol = getPairSymbol(symbols.get(sid).get(left), symbols.get(sid).get(right));
        if (symbol != null && symbol != best) {
            symbol.setFrequency(0);
        }
    }

    protected void updateActiveSymbols() {
        val symbols = new ArrayList<Symbol>();
        for (val symbol : symbolsCache.values()) {
            if (symbol.isBigram()) {
                computeFrequency(symbol);
                symbols.add(symbol);
            }
        }

        // At least kMinActiveSymbolsSize symbols must be in |active_symbols_|.
        val kMinActiveSymbolsSize = 1000;

        // Keeps top 5% frequent symbols.
        val kTopFrequentRatio = 0.05f;
        val size = Math.min(Math.max(kMinActiveSymbolsSize, (int)(symbolsCache.size() * kTopFrequentRatio)), symbols.size());

        //std::partial_sort(symbols.begin(), symbols.begin() + size, symbols.end(), [](Symbol *s1, Symbol *s2) { return s1->freq > s2->freq; });
        // FIXME: partial_sort
        Collections.sort(symbols, new Comparator<Symbol>() {
            @Override
            public int compare(Symbol o1, Symbol o2) {
                return Long.compare(o1.getFrequency(), o2.getFrequency());
            }
        });

        activeSymbols.clear();
        for (int e = 0; e < size; e++)
            activeSymbols.add(symbols.get(e));
    }


    protected Symbol getCharSymbol(int c) {
        return null;
    }

    protected Symbol getPairSymbol(Symbol left, Symbol right) {
        if (left == null || right == null || left.isUnknown() || right.isUnknown()) {
            return null;
        }

        val fp = Symbol.fingerprintCat(left.getFingerprint(), right.getFingerprint());
        if (symbolsCache.containsKey(fp))
            return symbolsCache.get(fp);

        Preconditions.checkArgument(!left.getChars().isEmpty(), "Left chars list is empty");
        Preconditions.checkArgument(!right.getChars().isEmpty(), "Right chars list is empty");

        val ut = new ArrayList<Integer>();
        for (val c : left.getChars())
            ut.add(c);

        for (val c : right.getChars())
            ut.add(c);

        // Do not make an invalid piece.
        if (!isValidSentencePiece(ut))
            return null;

        val s = Symbol.builder()
                .fingerprint(fp)
                .left(left)
                .right(right)
                .chars(ut)
                .build();

        allocated.add(s);

        Preconditions.checkArgument(!symbolsCache.containsKey(fp), "Duplicate key found!");
        symbolsCache.put(fp, s);

        return s;
    }
}
