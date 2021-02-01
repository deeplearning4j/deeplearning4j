/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.text.documentiterator;

import java.util.List;

/**
 * LabelAwareIterator wrapper which populates a LabelsSource while iterating.
 *
 * @author Benjamin Possolo
 */
public class LabelAwareIteratorWrapper implements LabelAwareIterator {

  private final LabelAwareIterator delegate;
  private final LabelsSource sink;
  
  public LabelAwareIteratorWrapper(LabelAwareIterator delegate, LabelsSource sink) {
    this.delegate = delegate;
    this.sink = sink;
  }

  @Override
  public boolean hasNext() {
    return delegate.hasNext();
  }

  @Override
  public boolean hasNextDocument() {
    return delegate.hasNextDocument();
  }

  @Override
  public LabelsSource getLabelsSource() {
    return sink;
  }

  @Override
  public LabelledDocument next() {
    return nextDocument();
  }

  @Override
  public void remove() {

  }

  @Override
  public LabelledDocument nextDocument() {
    LabelledDocument doc = delegate.nextDocument();
    List<String> labels = doc.getLabels();
    if (labels != null) {
      for (String label : labels) {
        sink.storeLabel(label);
      }
    }
    return doc;
  }

  @Override
  public void reset() {
    delegate.reset();
    sink.reset();
  }

  @Override
  public void shutdown() {}
}
