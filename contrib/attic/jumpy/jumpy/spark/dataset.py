#  /* ******************************************************************************
#   *
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   * License for the specific language governing permissions and limitations
#   * under the License.
#   *
#   * SPDX-License-Identifier: Apache-2.0
#   ******************************************************************************/

################################################################################
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

from ..ndarray import ndarray
from ..java_classes import JDataset


class Dataset(object):

    def __init__(self, features, labels, features_mask=None, labels_mask=None):
        self.features = ndarray(features)
        self.labels = ndarray(labels)
        if features_mask is None:
            self.features_mask = None
        else:
            self.features_mask = ndarray(features_mask)
        if labels_mask is None:
            self.labels_mask = None
        else:
            self.labels_mask = ndarray(labels_mask)

    def to_java(self):
        return JDataset(self.features.array, self.labels.array, self.features_mask, self.labels_mask)

    def __getstate__(self):
        return [self.features.numpy(),
                self.labels.numpy(),
                self.features_mask.numpy() if self.features_mask is not None else None,
                self.labels_mask.numpy() if self.labels_mask is not None else None]

    def __setstate__(self, state):
        ds = Dataset(*state)
        self.features = ds.features
        self.labels = ds.labels
        self.features_mask = ds.features_mask
        self.labels_mask = ds.labels_mask
