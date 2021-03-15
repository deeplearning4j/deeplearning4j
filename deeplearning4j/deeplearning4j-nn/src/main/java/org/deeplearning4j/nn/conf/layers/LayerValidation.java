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

package org.deeplearning4j.nn.conf.layers;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class LayerValidation {

	private LayerValidation() {
	}

	/**
	 * Asserts that the layer nIn and nOut values are set for the layer
	 *
	 * @param layerType  Type of layer ("DenseLayer", etc)
	 * @param layerName  Name of the layer (may be null if not set)
	 * @param layerIndex Index of the layer
	 * @param nIn        nIn value
	 * @param nOut       nOut value
	 */
	public static void assertNInNOutSet(final String layerType, String layerName, final long layerIndex, final long nIn,
			final long nOut) {
		if (nIn <= 0 || nOut <= 0) {

			if (layerName == null)
				layerName = "(name not set)";
			throw new DL4JInvalidConfigException(layerType + " (index=" + layerIndex + ", name=" + layerName + ") nIn="
					+ nIn + ", nOut=" + nOut + "; nIn and nOut must be > 0");
		}
	}

	/**
	 * Asserts that the layer nOut value is set for the layer
	 *
	 * @param layerType  Type of layer ("DenseLayer", etc)
	 * @param layerName  Name of the layer (may be null if not set)
	 * @param layerIndex Index of the layer
	 * @param nOut       nOut value
	 */
	public static void assertNOutSet(final String layerType, String layerName, final long layerIndex, final long nOut) {
		if (nOut <= 0) {

			if (layerName == null)
				layerName = "(name not set)";
			throw new DL4JInvalidConfigException(layerType + " (index=" + layerIndex + ", name=" + layerName + ") nOut="
					+ nOut + "; nOut must be > 0");
		}
	}

	public static void generalValidation(final String layerName, final Layer layer, final IDropout iDropout,
			final List<Regularization> regularization, final List<Regularization> regularizationBias,
			final List<LayerConstraint> allParamConstraints, final List<LayerConstraint> weightConstraints,
			final List<LayerConstraint> biasConstraints) {

		if (layer != null) {

			if (layer instanceof BaseLayer) {

				final BaseLayer bLayer = (BaseLayer) layer;
				configureBaseLayer(layerName, bLayer, iDropout, regularization, regularizationBias);
			} else if (layer instanceof FrozenLayer && ((FrozenLayer) layer).getLayer() instanceof BaseLayer) {

				final BaseLayer bLayer = (BaseLayer) ((FrozenLayer) layer).getLayer();
				configureBaseLayer(layerName, bLayer, iDropout, regularization, regularizationBias);
			} else if (layer instanceof Bidirectional) {

				final Bidirectional l = (Bidirectional) layer;
				generalValidation(layerName, l.getFwd(), iDropout, regularization, regularizationBias,
						allParamConstraints, weightConstraints, biasConstraints);
				generalValidation(layerName, l.getBwd(), iDropout, regularization, regularizationBias,
						allParamConstraints, weightConstraints, biasConstraints);
			}

			if (layer.getConstraints() == null || layer.constraints.isEmpty()) {

				final List<LayerConstraint> allConstraints = new ArrayList<>();
				if (allParamConstraints != null && !layer.initializer().paramKeys(layer).isEmpty()) {

					for (final LayerConstraint c : allConstraints) {

						final LayerConstraint c2 = c.clone();
						c2.setParams(new HashSet<>(layer.initializer().paramKeys(layer)));
						allConstraints.add(c2);
					}
				}

				if (weightConstraints != null && !layer.initializer().weightKeys(layer).isEmpty()) {

					for (final LayerConstraint c : weightConstraints) {

						final LayerConstraint c2 = c.clone();
						c2.setParams(new HashSet<>(layer.initializer().weightKeys(layer)));
						allConstraints.add(c2);
					}
				}

				if (biasConstraints != null && !layer.initializer().biasKeys(layer).isEmpty()) {

					for (final LayerConstraint c : biasConstraints) {

						final LayerConstraint c2 = c.clone();
						c2.setParams(new HashSet<>(layer.initializer().biasKeys(layer)));
						allConstraints.add(c2);
					}
				}

				if (!allConstraints.isEmpty()) {

					layer.setConstraints(allConstraints);
				} else {

					layer.setConstraints(null);
				}
			}
		}
	}

	private static void configureBaseLayer(final String layerName, final BaseLayer bLayer, final IDropout iDropout,
			final List<Regularization> regularization, final List<Regularization> regularizationBias) {
		if (regularization != null && !regularization.isEmpty()) {

			final List<Regularization> bLayerRegs = bLayer.getRegularization();
			if (bLayerRegs == null || bLayerRegs.isEmpty()) {

				bLayer.setRegularization(regularization);
			} else {

				boolean hasL1 = false;
				boolean hasL2 = false;
				final List<Regularization> regContext = regularization;
				for (final Regularization reg : bLayerRegs) {

					if (reg instanceof L1Regularization) {

						hasL1 = true;
					} else if (reg instanceof L2Regularization) {

						hasL2 = true;
					}
				}
				for (final Regularization reg : regContext) {

					if (reg instanceof L1Regularization) {

						if (!hasL1)
							bLayerRegs.add(reg);
					} else if (reg instanceof L2Regularization) {

						if (!hasL2)
							bLayerRegs.add(reg);
					} else
						bLayerRegs.add(reg);
				}
			}
		}
		if (regularizationBias != null && !regularizationBias.isEmpty()) {

			final List<Regularization> bLayerRegs = bLayer.getRegularizationBias();
			if (bLayerRegs == null || bLayerRegs.isEmpty()) {

				bLayer.setRegularizationBias(regularizationBias);
			} else {

				boolean hasL1 = false;
				boolean hasL2 = false;
				final List<Regularization> regContext = regularizationBias;
				for (final Regularization reg : bLayerRegs) {

					if (reg instanceof L1Regularization) {

						hasL1 = true;
					} else if (reg instanceof L2Regularization) {

						hasL2 = true;
					}
				}
				for (final Regularization reg : regContext) {

					if (reg instanceof L1Regularization) {

						if (!hasL1)
							bLayerRegs.add(reg);
					} else if (reg instanceof L2Regularization) {

						if (!hasL2)
							bLayerRegs.add(reg);
					} else
						bLayerRegs.add(reg);
				}
			}
		}

		if (bLayer.getIDropout() == null) {

			bLayer.setIDropout(iDropout);
		}
	}
}
