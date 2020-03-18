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

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.07.2018
//

#include <helpers/OpArgsHolder.h>


namespace sd {

////////////////////////////////////////////////////////////////////////
// default constructor
OpArgsHolder::OpArgsHolder() {

	_inArrs = std::vector<NDArray*>();
    _tArgs  = std::vector<double>();
    _iArgs  = std::vector<Nd4jLong>();
    _bArgs  = std::vector<bool>();

    _isArrAlloc = std::vector<bool>();

    _numInArrs = 0;
    _numTArgs  = 0;
    _numIArgs  = 0;
    _numBArgs  = 0;
}

////////////////////////////////////////////////////////////////////////
// copy constructor
OpArgsHolder::OpArgsHolder(const OpArgsHolder& other) {

	throw std::runtime_error("OpArgsHolder::OpArgsHolder copy constructor: don't use me !");
}


////////////////////////////////////////////////////////////////////////
// constructor
OpArgsHolder::OpArgsHolder(const std::vector<NDArray*>& inArrs,
			 			   const std::vector<double>& tArgs,
			 			   const std::vector<Nd4jLong>& iArgs,
			 			   const std::vector<bool>& bArgs) {
	_inArrs = inArrs;
    _tArgs  = tArgs;
    _iArgs  = iArgs;
    _bArgs  = bArgs;

    _isArrAlloc = std::vector<bool>();

    _numInArrs = _inArrs.size();
    _numTArgs  = _tArgs.size();
    _numIArgs  = _iArgs.size();
    _numBArgs  = _bArgs.size();
}

////////////////////////////////////////////////////////////////////////
// move constructor
OpArgsHolder::OpArgsHolder(OpArgsHolder&& other) noexcept: _inArrs(std::move(other._inArrs)),
												 		   _tArgs(std::move(other._tArgs)),
												  		   _iArgs(std::move(other._iArgs)),
												  		   _bArgs(std::move(other._bArgs)),
												  		   _isArrAlloc(std::move(other._isArrAlloc))  {

	other._isArrAlloc = std::vector<bool>();

    _numInArrs = _inArrs.size();
    _numTArgs  = _tArgs.size();
    _numIArgs  = _iArgs.size();
    _numBArgs  = _bArgs.size();
}

////////////////////////////////////////////////////////////////////////
// assignment operator
OpArgsHolder& OpArgsHolder::operator=(const OpArgsHolder& other) {

    throw std::runtime_error("OpArgsHolder::OpArgsHolder assignment operator: don't use me !");
}


////////////////////////////////////////////////////////////////////////
// move assignment operator
OpArgsHolder& OpArgsHolder::operator=(OpArgsHolder&& other) noexcept {

	if (this == &other)
        return *this;

    for (int i = 0; i < _isArrAlloc.size(); ++i)		// delete arrays if necessary
		if(_isArrAlloc[i])
			delete _inArrs[i];

	_inArrs 	= std::move(other._inArrs);
	_tArgs  	= std::move(other._tArgs);
	_iArgs  	= std::move(other._iArgs);
	_bArgs  	= std::move(other._bArgs);
	_isArrAlloc = std::move(other._isArrAlloc);

	other._isArrAlloc = std::vector<bool>();

	_numInArrs = _inArrs.size();
    _numTArgs  = _tArgs.size();
    _numIArgs  = _iArgs.size();
    _numBArgs  = _bArgs.size();

    return *this;
}

////////////////////////////////////////////////////////////////////////
OpArgsHolder OpArgsHolder::createArgsHolderForBP(const std::vector<NDArray*>& inGradArrs, const bool isInPlace) const {

	const int numInGradArrs = inGradArrs.size();

	OpArgsHolder result(std::vector<NDArray*>(_numInArrs + numInGradArrs, nullptr), _tArgs, _iArgs);

	if(isInPlace)
		result._isArrAlloc = std::vector<bool>(_numInArrs + numInGradArrs, false);

	for (int i = 0; i < _numInArrs; ++i) {

		if(isInPlace) {
			result._inArrs[i] = new NDArray(*_inArrs[i]);		// make copy
			result._isArrAlloc[i] = true;
		}
		else
			result._inArrs[i] = _inArrs[i];
	}

	// input gradients
	for (int i = 0; i < numInGradArrs; ++i)
		result._inArrs[_numInArrs + i] = inGradArrs[i];

	return result;
}

////////////////////////////////////////////////////////////////////////
// default destructor
OpArgsHolder::~OpArgsHolder() noexcept {

	for (int i = 0; i < _isArrAlloc.size(); ++i)
		if(_isArrAlloc[i])
			delete _inArrs[i];
}

}


