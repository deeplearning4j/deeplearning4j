//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.07.2018
//

#ifndef LIBND4J_OPARGSHOLDER_H
#define LIBND4J_OPARGSHOLDER_H


#include <NDArray.h>

namespace nd4j {
 
template<typename T>
class OpArgsHolder {

private: 
	std::vector<NDArray<T>*> _inArrs = std::vector<NDArray<T>*>();
    std::vector<T>           _tArgs  = std::vector<T>();
    std::vector<Nd4jLong>    _iArgs  = std::vector<int>();

    int _numInArrs = _inArrs.size();
    int _numTArgs  = _tArgs.size();
    int _numIArgs  = _iArgs.size();

    std::vector<bool> _isArrAlloc = std::vector<bool>();

public:

	OpArgsHolder(const std::vector<NDArray<T>*>& inArrs = std::vector<NDArray<T>*>(), const std::vector<T>& tArgs = std::vector<T>(), const std::vector<Nd4jLong>& iArgs = std::vector<Nd4jLong>())
    			: _inArrs(inArrs), _tArgs(tArgs), _iArgs(iArgs) { }

    const std::vector<NDArray<T>*>& getInArrs() const
    {return _inArrs; }

    const std::vector<T>& getTArgs() const
    {return _tArgs; }

    const std::vector<Nd4jLong>& getIArgs() const
    {return _iArgs; }

    const std::vector<bool>& getAllocInfo() const
    {return _isArrAlloc; }

    int getNumInArrs() const
    {return _numInArrs; }

    int getNumTArgs() const
    {return _numTArgs; }

    int getNumIArgs() const
    {return _numIArgs; }

    OpArgsHolder<T> createArgsHolderForBP(const std::vector<NDArray<T>*>& inGradArrs, const bool isInPlace = false) const;

    ~OpArgsHolder() noexcept; 
    
};





}

#endif //LIBND4J_OPARGSHOLDER_H
