//
// @author raver119@gmail.com
//

#include <pointercast.h>
#include <array/ShapeList.h>

namespace nd4j {
    //ShapeList::ShapeList(bool autoRemovable) {
//        _autoremovable = autoRemovable;
//    }

    ShapeList::ShapeList(Nd4jLong* shape) {
        if (shape != nullptr)
            _shapes.push_back(shape);
    }

    ShapeList::~ShapeList() {
        if (_autoremovable)
            destroy();
    }

    ShapeList::ShapeList(std::initializer_list<Nd4jLong*> shapes) {
        for (auto v:shapes)
            _shapes.push_back(v);
    }

    ShapeList::ShapeList(std::initializer_list<Nd4jLong*> shapes, bool isWorkspace) : ShapeList(shapes){
        _workspace = isWorkspace;
    }

    ShapeList::ShapeList(std::vector<Nd4jLong*>& shapes) {
        for (auto v:shapes)
            _shapes.push_back(v);
    }

    std::vector<Nd4jLong*>* ShapeList::asVector() {
        return &_shapes;
    }

    void ShapeList::destroy() {
        if (!_workspace)
            for (auto v:_shapes)
                if(v != nullptr)
                    delete[] v;
    }

    int ShapeList::size() {
        return (int) _shapes.size();
    }

    Nd4jLong* ShapeList::at(int idx) {
        return _shapes.at(idx);
    }

    void ShapeList::push_back(Nd4jLong *shape) {
        _shapes.push_back(shape);
    }

    void ShapeList::push_back(std::vector<Nd4jLong>& shape) {
        int dLen = shape::shapeInfoLength(shape.at(0));

        if (shape.size() != dLen)
            throw std::runtime_error("Bad shape was passed in");

        auto nShape = new Nd4jLong[dLen];
        std::memcpy(nShape, shape.data(), shape::shapeInfoByteLength(shape.at(0)));

        _shapes.push_back(nShape);
    }

    void ShapeList::detach() {
        for (int e = 0; e < _shapes.size(); e++) {
            _shapes[e] = shape::detachShape(_shapes[e]);
        }
        _workspace = false;
    }
}