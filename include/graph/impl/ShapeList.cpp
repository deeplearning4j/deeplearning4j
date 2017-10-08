//
// @author raver119@gmail.com
//

#include <pointercast.h>
#include <graph/ShapeList.h>

namespace nd4j {
    ShapeList::ShapeList(int* shape) {
        if (shape != nullptr)
            _shapes.push_back(shape);
    }

    ShapeList::ShapeList(std::initializer_list<int*> shapes) {
        for (auto v:shapes)
            _shapes.push_back(v);
    }

    ShapeList::ShapeList(std::vector<int*>& shapes) {
        for (auto v:shapes)
            _shapes.push_back(v);
    }

    std::vector<int*>* ShapeList::asVector() {
        return &_shapes;
    }

    void ShapeList::destroy() {
        for (auto v:_shapes)
            delete[] v;
    }

    int ShapeList::size() {
        return (int) _shapes.size();
    }

    int* ShapeList::at(int idx) {
        return _shapes.at(idx);
    }

    void ShapeList::push_back(int *shape) {
        _shapes.push_back(shape);
    }

    void ShapeList::push_back(std::vector<int>& shape) {
        int dLen = shape::shapeInfoLength(shape.at(0));

        if (shape.size() != dLen)
            throw "Bad shape was passed in";

        auto nShape = new int[dLen];
        std::memcpy(nShape, shape.data(), shape::shapeInfoByteLength(shape.at(0)));

        _shapes.push_back(nShape);
    }
}