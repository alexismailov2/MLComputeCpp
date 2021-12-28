#import "CppMLCTensorData.h"

#import <MLCompute/MLCTensorData.h>

#import <iostream>

CppMLCTensorData::CppMLCTensorData(void* mlcTensorData)
  : self{(MLCTensorData*)mlcTensorData}
{
    [(id)self retain];
}

CppMLCTensorData::CppMLCTensorData(void* bytes, uint64_t length)
  : self{[MLCTensorData dataWithBytesNoCopy:bytes length:length]}
{
}

CppMLCTensorData::CppMLCTensorData(void const* bytes, uint64_t length)
    : self{[MLCTensorData dataWithImmutableBytesNoCopy:bytes length:length]}
{
}

auto CppMLCTensorData::getBytes() const -> void const*
{
    return ((MLCTensorData*)self).bytes;
}

auto CppMLCTensorData::getLength() const -> uint64_t
{
    return ((MLCTensorData*)self).length;
}

std::ostream& operator<<(std::ostream& out, CppMLCTensorData const& tensorData)
{
    out << "CppMLCTensorData {\n" << "length: " << tensorData.getLength() << "\n}\n";
    return out;
}