#import "CppMLCTensorData.h"

#import <MLCompute/MLCTensorData.h>

CppMLCTensorData::CppMLCTensorData()
  : self{NULL}
{
}

CppMLCTensorData::~CppMLCTensorData()
{
    [(id)self dealloc];
}

CppMLCTensorData::CppMLCTensorData(void* mlcTensorData)
  : self{(MLCTensorData*)mlcTensorData}
{
}

CppMLCTensorData::CppMLCTensorData(void* bytes, uint64_t length)
  : self{[MLCTensorData dataWithBytesNoCopy:bytes length:length]}
{
}

CppMLCTensorData::CppMLCTensorData(void const* bytes, uint64_t length)
    : self{[MLCTensorData dataWithImmutableBytesNoCopy:bytes length:length]}
{
}

auto CppMLCTensorData::getBytes() -> void*
{
    return ((MLCTensorData*)self).bytes;
}

auto CppMLCTensorData::getLength() -> uint64_t
{
    return ((MLCTensorData*)self).length;
}
