#import "CppMLCTensorDescriptor.h"

#import <MLCompute/MLCTensorDescriptor.h>

namespace {
   auto toNSArray(std::vector<uint32_t> const& vector) -> NSArray<NSNumber*>* {
       id ns = [NSMutableArray new];
       std::for_each(vector.begin(), vector.end(), ^(uint32_t item) {
           [ns addObject:[NSNumber numberWithInteger:(NSInteger)item]];
       });
       return ns;
   }

   auto toNative(eMLCDataType dataType) -> MLCDataType {
       switch(dataType) {
           case eMLCDataType::Invalid: return MLCDataTypeInvalid;
           case eMLCDataType::Float32: return MLCDataTypeFloat32;
           case eMLCDataType::Boolean: return MLCDataTypeBoolean;
           case eMLCDataType::Int64: return MLCDataTypeInt64;
           case eMLCDataType::Int32: return MLCDataTypeInt32;
           case eMLCDataType::Count: return MLCDataTypeCount;
           default: return MLCDataTypeInvalid;
       }
   }

    auto MLCDataTypeToCpp(MLCDataType dataType) -> eMLCDataType {
        switch(dataType) {
            case MLCDataTypeInvalid: return eMLCDataType::Invalid;
            case MLCDataTypeFloat32: return eMLCDataType::Float32;
            case MLCDataTypeBoolean: return eMLCDataType::Boolean;
            case MLCDataTypeInt64: return eMLCDataType::Int64;
            case MLCDataTypeInt32: return eMLCDataType::Int32;
            case MLCDataTypeCount: return eMLCDataType::Count;
            default: return eMLCDataType::Invalid;
        }
    }
}

//auto
//CppMLCTensorDescriptor::ConvolutionWeightsDescriptor(uint32_t width,
//                                                     uint32_t height,
//                                                     uint32_t inputFeatureCount,
//                                                     uint32_t outputFeatureCount,
//                                                     eMLCDataType dataType) -> CppMLCTensorDescriptor
//{
//}
//
//auto
//CppMLCTensorDescriptor::ConvolutionBiasesDescriptor(uint32_t featureChannelCount,
//                                                    eMLCDataType dataType) -> CppMLCTensorDescriptor
//{
//}

CppMLCTensorDescriptor::CppMLCTensorDescriptor(std::vector<uint32_t> const& shape,
                                               eMLCDataType dataType)
    : self{[MLCTensorDescriptor descriptorWithShape:toNSArray(shape)
                                           dataType:toNative(dataType)]}
{
}

CppMLCTensorDescriptor::CppMLCTensorDescriptor(std::vector<uint32_t> const& shape,
                                               std::vector<uint32_t> const& sequenceLengths,
                                               bool sortedSequences,
                                               eMLCDataType dataType)
    : self{[MLCTensorDescriptor descriptorWithShape:toNSArray(shape)
                                    sequenceLengths:toNSArray(sequenceLengths)
                                    sortedSequences:(sortedSequences) ? YES : NO
                                           dataType:toNative(dataType)]}
{
}

CppMLCTensorDescriptor::CppMLCTensorDescriptor(uint32_t width,
                                               uint32_t height,
                                               uint32_t featureChannels,
                                               uint32_t batchSize,
                                               eMLCDataType dataType)
    : self{[MLCTensorDescriptor descriptorWithWidth:(NSUInteger)width
                                             height:(NSUInteger)height
                                featureChannelCount:(NSUInteger)featureChannels
                                          batchSize:(NSUInteger)batchSize
                                           dataType:toNative(dataType)]}
{
}

CppMLCTensorDescriptor::CppMLCTensorDescriptor(uint32_t width,
                                               uint32_t height,
                                               uint32_t featureChannels,
                                               uint32_t batchSize)
    : self{[MLCTensorDescriptor descriptorWithWidth:(NSUInteger)width
                                             height:(NSUInteger)height
                                featureChannelCount:(NSUInteger)featureChannels
                                          batchSize:(NSUInteger)batchSize]}
{
}

CppMLCTensorDescriptor::~CppMLCTensorDescriptor() {
    [(id)self dealloc];
}

CppMLCTensorDescriptor::CppMLCTensorDescriptor(void* tensorDescriptor)
    : self{tensorDescriptor}
{
}

auto CppMLCTensorDescriptor::getDataType() -> eMLCDataType {
    return MLCDataTypeToCpp(((MLCTensorDescriptor*)self).dataType);
}

auto CppMLCTensorDescriptor::getDimensionsCount() -> uint32_t {
    return (uint32_t)(((MLCTensorDescriptor*)self).dimensionCount);
}

auto CppMLCTensorDescriptor::getShape() -> std::vector<uint32_t> {
    __block std::vector<uint32_t> vectorList;
    vectorList.reserve([((MLCTensorDescriptor*)self).shape count]);
    [((MLCTensorDescriptor*)self).shape enumerateObjectsUsingBlock:^(NSNumber* _Nonnull obj, NSUInteger idx, BOOL* _Nonnull stop) {
        vectorList.push_back((uint32_t)obj.unsignedIntegerValue);
    }];
    return vectorList;
}

auto CppMLCTensorDescriptor::getStride() -> std::vector<uint32_t> {
    __block std::vector<uint32_t> vectorList;
    vectorList.reserve([((MLCTensorDescriptor*)self).stride count]);
    [((MLCTensorDescriptor*)self).stride enumerateObjectsUsingBlock:^(NSNumber* _Nonnull obj, NSUInteger idx, BOOL* _Nonnull stop) {
        vectorList.push_back((uint32_t)obj.unsignedIntegerValue);
    }];
    return vectorList;
}

auto CppMLCTensorDescriptor::getTensorAllocationSizeInBytes() -> uint64_t {
    return (uint64_t)((MLCTensorDescriptor*)self).tensorAllocationSizeInBytes;
}

auto CppMLCTensorDescriptor::getSequenceLengths() -> std::vector<uint32_t> {
    __block std::vector<uint32_t> vectorList;
    vectorList.reserve([((MLCTensorDescriptor*)self).sequenceLengths count]);
    [((MLCTensorDescriptor*)self).sequenceLengths enumerateObjectsUsingBlock:^(NSNumber* _Nonnull obj, NSUInteger idx, BOOL* _Nonnull stop) {
        vectorList.push_back((uint32_t)obj.unsignedIntegerValue);
    }];
    return vectorList;
}

bool CppMLCTensorDescriptor::isSortedSequences() {
    return ((MLCTensorDescriptor*)self).sortedSequences == YES;
}

auto CppMLCTensorDescriptor::getBatchSizePerSequenceStep() -> std::vector<uint32_t> {
    __block std::vector<uint32_t> vectorList;
    vectorList.reserve([((MLCTensorDescriptor*)self).batchSizePerSequenceStep count]);
    [((MLCTensorDescriptor*)self).batchSizePerSequenceStep enumerateObjectsUsingBlock:^(NSNumber* _Nonnull obj, NSUInteger idx, BOOL* _Nonnull stop) {
        vectorList.push_back((uint32_t)obj.unsignedIntegerValue);
    }];
    return vectorList;
}

auto CppMLCTensorDescriptor::getMaxTensorDimensions() -> uint64_t {
    return (uint64_t)MLCTensorDescriptor.maxTensorDimensions;
}

