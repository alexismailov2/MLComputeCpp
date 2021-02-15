#import "CppMLCTensor.h"
#import "CppMLCTensorDescriptor.h"
#import "CppMLCTensorData.h"

#import <MLCompute/MLCTensor.h>

namespace {
    auto toNSArray(std::vector<uint32_t> const& vector) -> NSArray<NSNumber*>* {
        id ns = [NSMutableArray new];
        std::for_each(vector.begin(), vector.end(), ^(uint32_t item) {
            [ns addObject:[NSNumber numberWithInteger:(NSInteger)item]];
        });
        return ns;
    }

    auto toNative(eMLCRandomInitializerType randomInitializerType) -> MLCRandomInitializerType {
        switch (randomInitializerType) {
            case eMLCRandomInitializerType::Invalid: return MLCRandomInitializerTypeInvalid;
            case eMLCRandomInitializerType::Uniform: return MLCRandomInitializerTypeUniform;
            case eMLCRandomInitializerType::GlorotUniform: return MLCRandomInitializerTypeGlorotUniform;
            case eMLCRandomInitializerType::Xavier: return MLCRandomInitializerTypeXavier;
            case eMLCRandomInitializerType::Count: return MLCRandomInitializerTypeCount;
            default: return MLCRandomInitializerTypeInvalid;
        }
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
} // end namespace anonymous

CppMLCTensor::CppMLCTensor()
    : self(NULL)
{
}

CppMLCTensor::~CppMLCTensor()
{
    [(id)self dealloc];
}

auto CppMLCTensor::getTensorId() -> uint64_t
{
    return (uint64_t)((MLCTensor*)self).tensorID;
}

auto CppMLCTensor::getDescriptor() -> CppMLCTensorDescriptor
{
    return CppMLCTensorDescriptor((void*)(((MLCTensor*)self).descriptor));
}

//auto CppMLCTensor::getData() -> NSData * {
//
//}

auto CppMLCTensor::getLabel() -> std::string
{
    return std::string([((MLCTensor*)self).label UTF8String]);
}

void CppMLCTensor::setLabel(std::string const& label)
{
    [((MLCTensor*)self).label initWithUTF8String:label.c_str()];
}

CppMLCTensor::CppMLCTensor(const CppMLCTensorDescriptor &tensorDescriptor)
    : self{[MLCTensor tensorWithDescriptor:(MLCTensorDescriptor*)tensorDescriptor.self]}
{
}

CppMLCTensor::CppMLCTensor(CppMLCTensorDescriptor const& tensorDescriptor,
                           eMLCRandomInitializerType randomInitializerType)
    : self{[MLCTensor tensorWithDescriptor:(MLCTensorDescriptor*)tensorDescriptor.self
                     randomInitializerType:toNative(randomInitializerType)]}
{
}

CppMLCTensor::CppMLCTensor(std::vector<uint32_t> const& shape)
    : self{[MLCTensor tensorWithShape:toNSArray(shape)]}
{
}

CppMLCTensor::CppMLCTensor(CppMLCTensorDescriptor const& tensorDescriptor, CppMLCTensorData const& data)
    : self{[MLCTensor tensorWithDescriptor:(MLCTensorDescriptor*)tensorDescriptor.self
                                      data:(MLCTensorData*)data.self]}
{
}

CppMLCTensor::CppMLCTensor(std::vector<uint32_t> const& shape, eMLCRandomInitializerType randomInitializerType)
    : self{[MLCTensor tensorWithShape:toNSArray(shape)
                randomInitializerType:toNative(randomInitializerType)]}
{
}

CppMLCTensor::CppMLCTensor(const std::vector<uint32_t> &shape, eMLCDataType dataType)
    : self{[MLCTensor tensorWithShape:toNSArray(shape)
                             dataType:toNative(dataType)]}
{
}

CppMLCTensor::CppMLCTensor(const std::vector<uint32_t> &shape, const CppMLCTensorData &data, eMLCDataType dataType)
    : self{[MLCTensor tensorWithShape:toNSArray(shape)
                                 data:(MLCTensorData*)data.self
                             dataType:toNative(dataType)]}
{
}

CppMLCTensor::CppMLCTensor(const std::vector<uint32_t> &shape, uint32_t fillData, eMLCDataType dataType)
    : self{[MLCTensor tensorWithShape:toNSArray(shape)
                         fillWithData:[NSNumber numberWithUnsignedInt:(NSUInteger)fillData]
                             dataType:toNative(dataType)]}
{
}

CppMLCTensor::CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize)
    : self{[MLCTensor tensorWithWidth:(NSUInteger)width
                               height:(NSUInteger)height
                  featureChannelCount:(NSUInteger)featureChannelCount
                            batchSize:(NSUInteger)batchSize]}
{
}

CppMLCTensor::CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize,
                           float fillData, eMLCDataType dataType)
    : self{[MLCTensor tensorWithWidth:(NSUInteger)width
                               height:(NSUInteger)height
                  featureChannelCount:(NSUInteger)featureChannelCount
                            batchSize:(NSUInteger)batchSize
                         fillWithData:fillData
                             dataType:toNative(dataType)]}
{
}

CppMLCTensor::CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize,
                           eMLCRandomInitializerType randomInitializerType)
    : self{[MLCTensor tensorWithWidth:(NSUInteger)width
                               height:(NSUInteger)height
                  featureChannelCount:(NSUInteger)featureChannelCount
                            batchSize:(NSUInteger)batchSize
                randomInitializerType:toNative(randomInitializerType)]}
{
}

CppMLCTensor::CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize,
                           const CppMLCTensorData &data)
    : self{[MLCTensor tensorWithWidth:(NSUInteger)width
                               height:(NSUInteger)height
                  featureChannelCount:(NSUInteger)featureChannelCount
                            batchSize:(NSUInteger)batchSize
                                 data:(MLCTensorData*)data.self]}
{
}

CppMLCTensor::CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize,
                           const CppMLCTensorData &data, eMLCDataType dataType)
    : self{[MLCTensor tensorWithWidth:(NSUInteger)width
                               height:(NSUInteger)height
                  featureChannelCount:(NSUInteger)featureChannelCount
                            batchSize:(NSUInteger)batchSize
                                 data:(MLCTensorData*)data.self
                             dataType:toNative(dataType)]}
{
}

bool CppMLCTensor::hasValidNumerics()
{
    return ((MLCTensor*)self).hasValidNumerics == YES;
}

bool CppMLCTensor::synchronizeData()
{
    return [(MLCTensor*)self synchronizeData] == YES;
}

bool CppMLCTensor::synchronizeOptimizerData()
{
    return [(MLCTensor*)self synchronizeOptimizerData] == YES;
}

bool CppMLCTensor::copyDataFromDeviceMemoryToBytes(void* bytes, uint32_t length, bool synchronizeWithDevice)
{
    return [(MLCTensor*)self copyDataFromDeviceMemoryToBytes:bytes
                                                      length:(NSUInteger)length
                                       synchronizeWithDevice:synchronizeWithDevice ? YES : NO] == YES;
}

bool CppMLCTensor::bindAndWriteData(CppMLCTensorData const& data, CppMLCDevice const& toDevice)
{
    return [(MLCTensor*)self bindAndWriteData:(MLCTensorData*)data.self
                                       toDevice:(MLCDevice*)toDevice.self] == YES;
}
