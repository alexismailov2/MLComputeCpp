#include "CppMLCGraph.h"
#include "CppMLCDevice.h"
#include "CppMLCTypesPrivate.h"
#include "CppMLCLayer.h"
#include "CppMLCTensor.h"

#import <MLCompute/MLCDevice.h>
#import <MLCompute/MLCGraph.h>
#import <MLCompute/MLCDevice.h>

#include <string>

auto CppMLCGraph::device() -> CppMLCDevice
{
    return CppMLCDevice{((MLCGraph*)self).device};
}

auto CppMLCGraph::layers() -> std::vector<CppMLCLayer>
{
    return CppMLCTypesPrivate::MLCLayerArrayToVector(((MLCGraph*)self).layers);
}

auto CppMLCGraph::graph() -> CppMLCGraph
{
    return CppMLCGraph{[MLCGraph graph]};
}

auto CppMLCGraph::summarizedDOTDescription() -> std::string
{
    return std::string([((MLCGraph*)self).summarizedDOTDescription UTF8String]);;
}

auto CppMLCGraph::nodeWithLayer(CppMLCLayer const& layer,
                                CppMLCTensor const& source) -> CppMLCTensor
{
    [(id)source.self retain];
    [(id)layer.self retain];
    MLCTensor* tensor = [(MLCGraph*)self nodeWithLayer:(MLCLayer*)layer.self
                                                source:(MLCTensor*)source.self];
    return CppMLCTensor{tensor};
}

auto CppMLCGraph::nodeWithLayer(CppMLCLayer const& layer,
                                std::vector<CppMLCTensor> const& sources) -> CppMLCTensor
{
    return CppMLCTensor{[(MLCGraph*)self nodeWithLayer:(MLCLayer*)layer.self
                                               sources:CppMLCTypesPrivate::toNSArray(sources)]};
}

auto CppMLCGraph::nodeWithLayerPtr(CppMLCLayer const& layer,
                                   std::vector<CppMLCTensor*> const& sources) -> CppMLCTensor
{
    auto ptr = [(MLCGraph*)self nodeWithLayer:(MLCLayer*)layer.self
                                      sources:CppMLCTypesPrivate::toNSArray(sources)];
    return CppMLCTensor{ptr};
}

auto CppMLCGraph::nodeWithLayer(CppMLCLayer const& layer,
                                std::vector<CppMLCTensor> const& sources,
                                bool disableUpdate) -> CppMLCTensor
{
    return CppMLCTensor{[(MLCGraph*)self nodeWithLayer:(MLCLayer*)layer.self
                                               sources:CppMLCTypesPrivate::toNSArray(sources)
                                         disableUpdate:disableUpdate ? YES : NO]};
}

auto CppMLCGraph::nodeWithLayer(CppMLCLayer const& layer,
                                std::vector<CppMLCTensor> const& sources,
                                std::vector<CppMLCTensor> const& lossLabels) -> CppMLCTensor
{
    return CppMLCTensor{[(MLCGraph*)self nodeWithLayer:(MLCLayer*)layer.self
                                               sources:CppMLCTypesPrivate::toNSArray(sources)
                                            lossLabels:CppMLCTypesPrivate::toNSArray(lossLabels)]};
}

auto CppMLCGraph::splitWithSource(CppMLCTensor const& source,
                                  uint32_t splitCount,
                                  uint32_t dimension) -> std::vector<CppMLCTensor>
{
    return CppMLCTypesPrivate::MLCTensorArrayToVector([(MLCGraph*)self splitWithSource:(MLCTensor*)source.self
                                                                            splitCount:(NSUInteger)splitCount
                                                                             dimension:(NSUInteger)dimension]);
}

auto CppMLCGraph::splitWithSource(CppMLCTensor const& source,
                                  std::vector<uint32_t> const& splitSectionLengths,
                                  uint32_t dimension) -> std::vector<CppMLCTensor>
{
    return CppMLCTypesPrivate::MLCTensorArrayToVector([(MLCGraph*)self splitWithSource:(MLCTensor*)source.self
                                                                   splitSectionLengths:CppMLCTypesPrivate::toNSArray(splitSectionLengths)
                                                                             dimension:(NSUInteger)dimension]);
}

auto CppMLCGraph::concatenateWithSources(std::vector<CppMLCTensor> const& sources,
                                         uint32_t dimension) -> CppMLCTensor
{
    return CppMLCTensor{[(MLCGraph*)self concatenateWithSources:CppMLCTypesPrivate::toNSArray(sources)
                                                           dimension:(NSUInteger)dimension]};
}

auto CppMLCGraph::reshapeWithShape(std::vector<uint32_t> const& shape,
                                   CppMLCTensor const& source) -> CppMLCTensor
{
    return CppMLCTensor{[(MLCGraph*)self reshapeWithShape:CppMLCTypesPrivate::toNSArray(shape)
                                                        source:(MLCTensor*)source.self]};
}

auto CppMLCGraph::transposeWithDimensions(std::vector<uint32_t> const& dimensions,
                                          CppMLCTensor const& source) -> CppMLCTensor
{
    return CppMLCTensor{[(MLCGraph*)self transposeWithDimensions:CppMLCTypesPrivate::toNSArray(dimensions)
                                                               source:(MLCTensor*)source.self]};
}

bool CppMLCGraph::bindAndWriteData(std::map<std::string, CppMLCTensorData> const& inputsData,
                                   std::map<std::string, CppMLCTensor> const& inputTensors,
                                   CppMLCDevice const& device,
                                   uint32_t batchSize,
                                   bool synchronous)
{
    return [(MLCGraph*)self bindAndWriteData:CppMLCTypesPrivate::toNSDictionary(inputsData)
                                   forInputs:CppMLCTypesPrivate::toNSDictionary(inputTensors)
                                    toDevice:(MLCDevice*)device.self
                                   batchSize:(NSUInteger)batchSize
                                 synchronous:(synchronous ? YES : NO)] == YES;
}

bool CppMLCGraph::bindAndWriteData(std::map<std::string, CppMLCTensorData> const& inputsData,
                                   std::map<std::string, CppMLCTensor> const& inputTensors,
                                   CppMLCDevice const& device,
                                   bool synchronous)
{
    return [(MLCGraph*)self bindAndWriteData:CppMLCTypesPrivate::toNSDictionary(inputsData)
                                   forInputs:CppMLCTypesPrivate::toNSDictionary(inputTensors)
                                    toDevice:(MLCDevice*)device.self
                                 synchronous:(synchronous ? YES : NO)] == YES;
}

auto CppMLCGraph::sourceTensorsForLayer(CppMLCLayer const& layer) -> std::vector<CppMLCTensor>
{
    return CppMLCTypesPrivate::MLCTensorArrayToVector([(MLCGraph*)self sourceTensorsForLayer:(MLCLayer*)layer.self]);
}

auto CppMLCGraph::resultTensorsForLayer(CppMLCLayer const& layer) -> std::vector<CppMLCTensor>
{
    return CppMLCTypesPrivate::MLCTensorArrayToVector([(MLCGraph*)self resultTensorsForLayer:(MLCLayer*)layer.self]);
}

//CppMLCGraph::CppMLCGraph()
//    : self{[MLCGraph graph]}
//{
//}

CppMLCGraph::CppMLCGraph(void *self)
    : self{self}
{
    [(id)self retain];
}

CppMLCGraph::~CppMLCGraph()
{
    [(id)self release];
}
