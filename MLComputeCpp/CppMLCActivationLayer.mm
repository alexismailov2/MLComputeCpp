#import "CppMLCActivationLayer.h"
#import "CppMLCActivationDescriptor.h"

#import <MLCompute/MLCActivationLayer.h>

CppMLCActivationLayer::CppMLCActivationLayer(void *self)
    : CppMLCLayer{self}
    , self{self}
{
    [(id)self retain];
}

CppMLCActivationLayer::~CppMLCActivationLayer()
{
    //[(id)self release];
}

CppMLCActivationDescriptor CppMLCActivationLayer::descriptor() {
    return CppMLCActivationDescriptor{((MLCActivationLayer*)self).descriptor};
}

CppMLCActivationLayer CppMLCActivationLayer::layerWithDescriptor(CppMLCActivationDescriptor const& descriptor) {
    [(id)descriptor.self retain];
    return CppMLCActivationLayer{descriptor.self};
}

CppMLCActivationLayer CppMLCActivationLayer::reluLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.reluLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::relu6Layer() {
    return CppMLCActivationLayer{MLCActivationLayer.relu6Layer};
}

CppMLCActivationLayer CppMLCActivationLayer::leakyReLULayer() {
    return CppMLCActivationLayer{MLCActivationLayer.leakyReLULayer};
}

CppMLCActivationLayer CppMLCActivationLayer::leakyReLULayerWithNegativeSlope(float negativeSlope) {
    return CppMLCActivationLayer{[MLCActivationLayer leakyReLULayerWithNegativeSlope:negativeSlope]};
}

CppMLCActivationLayer CppMLCActivationLayer::linearLayerWithScale(float scale, float bias) {
    return CppMLCActivationLayer{[MLCActivationLayer linearLayerWithScale:scale bias:bias]};
}

CppMLCActivationLayer CppMLCActivationLayer::sigmoidLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.sigmoidLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::hardSigmoidLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.hardSigmoidLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::tanhLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.tanhLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::absoluteLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.absoluteLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::softPlusLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.softPlusLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::softPlusLayerWithBeta(float beta) {
    return CppMLCActivationLayer{[MLCActivationLayer softPlusLayerWithBeta:beta]};
}

CppMLCActivationLayer CppMLCActivationLayer::softSignLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.softSignLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::eluLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.eluLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::eluLayerWithA(float a) {
    return CppMLCActivationLayer{[MLCActivationLayer eluLayerWithA:a]};
}

CppMLCActivationLayer CppMLCActivationLayer::relunLayerWithA(float a, float b) {
    return CppMLCActivationLayer{[MLCActivationLayer relunLayerWithA:a b:b]};
}

CppMLCActivationLayer CppMLCActivationLayer::logSigmoidLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.logSigmoidLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::seluLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.seluLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::celuLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.celuLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::celuLayerWithA(float a) {
    return CppMLCActivationLayer{[MLCActivationLayer celuLayerWithA:a]};
}

CppMLCActivationLayer CppMLCActivationLayer::hardShrinkLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.hardShrinkLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::hardShrinkLayerWithA(float a) {
    return CppMLCActivationLayer{[MLCActivationLayer hardShrinkLayerWithA:a]};
}

CppMLCActivationLayer CppMLCActivationLayer::softShrinkLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.softShrinkLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::softShrinkLayerWithA(float a) {
    return CppMLCActivationLayer{[MLCActivationLayer softShrinkLayerWithA:a]};
}

CppMLCActivationLayer CppMLCActivationLayer::tanhShrinkLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.tanhShrinkLayer};
}

CppMLCActivationLayer CppMLCActivationLayer::thresholdLayerWithThreshold(float threshold, float replacement) {
    return CppMLCActivationLayer{[MLCActivationLayer thresholdLayerWithThreshold:threshold replacement:replacement]};
}

CppMLCActivationLayer CppMLCActivationLayer::geluLayer() {
    return CppMLCActivationLayer{MLCActivationLayer.geluLayer};
}
