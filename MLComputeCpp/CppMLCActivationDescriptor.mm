#import "CppMLCActivationDescriptor.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCActivationDescriptor.h>

auto CppMLCActivationDescriptor::getActivationType() -> eMLCActivationType {
    return MLCActivationTypeToCpp(((MLCActivationDescriptor*)self).activationType);
}

auto CppMLCActivationDescriptor::getA() -> float {
    return ((MLCActivationDescriptor*)self).a;
}

auto CppMLCActivationDescriptor::getB() -> float {
    return ((MLCActivationDescriptor*)self).b;
}

auto CppMLCActivationDescriptor::getC() -> float {
    return ((MLCActivationDescriptor*)self).c;
}

CppMLCActivationDescriptor
CppMLCActivationDescriptor::descriptorWithType(eMLCActivationType activationType) {
    return CppMLCActivationDescriptor{[MLCActivationDescriptor descriptorWithType:toNative(activationType)]};
}

CppMLCActivationDescriptor
CppMLCActivationDescriptor::descriptorWithType(eMLCActivationType activationType, float a) {
    return CppMLCActivationDescriptor{[MLCActivationDescriptor descriptorWithType:toNative(activationType)
                                                                                a:a]};
}

CppMLCActivationDescriptor
CppMLCActivationDescriptor::descriptorWithType(eMLCActivationType activationType, float a, float b) {
    return CppMLCActivationDescriptor{[MLCActivationDescriptor descriptorWithType:toNative(activationType)
                                                                                a:a
                                                                                b:b]};
}

CppMLCActivationDescriptor
CppMLCActivationDescriptor::descriptorWithType(eMLCActivationType activationType, float a, float b, float c) {
    return CppMLCActivationDescriptor{[MLCActivationDescriptor descriptorWithType:toNative(activationType)
                                                                                a:a
                                                                                b:b
                                                                                c:c]};
}

CppMLCActivationDescriptor::CppMLCActivationDescriptor(void *self)
    : self{self}
{
    [(id)self retain];
}

CppMLCActivationDescriptor::~CppMLCActivationDescriptor()
{
    //[(id)self release];
}
