#pragma once

#import "CppMLCTypes.h"

class CppMLCActivationLayer;

/*!
 @class      CppMLCActivationDescriptor
 @discussion The CppMLCActivationDescriptor specifies a neuron descriptor.
             Supported neuron types:

             Neuron type "none": f(x) = x
             Parameters: none

             ReLU neuron filter: f(x) = x >= 0 ? x : a * x
             This is called Leaky ReLU in literature. Some literature defines
             classical ReLU as max(0, x). If you want this behavior, simply pass a = 0.
             Parameters: a
             For default behavior, set the value of a to 0.0f.

             Linear neuron filter: f(x) = a * x + b
             Parameters: a, b
             For default behavior, set the value of a to 1.0f and the value of b to 0.0f.

             Sigmoid neuron filter: f(x) = 1 / (1 + e^-x)
             Parameters: none

             Hard Sigmoid filter: f(x) = clamp((x * a) + b, 0, 1)
             Parameters: a, b
             For default behavior, set the value of a to 0.2f and the value of b to 0.5f.

             Hyperbolic tangent (TanH) neuron filter: f(x) = a * tanh(b * x)
             Parameters: a, b
             For default behavior, set the value of a to 1.0f and the value of b to 1.0f.

             Absolute neuron filter: f(x) = fabs(x)
             Parameters: none

             Parametric Soft Plus neuron filter: f(x) = a * log(1 + e^(b * x))
             Parameters: a, b
             For default behavior, set the value of a to 1.0f and the value of b to 1.0f.

             Parametric Soft Sign neuron filter: f(x) = x / (1 + abs(x))
             Parameters: none

             Parametric ELU neuron filter: f(x) = x >= 0 ? x : a * (exp(x) - 1)
             Parameters: a
             For default behavior, set the value of a to 1.0f.

             ReLUN neuron filter: f(x) = min((x >= 0 ? x : a * x), b)
             Parameters: a, b
             As an example, the TensorFlow Relu6 activation layer can be implemented
             by setting the parameter b to 6.0f:

             LogSigmoid neuron filter: f(x) =  log(1 / (1 + exp(-x)))
             Parameters: none

             SELU neuron filter: f(x) =  scale∗(max(0,x)+min(0,α∗(exp(x)−1))), with
                α = 1.6732632423543772848170429916717 and scale = 1.0507009873554804934193349852946
             Parameters: none

             CELU neuron filter: f(x) =  max(0,x)+min(0,a∗(exp(x/a)−1))
             Parameters: a
             For default behavior, set the value of a to 1.0f.

             HardShrink neuron filter: f(x) = x, if x > λ or x < −λ, 0 otherwise
             Parameters: lambda
             For default behavior, set the value of a to 0.5f

             SoftShtrink neuron filter: f(x) = x - λ, if x > λ, x + λ, if x < −λ, 0 otherwise
             Parameters: lambda (specified as parameter "a")
             For default behavior, set the value of a to 0.5f

             TanhShrink neuron filter: f(x) = x - Tanh(x)
             Parameters: none

             Threshold neuron filter: f(x) = x, if x > threshold, value otherwise
             Parameters: threshold (specified as parameter "a"), value (specified as parameter "b")

             GELU neuron filter: f(x) = x * CDF(x)
             Parameters: none
 */
class CppMLCActivationDescriptor
{
public:
    /*! @property   activationType
        @abstract   The type of activation function
     */
    auto getActivationType() -> eMLCActivationType;

    /*! @property   a
        @abstract   Parameter to the activation function
     */
    auto getA() -> float;

    /*! @property   b
        @abstract   Parameter to the activation function
     */
    auto getB() -> float;;

    /*! @property   c
        @abstract   Parameter to the activation function
     */
    auto getC() -> float;;

    /*! @abstract  Create a MLCActivationDescriptor object
        @param     activationType  A type of activation function.
        @return    A new neuron descriptor or nil if failure
     */
    static CppMLCActivationDescriptor descriptorWithType(eMLCActivationType activationType);

    /*! @abstract  Create a MLCActivationDescriptor object
        @param     activationType  A type of activation function.
        @param     a                      Parameter "a".
        @return    A new neuron descriptor or nil if failure
     */
    static CppMLCActivationDescriptor descriptorWithType(eMLCActivationType activationType, float a);

    /*! @abstract  Create a MLCActivationDescriptor object
        @param     activationType  A type of activation function.
        @param     a                      Parameter "a".
        @param     b                      Parameter "b".
        @return    A new neuron descriptor or nil if failure
     */
    static CppMLCActivationDescriptor descriptorWithType(eMLCActivationType activationType, float a, float b);


    /*! @abstract  Create a MLCActivationDescriptor object
        @param     activationType  A type of activation function.
        @param     a                      Parameter "a".
        @param     b                      Parameter "b".
        @param     c                      Parameter "c".
        @return    A new neuron descriptor or nil if failure
     */
    static CppMLCActivationDescriptor descriptorWithType(eMLCActivationType activationType, float a, float b, float c);

private:
    CppMLCActivationDescriptor(void* self);

public:
    ~CppMLCActivationDescriptor();

private:
    void* self;

    friend CppMLCActivationLayer;
};
