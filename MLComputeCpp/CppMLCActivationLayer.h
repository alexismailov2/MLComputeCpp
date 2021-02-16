#pragma once

#import "CppMLCLayer.h"

class CppMLCDevice;
class CppMLCTensor;
class CppMLCActivationDescriptor;

class CppMLCActivationLayer : public CppMLCLayer
{
public:
    CppMLCActivationLayer(void* self);

    /*! @property   descriptor
        @abstract   The activation descriptor
     */
    CppMLCActivationDescriptor descriptor();

    /*! @abstract   Create an activation layer
        @param      descriptor     The activation descriptor
        @return     A new activation layer
     */
    static CppMLCActivationLayer layerWithDescriptor(CppMLCActivationDescriptor const& descriptor);

    /*! @abstract   Create a ReLU activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer reluLayer();

    /*! @abstract   Create a ReLU6 activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer relu6Layer();

    /*! @abstract   Create a leaky ReLU activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer leakyReLULayer();

    /*! @abstract   Create a leaky ReLU activation layer
        @param      negativeSlope     Controls the angle of the negative slope
        @return     A new activation layer
     */
    static CppMLCActivationLayer leakyReLULayerWithNegativeSlope(float negativeSlope);

    /*! @abstract   Create a linear activation layer
        @param      scale   The scale factor
        @param      bias     The bias value
        @return     A new activation layer
     */
    static CppMLCActivationLayer linearLayerWithScale(float scale, float bias);

    /*! @abstract   Create a sigmoid activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer sigmoidLayer();

    /*! @abstract   Create a hard sigmoid activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer hardSigmoidLayer();

    /*! @abstract   Create a tanh activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer tanhLayer();

    /*! @abstract   Create an absolute activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer absoluteLayer();

    /*! @abstract   Create a soft plus activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer softPlusLayer();

    /*! @abstract   Create a soft plus activation layer
        @param      beta    The beta value for the softplus formation
        @return     A new activation layer
     */
    static CppMLCActivationLayer softPlusLayerWithBeta(float beta);

    /*! @abstract   Create a soft sign activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer softSignLayer();

    /*! @abstract   Create an ELU activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer eluLayer();

    /*! @abstract   Create an ELU activation layer
        @param      a   The \p a value for the ELU formation
        @return     A new activation layer
     */
    static CppMLCActivationLayer eluLayerWithA(float a);

    /*! @abstract   Create a ReLUN activation layer
        @discussion This can be used to implement layers such as ReLU6 for example.
        @param      a   The \p a value
        @param      b   The \p b value
        @return     A new activation layer
     */
    static CppMLCActivationLayer relunLayerWithA(float a, float b);

    /*! @abstract   Create a log sigmoid activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer logSigmoidLayer();

    /*! @abstract   Create a SELU activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer seluLayer();

    /*! @abstract   Create a CELU activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer celuLayer();

    /*! @abstract   Create a CELU activation layer
        @param      a   The \p a value for the CELU formation
        @return     A new activation layer
     */
    static CppMLCActivationLayer celuLayerWithA(float a);

    /*! @abstract   Create a hard shrink activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer hardShrinkLayer();

    /*! @abstract   Create a hard shrink activation layer
        @param      a   The \p a value for the hard shrink formation
        @return     A new activation layer
     */
    static CppMLCActivationLayer hardShrinkLayerWithA(float a);

    /*! @abstract   Create a soft shrink activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer softShrinkLayer();

    /*! @abstract   Create a soft shrink activation layer
        @param      a   The \p a value for the soft shrink formation
        @return     A new activation layer
     */
    static CppMLCActivationLayer softShrinkLayerWithA(float a);

    /*! @abstract   Create a TanhShrink activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer tanhShrinkLayer();

    /*! @abstract   Create a threshold activation layer
        @param      threshold    The value to threshold at
        @param      replacement  The value to replace with
        @return     A new activation layer
     */
    static CppMLCActivationLayer thresholdLayerWithThreshold(float threshold, float replacement);

    /*! @abstract   Create a GELU activation layer
        @return     A new activation layer
     */
    static CppMLCActivationLayer geluLayer();

private:
    void* self;
};
