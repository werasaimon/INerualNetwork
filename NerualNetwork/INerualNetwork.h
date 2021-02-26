#ifndef INERUALNETWORK_H
#define INERUALNETWORK_H


#include <math.h>
#include <vector>

#include "ILayerNeurons.hpp"

using namespace std;


class INerualNetwork
{

    float (*activation)(float);
    float (*derivative)(float);

public:

    /**
     * Deffault-Constructor
     * @param n_copy
     */
    INerualNetwork(){}

    /**
     * Copy-Constructor
     * @param n_copy
     */
    INerualNetwork(const INerualNetwork& n_copy);

    /**
     * Initilization-Constructor
     * @param n_copy
     */
    INerualNetwork(float (*_activation)(float),
                   float (*_derivative)(float),
                   std::initializer_list<unsigned int>&& values);

   // myNeuro();
    ~INerualNetwork();


    void InitRandom();


    float *feedForwarding(const float *_input);
    void backPropagate(const float *_inputes, const float *targetes,const float& learningRate);


    unsigned int getCountNeuronsOutput() const { return m_nOutputNeurons; }
    unsigned int getCountNeuronsInput() const { return m_nInputNeurons; }
    unsigned int getCountLayers() const { return m_nlCountLayers; }

    ILayerNeurons* getLayer(int indx) const
    {
        return m_nLayers[indx];
    }

    ILayerNeurons* getLayer(int indx)
    {
       return m_nLayers[indx];
    }

    void CopyWeightAndErrorBias(const INerualNetwork& other);

private:
    std::vector<ILayerNeurons*> m_nLayers;
    int m_nInputNeurons;
    int m_nOutputNeurons;
    int m_nlCountLayers;

};

#endif // INERUALNETWORK_H
