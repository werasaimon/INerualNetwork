#include "INerualNetwork.h"


INerualNetwork::INerualNetwork(float (*_activation)(float),
                 float (*_derivative)(float),
                 std::initializer_list<unsigned int> &&values)
    :activation(_activation),
     derivative(_derivative)
{
    //--- "Neyeral Network" this equal "NN"
    //---set layer count for NN,
    //---where input neuerons for first layer equal NN input
    //---and output neuerons for last layer equal NN output
    m_nlCountLayers = values.size()-1;
    m_nLayers = vector<ILayerNeurons*>(m_nlCountLayers);

    int i = 0;
    for (auto it = values.begin()+1; i < m_nlCountLayers; ++i, ++it)
    {
        int OutSize = *it;
        if(i < m_nlCountLayers)
        {
            int InSize = *(it-1);
            if(i==0) m_nInputNeurons = InSize;
            else if(i==m_nlCountLayers-1)  m_nOutputNeurons = OutSize;
            m_nLayers[i] = new ILayerNeurons(InSize,OutSize);
        }
   }
}

INerualNetwork::~INerualNetwork()
{
    for (int i =0; i<m_nlCountLayers; i++)
    {
        delete m_nLayers[i];
    }
}


float* INerualNetwork::feedForwarding(const float* _inputes)
{
    for (int i = 0; i<m_nlCountLayers; i++)
    {
        auto &Layer = m_nLayers[i];

        for(int hid =0; hid < Layer->getOutCount(); hid++)
        {
            float tmpS = 0.0;
            for(int inp =0; inp < Layer->getInCount(); inp++)
            {
                if(i == 0)
                {
                    //--- for first layer argument is _inputs
                   tmpS += _inputes[inp] * Layer->getMatrix(inp,hid);
                }
                else
                {
                    //--- for other layer argument is "hidden" array previous's layer
                   tmpS += m_nLayers[i-1]->getHidden()[inp] * Layer->getMatrix(inp,hid);
                }
            }
            tmpS += Layer->getMatrix(Layer->getInCount(),hid);
            Layer->getHidden()[hid] = activation(tmpS);
        }
    }

    return m_nLayers[m_nlCountLayers-1]->getHidden();
}


void INerualNetwork::backPropagate(const float *_inputes, const float *_targetes , const float& learningRate)
{
    //--- for other layer argument is "hidden" array previous's layer
    feedForwarding(_inputes);

    //--- calculate errors for last layer
    for(int i =0; i < m_nLayers[m_nlCountLayers-1]->getOutCount(); i++)
    {
        auto &Layer = m_nLayers[m_nlCountLayers-1];
        Layer->getErrors()[i] = (_targetes[i] - Layer->getHidden()[i]) * derivative(Layer->getHidden()[i]);
    }

    //--- for others layers to calculate errors we need information about "next layer"
    //---   //for example// to calculate 4'th layer errors we need 5'th layer errors
    for (int i = m_nlCountLayers-2; i>=0; i--)
    {
        auto &Layer_prev = m_nLayers[i];
        auto &Layer = m_nLayers[i+1];
        for(int hid =0; hid < Layer->getInCount(); hid++)
        {
            Layer_prev->getErrors()[hid] = 0.0;
            for(int ou =0; ou < Layer->getOutCount(); ou++)
            {
                Layer_prev->getErrors()[hid] += Layer->getErrors()[ou] * Layer->getMatrix(hid,ou);
            }
            Layer_prev->getErrors()[hid] *= derivative(Layer_prev->getHidden()[hid]);
        }
    }


    //----------------------------------------------------------------------//
    for (int i = m_nlCountLayers-1; i>=0; i--)
    {
        auto &Layer = m_nLayers[i];
        for(int ou =0; ou < Layer->getOutCount(); ou++)
        {
            for(int hid =0; hid < Layer->getInCount(); hid++)
            {
                if(i==0)
                {
                    //--- first layer hasn't previous layer.
                    //--- for him "hidden" value array of previous layer be NN input
                    Layer->getMatrix(hid,ou) += (learningRate * Layer->getErrors()[ou] * _inputes[hid]);
                }
                else if(i>0)
                {
                    //--- updating weights
                    //--- to UPD weight for current layer we must get "hidden" value array of previous layer
                    Layer->getMatrix(hid,ou) += (learningRate * Layer->getErrors()[ou] * m_nLayers[i-1]->getHidden()[hid]);
                }
            }
            Layer->getMatrix(Layer->getInCount(),ou) += (learningRate * Layer->getErrors()[ou]);
        }
    }
}
