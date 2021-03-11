#include "INerualNetwork.h"
#include "INerualNetwork.h"

#include "INerualNetwork.h"
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <fstream>
#include <sstream>

INerualNetwork::INerualNetwork(const INerualNetwork &n_copy) :
    m_nInputNeurons(n_copy.m_nInputNeurons),
    m_nOutputNeurons(n_copy.m_nOutputNeurons),
    m_nlCountLayers(n_copy.m_nlCountLayers),
    m_NetworkStructure(n_copy.m_NetworkStructure)

{
    m_nLayers.resize(m_nlCountLayers);
    for (unsigned int i = 0; i < n_copy.m_nLayers.size(); ++i)
    {
        m_nLayers[i] = new ILayerNeurons(n_copy.m_nLayers[i]->getInCount(),n_copy.m_nLayers[i]->getOutCount());
        m_nLayers[i]->getErrors()[i] =  n_copy.m_nLayers[i]->getErrors()[i];
        m_nLayers[i]->getHidden()[i] =  n_copy.m_nLayers[i]->getHidden()[i];
        for(unsigned int inp =0; inp < n_copy.m_nLayers[i]->getInCount()+1; inp++)
        {
            for(unsigned int outp =0; outp < n_copy.m_nLayers[i]->getOutCount(); outp++)
            {
                m_nLayers[i]->getMatrix(inp,outp) = n_copy.m_nLayers[i]->getMatrix(inp,outp);
            }
        }
    }
}

INerualNetwork::INerualNetwork(std::initializer_list<unsigned int> &&values )
{
    //--- "Neyeral Network" this equal "NN"
    //---set layer count for NN,
    //---where input neuerons for first layer equal NN input
    //---and output neuerons for last layer equal NN output
    m_nlCountLayers = values.size()-1;
    m_nLayers = vector<ILayerNeurons*>(m_nlCountLayers);
    m_NetworkStructure.resize(values.size());


    int i = 0;
    for (auto it = values.begin()+1; i < m_nlCountLayers; ++i, ++it)
    {
        int OutSize = *it;
        int InSize  = *(it-1);
        if(i==0) m_nInputNeurons = InSize;
        else if(i==m_nlCountLayers-1)  m_nOutputNeurons = OutSize;
        m_nLayers[i] = new ILayerNeurons(InSize,OutSize);
        m_NetworkStructure[i] = InSize;
        m_nLayers[i]->InitRandomWeight();
    }

    m_NetworkStructure[m_nlCountLayers] = m_nOutputNeurons;
}

INerualNetwork::INerualNetwork(std::vector<unsigned int> NetworkStructure)
    : m_NetworkStructure(NetworkStructure)
{
    //--- "Neyeral Network" this equal "NN"
    //---set layer count for NN,
    //---where input neuerons for first layer equal NN input
    //---and output neuerons for last layer equal NN output
    m_nlCountLayers = NetworkStructure.size()-1;
    m_nLayers = vector<ILayerNeurons*>(m_nlCountLayers);

    int i = 0;
    for (auto it = NetworkStructure.begin()+1; i < m_nlCountLayers; ++i, ++it)
    {
        int OutSize = *it;
        int InSize  = *(it-1);
        if(i==0) m_nInputNeurons = InSize;
        else if(i==m_nlCountLayers-1)  m_nOutputNeurons = OutSize;
        m_nLayers[i] = new ILayerNeurons(InSize,OutSize);
        m_nLayers[i]->InitRandomWeight();
    }
}


INerualNetwork::~INerualNetwork()
{
    for (int i =0; i<m_nlCountLayers; i++)
    {
        delete m_nLayers[i];
    }
}

void INerualNetwork::InitRandom()
{
    for (unsigned int i = 0; i < m_nLayers.size(); ++i)
    {
        m_nLayers[i]->InitRandomWeight();
    }
}


float* INerualNetwork::feedForwarding(const float* _inputes)
{
    for (int i = 0; i<m_nlCountLayers; i++)
    {
        auto &Layer = m_nLayers[i];
        for(unsigned int hid =0; hid < Layer->getOutCount(); hid++)
        {
            float tmpS = 0.0;
            for(unsigned int inp =0; inp < Layer->getInCount(); inp++)
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
            Layer->getHidden()[hid] = activation(tmpS,Layer->getLayer_type());
        }
    }

    return m_nLayers[m_nlCountLayers-1]->getHidden();
}


void INerualNetwork::backPropagate(const float *_inputes, const float *_targetes , const float& learningRate)
{
    //--- for other layer argument is "hidden" array previous's layer
    feedForwarding(_inputes);

    //--- calculate errors for last layer
    for(unsigned int i =0; i < m_nLayers[m_nlCountLayers-1]->getOutCount(); i++)
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
        for(unsigned int hid =0; hid < Layer->getInCount(); hid++)
        {
            Layer_prev->getErrors()[hid] = 0.0;
            for(unsigned int ou =0; ou < Layer->getOutCount(); ou++)
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
        for(unsigned int ou =0; ou < Layer->getOutCount(); ou++)
        {
            for(unsigned int hid =0; hid < Layer->getInCount(); hid++)
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


void INerualNetwork::CopyWeightAndErrorBias(const INerualNetwork &other)
{
    assert(m_nInputNeurons == other.m_nInputNeurons);
    assert(m_nOutputNeurons == other.m_nOutputNeurons);
    assert(m_nlCountLayers == other.m_nlCountLayers);

    for (unsigned int i = 0; i < other.m_nLayers.size(); ++i)
    {
        assert( m_nLayers[i]->getInCount() == other.m_nLayers[i]->getInCount());
        assert( m_nLayers[i]->getOutCount() == other.m_nLayers[i]->getOutCount());
        m_nLayers[i]->getErrors()[i] =  other.m_nLayers[i]->getErrors()[i];
       /** m_nLayers[i]->getHidden()[i] =  other.m_nLayers[i]->getHidden()[i]; /**/
        for(unsigned int inp =0; inp < other.m_nLayers[i]->getInCount()+1; inp++)
        {
            for(unsigned int outp =0; outp < other.m_nLayers[i]->getOutCount(); outp++)
            {
                m_nLayers[i]->getMatrix(inp,outp) = other.m_nLayers[i]->getMatrix(inp,outp);
            }
        }
    }
}

string INerualNetwork::serialize() const
{
    string retBuf;
    size_t nbrOfTopoElements = m_NetworkStructure.size() + 1;
    unsigned int* topoBuf = new unsigned int[ nbrOfTopoElements ];
    topoBuf[0] = m_NetworkStructure.size();//getCountLayers();
    for( size_t i = 0; i < m_NetworkStructure.size(); i++ )
    {
        topoBuf[i+1] = m_NetworkStructure.at(i);
    }

    retBuf.append( string( (char*)topoBuf, nbrOfTopoElements*sizeof(unsigned int) ) );
    for( auto l : m_nLayers )
    {
        string lBuf = l->serialize();
        unsigned int lBufSize = lBuf.size();
        retBuf.append( string( (char*)(&lBufSize), 1*sizeof(unsigned int) ) );
        retBuf.append(lBuf);
    }
    delete [] topoBuf;
    return retBuf;
}


INerualNetwork *INerualNetwork::deserialize(const string &buffer)
{
    const char* buff = buffer.c_str();
    unsigned int nbrOfLayers = ((unsigned int*)buff)[0];
    std::vector<unsigned int> networkStructure;
    for( unsigned int i = 0; i < nbrOfLayers; i++ )
    {
        networkStructure.push_back( ((unsigned int*)buff)[i+1] );
    }

    INerualNetwork* network = new INerualNetwork( networkStructure );
    size_t offset = (nbrOfLayers+1) * sizeof(unsigned int);
    for( unsigned int i = 0; i < nbrOfLayers-1; i++ )
    {
        const char* layerBuf = buff + offset;
        unsigned int sizeOfThisLayer = ((unsigned int*)layerBuf)[0];
        string layerData( layerBuf + sizeof(unsigned int), sizeOfThisLayer );
        ILayerNeurons* l = ILayerNeurons::deserialize( layerData );
        network->m_nLayers[i] = l;
        offset = offset + sizeOfThisLayer + sizeof(unsigned int);
    }
    return network;
}

bool INerualNetwork::save(const string &filePath)
{
    ofstream netFile;
    netFile.open( filePath ,ios::binary);

    if( !netFile.is_open() )
        return false;

    netFile << serialize();
    netFile.close();
    return true;
}

INerualNetwork *INerualNetwork::load(const string &filePath)
{
    ifstream netFile( filePath , ios::binary);
    if( ! netFile.is_open() )
    {
        //std::cout << "Error File \n";
        return NULL;
    }

    // read the whole file
    std::string netAsBuffer((std::istreambuf_iterator<char>(netFile)),
                             std::istreambuf_iterator<char>());
    netFile.close();

    return INerualNetwork::deserialize( netAsBuffer );
}


