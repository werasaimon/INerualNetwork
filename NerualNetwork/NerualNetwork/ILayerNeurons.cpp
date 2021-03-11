#include "ILayerNeurons.h"

void ILayerNeurons::InitRandomWeight()
{
    srand(time(NULL));
    std::mt19937 rng;
    // initialize the random number generator with time-dependent seed
    uint32_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>16)};
    rng.seed(ss);
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif(0, 1);

    for(int outp =0; outp < out_count; outp++)
    {
        //errorsBias[outp] =  (((float)rand() / (float)RAND_MAX) - 0.5)/out_count;// * pow(out_count,-0.5);
          errorsBias[outp] = (unif(rng) - 0.5) * pow(out_count,-0.5);
    }


    for(int inp =0; inp < in_count+1; inp++)
    {
        for(int outp =0; outp < out_count; outp++)
        {
            getMatrix(inp,outp) =  (unif(rng) - 0.5) * pow(out_count,-0.5);
            //getMatrix(inp,outp) = ( ((float)rand() / (float)RAND_MAX) - 0.5 )* pow(out_count,-0.5);
        }
    }
}

std::string ILayerNeurons::serialize() const
{
    unsigned int* topoBuf = new unsigned int[3];
    topoBuf[0] = in_count;
    topoBuf[1] = out_count;
    topoBuf[2] = static_cast<unsigned int>(m_layer_type);

    size_t nbrOfDoublesWeightMatrix = (in_count+1) * out_count; // + in_count;
    float* weightBuf = new float[ nbrOfDoublesWeightMatrix ];
    for( int m = 0; m < in_count+1; m++ )
    {
        for( int n = 0; n < out_count; n++ )
        {
            weightBuf[m*out_count + n] = matrixWeight[m*out_count + n];
        }
    }

    size_t nbrOfDoublesBias = out_count;
    float* biasBuf = new float[ nbrOfDoublesBias ];
    for( size_t m = 0; m < nbrOfDoublesBias; m++ )
    {
        biasBuf[m] = errorsBias[m];
    }

    std::string retBuffer;
    retBuffer.append( std::string( (char*)topoBuf, 3*sizeof(unsigned int) ) );
    retBuffer.append( std::string( (char*)weightBuf, nbrOfDoublesWeightMatrix*sizeof(float) ) );
    retBuffer.append( std::string( (char*)biasBuf, nbrOfDoublesBias*sizeof(float) ) );

    delete[] topoBuf;
    delete[] weightBuf;
    delete[] biasBuf;

    return retBuffer;
}

ILayerNeurons *ILayerNeurons::deserialize(const std::string &buffer)
{
    const char* buf = buffer.c_str();
    unsigned int nbrOfNeurons = ((unsigned int*)(buf))[0];
    unsigned int nbrOfOutputs = ((unsigned int*)(buf))[1];
    ILayerNeurons::LayerOutputType lType = static_cast<ILayerNeurons::LayerOutputType>(((unsigned int*)(buf))[2]);

    ILayerNeurons* layer = new ILayerNeurons( nbrOfNeurons, nbrOfOutputs , lType );

    size_t offset = 3 * sizeof(unsigned int);
    const float* weightBuf = (const float*)((buf + offset));
    for( size_t m = 0; m < nbrOfNeurons+1; m++ )
    {
        for( size_t n = 0; n < nbrOfOutputs; n++ )
        {
            layer->matrixWeight[m*nbrOfOutputs + n] = weightBuf[ m*nbrOfOutputs + n];
        }
    }

    offset = offset + (nbrOfNeurons+1)*nbrOfOutputs*sizeof(float);
    const float* biasBuf = (const float*)((buf + offset));
    for( size_t m = 0; m < nbrOfOutputs; m++ )
    {
        layer->errorsBias[m] = biasBuf[m];
    }

    return layer;
}

ILayerNeurons::LayerOutputType ILayerNeurons::getLayer_type() const
{
    return m_layer_type;
}

void ILayerNeurons::setLayer_type(const LayerOutputType &layer_type)
{
    m_layer_type = layer_type;
}
