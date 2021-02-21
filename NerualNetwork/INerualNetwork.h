#ifndef INERUALNETWORK_H
#define INERUALNETWORK_H


#include <math.h>
#include <vector>
#include <time.h>
using namespace std;


class ILayerNeurons
{

private:

        //--- information about input/out width for neuro layer
       int inSize;
       int outSize;
       //--- weight matrix
       float* matrixWeight;
       //--- current hidden value array
       float* hiddenNeurons;
       //--- current errors for backPropagate
       float* errorsBias;
public:

       ILayerNeurons(){}
       ILayerNeurons(int in_count , int out_count )
        :inSize(in_count),
         outSize(out_count)
       {
           srand((unsigned)time(NULL));
           //--- initialization values and allocating memory
           errorsBias = new float[outSize];
           hiddenNeurons = new float[outSize];
           matrixWeight = new float[(inSize+1) * outSize];
           for(int inp =0; inp < inSize+1; inp++)
           {
               for(int outp =0; outp < outSize; outp++)
               {
                   getMatrix(inp,outp) =  (((float)rand() / (float)RAND_MAX) - 0.5) * pow(outSize,-0.5);
               }
           }
       }

       ~ILayerNeurons()
       {
           inSize=0;
           outSize=0;
           if(errorsBias) delete[] errorsBias;
           if(hiddenNeurons) delete[] hiddenNeurons;
           if(matrixWeight) delete[] matrixWeight;
       }


       int getInCount(){return inSize;}
       int getOutCount(){return outSize;}
       float *getMatrix(){return matrixWeight;}
       float* getHidden(){ return hiddenNeurons;}
       float* getErrors() { return errorsBias;}
       float &getMatrix(int i , int j)
       {
           return matrixWeight[i*outSize + j];
       }
};

class INerualNetwork
{

    float (*activation)(float);
    float (*derivative)(float);


public:

    INerualNetwork(float (*_activation)(float),
                   float (*_derivative)(float),
                  std::initializer_list<unsigned int>&& values);

   // myNeuro();
    ~INerualNetwork();


    float *feedForwarding(const float *_input);
    void backPropagate(const float *_inputes, const float *targetes,const float& learningRate);

    int getCountNeuronsOutput() const { return m_nOutputNeurons; }
    int getCountNeuronsInput() const { return m_nInputNeurons; }
    int getCountLayers() const { return m_nlCountLayers; }

private:
    std::vector<ILayerNeurons*> m_nLayers;
    int m_nInputNeurons;
    int m_nOutputNeurons;
    int m_nlCountLayers;

};

#endif // INERUALNETWORK_H
