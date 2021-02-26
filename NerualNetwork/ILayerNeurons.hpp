#ifndef ILAYERNEURONS_H
#define ILAYERNEURONS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include <chrono>


class ILayerNeurons
{

private:
        //--- information about input/out width for neuro layer
       int in_count;
       int out_count;
       //--- weight matrix
       float* matrixWeight;
       //--- current hidden value array
       float* hiddenNeurons;
       //--- current errors for backPropagate
       float* errorsBias;


public:

       ILayerNeurons(){}
       ILayerNeurons(int _in_count , int _out_count )
        :in_count(_in_count),
         out_count(_out_count)
       {
           //--- initialization values and allocating memory
           errorsBias = new float[out_count];
           hiddenNeurons = new float[out_count];
           matrixWeight = new float[(in_count+1) * out_count];
       }

       void InitRandomWeight()
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
               errorsBias[outp] = (unif(rng) - 0.5) * 3 +(unif(rng) - 0.5);// /out_count;
           }


           for(int inp =0; inp < in_count+1; inp++)
           {
               for(int outp =0; outp < out_count; outp++)
               {
                   //getMatrix(inp,outp) =  (((float)rand() / (float)RAND_MAX) - 0.5)/out_count;// * pow(out_count,-0.5);
                   getMatrix(inp,outp) =  (unif(rng) - 0.5) * 3 +(unif(rng) - 0.5);//out_count;// * (rand()%100);// - 0.5);// / (out_count);
               }
           }
       }

       ~ILayerNeurons()
       {
           in_count=out_count=0;
           if(errorsBias) delete[] errorsBias;
           if(hiddenNeurons) delete[] hiddenNeurons;
           if(matrixWeight) delete[] matrixWeight;
       }

       float &getMatrix(int i , int j) {
           return matrixWeight[i*out_count + j];
       }

       unsigned int getInCount(){return in_count;}
       unsigned int getOutCount(){return out_count;}
       float *getMatrixWeight(){return matrixWeight;}
       float* getHidden() { return hiddenNeurons; }
       float* getErrors() { return errorsBias; }

};


#endif // ILAYERNEURONS_H
