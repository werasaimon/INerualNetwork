/********************************************************************************
*
* ILayerNeurons.h
*
* IMath : INeural Network library,
* Copyright (c)  *
* Created on: 3 July. 2020 г.
* Author: werasaimon                                     *
*********************************************************************************
*                                                                               *
* This software is provided 'as-is', without any express or implied warranty.   *
* In no event will the authors be held liable for any damages arising from the  *
* use of this software.                                                         *
*                                                                               *
* Permission is granted to anyone to use this software for any purpose,         *
* including commercial applications, and to alter it and redistribute it        *
* freely, subject to the following restrictions:                                *
*                                                                               *
* 1. The origin of this software must not be misrepresented; you must not claim *
*    that you wrote the original software. If you use this software in a        *
*    product, an acknowledgment in the product documentation would be           *
*    appreciated but is not required.                                           *
*                                                                               *
* 2. Altered source versions must be plainly marked as such, and must not be    *
*    misrepresented as being the original software.                             *
*                                                                               *
* 3. This notice may not be removed or altered from any source distribution.    *
*                                                                               *
********************************************************************************/

#ifndef ILAYERNEURONS_H
#define ILAYERNEURONS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include <chrono>

#include <iostream>

class ILayerNeurons
{
    friend class INeuralNetwork;

public:

    enum LayerOutputType
    {
        Sigmoid = 0x00, // Sigmoid activaton
        Tanh = 0x01,       // Softmax activation
        ReLU = 0x02
    };
private:
        //--- information about input/out width for neuro layer
       int in_count;
       int out_count;
       LayerOutputType  m_layer_type;
       //--- weight matrix
       float* matrixWeight;
       //--- current hidden value array
       float* hiddenNeurons;
       //--- current errors for backPropagate
       float* errorsBias;


public:

       ILayerNeurons(){}
       ILayerNeurons(int _in_count , int _out_count , const LayerOutputType& type = Sigmoid)
        :in_count(_in_count),
         out_count(_out_count),
         m_layer_type(type)
       {
           //--- initialization values and allocating memory
           errorsBias = new float[out_count];
           hiddenNeurons = new float[out_count];
           matrixWeight = new float[(in_count+1) * out_count];
       }

       void InitRandomWeight();

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
       float* getMatrixWeight(){return matrixWeight;}
       float* getHidden() { return hiddenNeurons; }
       float* getErrors() { return errorsBias; }



       /**
        * Serialize the layer (weights, biases).
        * @return string holding binary representation of the layer.
        */
       std::string serialize() const;


       /**
        * Deserialize a binary representation of a layer.
        * @param buffer Binaray data.
        * @return Initialized layer
        */
       static ILayerNeurons* deserialize( const std::string& buffer );


       /**
        * Returns the layer type
        * @return Layer type;
        */
       LayerOutputType getLayer_type() const;


       /**
        * Sets the layer type.
        * @param type Layer type.
        */
       void setLayer_type(const LayerOutputType &layer_type);
};






#endif // ILAYERNEURONS_H
