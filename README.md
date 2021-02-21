# INerualNetwork
NerualNetwork Nerual  Network Neuron

simple neuero network with C/C++ and QT Code


![DATASET] https://www.kaggle.com/oddrationale/mnist-in-csv

```cpp
#include <QCoreApplication>
#include <QTime>
#include <iostream>

#include "NerualNetwork/INerualNetwork.h"


#define learningRate 0.4

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    auto sigmoida = [](float val) -> float {
        return (1.0 / (1.0 + exp(-val)));
    }
    ;
    auto sigmoidasDerivate = [](float val) -> float {
        return (val * (1.0 - val));
    };

    INerualNetwork *NerualNet = new INerualNetwork(sigmoida,sigmoidasDerivate,{100,20,6,3,2});


    //----------------------------------INPUTS----GENERATOR-------------
    qsrand((QTime::currentTime().second()));
    float *abc = new float[100];
    for(int i=0; i<100;i++)
    {
        abc[i] =(qrand()%98)*0.01+0.01;
    }

    float *cba = new float[100];
    for(int i=0; i<100;i++)
    {
        cba[i] =(qrand()%98)*0.01+0.01;
    }


    //---------------------------------TARGETS----GENERATOR-------------
    float *target1 = new float[2];
    target1[0] =0.01;
    target1[1] =0.99;
    float *target2 = new float[2];
    target2[0] =0.99;
    target2[1] =0.01;


    //--------------------------------NN---------WORKING---------------
    int i=0;
    while(i<10000)
    {
        NerualNet->backPropagate(abc,target1,learningRate);
        NerualNet->backPropagate(cba,target2,learningRate);
        i++;
    }


    std::cout<<"\n ___________________RESULT ABC_____________ \n";
    float *output_neuron = NerualNet->feedForwarding(abc);
    for (int i = 0; i < NerualNet->getCountNeuronsOutput(); ++i)
    {
       std::cout << "neuron: " << i << "  " << output_neuron[i] << std::endl;
    }


    std::cout<<"\n ___________________RESULT CBA_____________ \n";
    float *output_neuron2 = NerualNet->feedForwarding(cba);
    for (int i = 0; i < NerualNet->getCountNeuronsOutput(); ++i)
    {
       std::cout << "neuron: " << i << "  " << output_neuron2[i] << std::endl;
    }

    delete NerualNet;
    NerualNet = nullptr;

    return a.exec();
}
```
