#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "NerualNetwork/INerualNetwork.h"


int main(int argc, char *argv[])
{
    INerualNetwork *NerualNet = new INerualNetwork({100,20,2});
    {

        //----------------------------------INPUTS----GENERATOR-------------
        // /! создаём 2 случайнозаполненных входных вектора
        srand( time( NULL ));
        float *abc = new float[100];
        for(int i=0; i<100;i++)
        {
            abc[i] =(rand()%98)*0.01+0.01;
        }

        float *cba = new float[100];
        for(int i=0; i<100;i++)
        {
            cba[i] =(rand()%98)*0.01+0.01;
        }

        //---------------------------------TARGETS----GENERATOR-------------
        // создаем 2 цели обучения
        float *tar1 = new float[2];
        tar1[0] =0.01;
        tar1[1] =0.99;
        float *tar2 = new float[2];
        tar2[0] =0.99;
        tar2[1] =0.01;

        //--------------------------------NN---------WORKING---------------
        // первичный опрос сети

        std::cout << "___________________ input: ABC _____________ \n";

        float *out = nullptr;
        out = NerualNet->feedForwarding(abc);
        for (unsigned int i=0; i<NerualNet->getCountNeuronsOutput(); ++i)
        {
            std::cout << "out neuron " << i << " = " << out[i] << std::endl;
        }

        std::cout << "___________________ input: CBA _____________ \n";

        out = NerualNet->feedForwarding(cba);
        for (unsigned int i=0; i<NerualNet->getCountNeuronsOutput(); ++i)
        {
            std::cout << "out neuron " << i << " = " << out[i] << std::endl;
        }

        std::cout << "\n \n ___________________ Train _____________ \n";
        float lerning_rate = 0.2;
        // обучение
        int i=0;
        while(i<100000)
        {
            NerualNet->backPropagate(abc,tar1,lerning_rate);
            NerualNet->backPropagate(cba,tar2,lerning_rate);
            i++;
            if(i%1000 == 0) std::cout << " . ";
        }

        //просмотр результатов обучения (опрос сети второй раз)
        std::cout << "\n \n ___________________ input: ABC _____________ \n";

        out = nullptr;
        out = NerualNet->feedForwarding(abc);
        for (unsigned int i=0; i<NerualNet->getCountNeuronsOutput(); ++i)
        {
            std::cout << "out neuron " << i << " = " << out[i] << std::endl;
        }

        std::cout << "___________________ input: CBA _____________ \n";

        out = NerualNet->feedForwarding(cba);
        for (unsigned int i=0; i<NerualNet->getCountNeuronsOutput(); ++i)
        {
            std::cout << "out neuron " << i << " = " << out[i] << std::endl;
        }
    }

    return 0;
}
