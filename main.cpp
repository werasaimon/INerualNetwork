#include <QCoreApplication>
#include <QDebug>
#include <QTime>
#include <QDebug>
#include <QFile>

#include "NerualNetwork/INerualNetwork.h"


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    INerualNetwork *NerualNet = new INerualNetwork({100,20,2});
    {

        //----------------------------------INPUTS----GENERATOR-------------
        // /! создаём 2 случайнозаполненных входных вектора
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
        // создаем 2 цели обучения
        float *tar1 = new float[2];
        tar1[0] =0.01;
        tar1[1] =0.99;
        float *tar2 = new float[2];
        tar2[0] =0.99;
        tar2[1] =0.01;

        //--------------------------------NN---------WORKING---------------
        // первичный опрос сети

        std::cout << "___________________ABC_____________ \n";

        float *out = nullptr;
        out = NerualNet->feedForwarding(abc);
        for (unsigned int i=0; i<NerualNet->getCountNeuronsOutput(); ++i)
        {
            std::cout << "out neuron " << i << " = " << out[i] << std::endl;
        }

        std::cout << "___________________CBA_____________ \n";

        out = NerualNet->feedForwarding(cba);
        for (unsigned int i=0; i<NerualNet->getCountNeuronsOutput(); ++i)
        {
            std::cout << "out neuron " << i << " = " << out[i] << std::endl;
        }

        std::cout << "\n \n ___________________Train_____________ \n";
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
        std::cout << "\n \n ___________________ABC_____________ \n";

        out = nullptr;
        out = NerualNet->feedForwarding(abc);
        for (unsigned int i=0; i<NerualNet->getCountNeuronsOutput(); ++i)
        {
            std::cout << "out neuron " << i << " = " << out[i] << std::endl;
        }

        std::cout << "___________________CBA_____________ \n";

        out = NerualNet->feedForwarding(cba);
        for (unsigned int i=0; i<NerualNet->getCountNeuronsOutput(); ++i)
        {
            std::cout << "out neuron " << i << " = " << out[i] << std::endl;
        }
    }

    return a.exec();
}
