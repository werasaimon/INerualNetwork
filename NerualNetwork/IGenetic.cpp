#include "IGenetic.h"

#include <iostream>

IGenetic::IGenetic()
{

}

INerualNetwork* IGenetic::Croosover(const INerualNetwork *a, const INerualNetwork *b, float mutation_rate)
{

    INerualNetwork *cross =  new INerualNetwork(*(a));


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> crossOv(0, 1);

    std::uniform_real_distribution<double> mutation(0.0, 1.0);
    std::normal_distribution<double> mutationVal(0.0, 1.0);


    //----------------------------------------------------//
    std::mt19937 rng;
    // initialize the random number generator with time-dependent seed
    uint32_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>16)};
    rng.seed(ss);
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif(0, 1);
    //----------------------------------------------------//

   // srand(time(NULL));
    for( unsigned int i = 0; i < cross->getCountLayers(); i++ )
    {
        auto al = a->getLayer(i);
        auto bl = b->getLayer(i);
       // ILayerNeurons* crl = cross->getLayer(i);

        // crossover weight matrix
        for( size_t m = 0; m < al->getInCount()+1; m++ )
        {
            for( size_t n = 0; n < al->getOutCount(); n++ )
            {
                if( mutation(rng) <  mutation_rate)
                {
                   // std::cout << "GenMutate" << std::endl;
                    // do mutation
                    cross->getLayer(i)->getMatrix(m, n) = (unif(rng) - 0.5) * 3.0 + (unif(rng) - 0.5);//al->getOutCount();
                    //cross->getLayer(i)->getMatrix(m, n) = mutationVal(gen);
                    //cross->getLayer(i)->getMatrix(m, n) = (((float)rand() / (float)RAND_MAX) - 0.5) * pow(al->getOutCount(),-0.5);//mutationVal(gen);
                }
                else
                {
                    // do crossover
                    if (crossOv(gen) == 0)
                    {
                       // std::cout << "GenA" << std::endl;
                        cross->getLayer(i)->getMatrix(m, n) = al->getMatrix(m, n);
                    }
                    else
                    {
                        //std::cout << "GenB" << std::endl;
                        cross->getLayer(i)->getMatrix(m, n) = bl->getMatrix(m, n);
                    }
                }
            }
        }



        for( size_t m = 0; m < cross->getLayer(i)->getOutCount(); m++ )
        {
            if( mutation(rng) < mutation_rate )
            {
                // do mutation

                  cross->getLayer(i)->getErrors()[m] = (unif(rng) - 0.5) * 3.0 + (unif(rng) - 0.5);//(unif(rng) - 0.5)/al->getOutCount();
                //cross->getLayer(i)->getErrors()[m] = mutationVal(gen);
                //cross->getLayer(i)->getErrors()[m] = (((float)rand() / (float)RAND_MAX) - 0.5);//mutationVal(gen);
            }
            else
            {
                // do crossover
                if( crossOv(gen) == 0 )
                {
                    cross->getLayer(i)->getErrors()[m] = al->getErrors()[m];
                }
                else
                {
                    cross->getLayer(i)->getErrors()[m] = bl->getErrors()[m];
                }
            }
        }
    }

    return cross;

}
