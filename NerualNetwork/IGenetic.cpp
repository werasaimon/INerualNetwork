#include "IGenetic.h"

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

        // crossover weight matrix
        for( unsigned int m = 0; m < al->getInCount()+1; m++ )
        {
            for( unsigned int n = 0; n < al->getOutCount(); n++ )
            {
                if( mutation(rng) <  mutation_rate)
                {
                    // do mutation
                      cross->getLayer(i)->getMatrix(m, n) = (unif(rng) - 0.5);
                }
                else
                {
                    // do crossover
                    if (crossOv(rng) == 0)
                    {
                        cross->getLayer(i)->getMatrix(m, n) = al->getMatrix(m, n);
                    }
                    else
                    {
                        cross->getLayer(i)->getMatrix(m, n) = bl->getMatrix(m, n);
                    }
                }
            }
        }



        for( unsigned int m = 0; m < cross->getLayer(i)->getOutCount(); m++ )
        {
            if( mutation(rng) < mutation_rate )
            {
                // do mutation
                  cross->getLayer(i)->getErrors()[m] = (unif(rng) - 0.5);// * 3.0 + (unif(rng) - 0.5);
            }
            else
            {
                // do crossover
                if( crossOv(rng) == 0 )
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

void IGenetic::Mutatation(INerualNetwork *a, float MutationChance, float MutationStrength)
{

    //----------------------------------------------------//
    std::mt19937 rng;
    // initialize the random number generator with time-dependent seed
    uint32_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>16)};
    rng.seed(ss);
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif(0, MutationStrength);
    std::uniform_real_distribution<double> mutation(0.0, 1.0);
    std::normal_distribution<double> mutationVal(0.0, 1.0);
    //----------------------------------------------------//


   // srand(time(NULL));
    for( unsigned int i = 0; i < a->getCountLayers(); i++ )
    {
        auto l = a->getLayer(i);

        // weight matrix
        for( unsigned int m = 0; m < l->getInCount()+1; m++ )
        {
            for( unsigned int n = 0; n < l->getOutCount(); n++ )
            {
                if( mutation(rng) < MutationChance )
                {
                    // do mutation
                    a->getLayer(i)->getMatrix(m, n) += (unif(rng) - MutationStrength * 0.5);
                }
            }
        }

        for( unsigned int m = 0; m < l->getOutCount(); m++ )
        {
            if( mutation(rng) < MutationChance )
            {
                // do mutation
                a->getLayer(i)->getErrors()[m] += (unif(rng) - MutationStrength * 0.5);
            }
        }
    }
}

