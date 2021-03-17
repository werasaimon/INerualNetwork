#ifndef IGENETIC_H
#define IGENETIC_H

#include "INerualNetwork.h"

class IGenetic
{
public:

    static INerualNetwork* Croosover(const INerualNetwork* a , const INerualNetwork* b , float mutation_rate = 0);
    static void Mutatation(INerualNetwork* a, float MutationChance, float MutationStrength);

};

#endif // IGENETIC_H
