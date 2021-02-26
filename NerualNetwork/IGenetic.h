#ifndef IGENETIC_H
#define IGENETIC_H

#include "INerualNetwork.h"

class IGenetic
{
public:
    IGenetic();


    static INerualNetwork* Croosover(const INerualNetwork* a , const INerualNetwork* b , float mutation_rate);


};

#endif // IGENETIC_H
