#include "Tensor.h"


class PPOInterface {
    virtual int     new_game() = 0;
    virtual bool    is_terminal() = 0;
    virtual int     step( int action, float & reward ) = 0;
};

