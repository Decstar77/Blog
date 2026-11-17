// ProximalPolicyOptim.cpp : Defines the entry point for the application.
//

#include "GridWorld.h"
#include "OneStepActorCritic.h"
#include "Tensor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void print_learning_curve( const GridWorld & env, int bucket ) {
    printf( "Average episode length (shortest path = %d):\n", env.width - 1 + env.height - 1 );
    int n = (int)env.episodeLengths.size();
    for ( int start = 0; start < n; start += bucket ) {
        int end = start + bucket < n ? start + bucket : n;
        long long total = 0;
        for ( int i = start; i < end; i++ ) {
            total += env.episodeLengths[i];
        }
        printf( "  episodes %5d-%5d: %7.2f\n", start, end - 1, (double)total / ( end - start ) );
    }
}

static void print_policy( GridWorld & env, const Tensor & theta ) {
    const char * arrows[GridWorld::ACTION_COUNT] = { "^", "v", "<", ">" };
    printf( "\nGreedy policy (most likely action, with its probability):\n" );
    for ( int row = 0; row < env.height; row++ ) {
        printf( "  " );
        for ( int col = 0; col < env.width; col++ ) {
            int s = row * env.width + col;
            if ( s == env.goal_state() ) {
                printf( "   G     " );
                continue;
            }
            const Tensor & p = env.policy( s, theta );
            int a = tn_argmax( p );
            printf( "   %s %.2f ", arrows[a], p.data[a] );
        }
        printf( "\n" );
    }
}

static void print_values( const GridWorld & env, const Tensor & w, float gamma ) {
    printf( "\nCritic v(s)  vs  optimal v*(s) = -(1 - gamma^d) / (1 - gamma):\n" );
    int goalRow = env.height - 1;
    int goalCol = env.width - 1;
    for ( int row = 0; row < env.height; row++ ) {
        printf( "  " );
        for ( int col = 0; col < env.width; col++ ) {
            int s = row * env.width + col;
            int d = abs( goalRow - row ) + abs( goalCol - col );
            float optimal = -( 1.0f - powf( gamma, (float)d ) ) / ( 1.0f - gamma );
            printf( " %6.2f/%6.2f", tn_at( w, s ), optimal );
        }
        printf( "\n" );
    }
}

int main() {
    const int   episodes = 2000;
    const float alphaTheta = 0.1f;
    const float alphaW = 0.2f;
    const float gamma = 0.99f;

    tn_seed( 42 );

    GridWorld env( 4, 4, 1000 );
    ActorCritResult res = one_step_actor_critic( episodes, alphaTheta, alphaW, gamma, &env );

    print_learning_curve( env, 200 );
    print_policy( env, res.actor );
    print_values( env, res.critic, gamma );

    tn_free( res.actor );
    tn_free( res.critic );
    return 0;
}
