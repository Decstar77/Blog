#pragma once

#include "OneStepActorCritic.h"

#include <vector>

// Tabular grid world for testing the actor-critic.
//
//   S . . .
//   . . . .
//   . . . .
//   . . . G
//
// Start top-left, goal bottom-right, reward -1 per step. Moving into a wall
// leaves the agent in place. Shortest path is (W-1)+(H-1) steps.
//
// Actor:  theta is [states x actions], pi(.|s) = softmax(theta[s])
// Critic: w is [states],               v(s)    = w[s]
class GridWorld : public ActorCriticInterface {
  public:
    static constexpr int ACTION_COUNT = 4;
    enum Action { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

    int width;
    int height;
    int maxSteps;   // safety cap so a bad early policy can't wander forever

    std::vector<int> episodeLengths;

    GridWorld( int width, int height, int maxSteps )
        : width( width ), height( height ), maxSteps( maxSteps ) {
        thetaGrad = tn_create( state_count(), ACTION_COUNT );
        wGrad = tn_create( state_count() );
        probs = tn_create( ACTION_COUNT );
    }

    ~GridWorld() {
        tn_free( thetaGrad );
        tn_free( wGrad );
        tn_free( probs );
    }

    int state_count() const { return width * height; }
    int goal_state() const { return state_count() - 1; }

    // ---- Environment -------------------------------------------------------

    int new_game() override {
        state = 0;
        steps = 0;
        done = false;
        return state;
    }

    bool is_terminal() override { return done; }

    int step( int action, float & reward ) override {
        int row = state / width;
        int col = state % width;
        switch ( action ) {
            case UP:    if ( row > 0 )          row--; break;
            case DOWN:  if ( row < height - 1 ) row++; break;
            case LEFT:  if ( col > 0 )          col--; break;
            case RIGHT: if ( col < width - 1 )  col++; break;
        }
        state = row * width + col;
        steps++;
        reward = -1.0f;

        // Note: hitting maxSteps is a time limit, not a real terminal state, but
        // the interface can't tell them apart so the loop will bootstrap with
        // v = 0. Keep maxSteps large enough that this rarely happens.
        if ( state == goal_state() || steps >= maxSteps ) {
            done = true;
            episodeLengths.push_back( steps );
        }
        return state;
    }

    // ---- Actor -------------------------------------------------------------

    // Fills `probs` with pi(.|s, theta)
    const Tensor & policy( int s, const Tensor & theta ) {
        for ( int a = 0; a < ACTION_COUNT; a++ ) {
            probs.data[a] = tn_at( theta, s, a );
        }
        tn_softmax( probs );
        return probs;
    }

    int next_action( int s, const Tensor & theta ) override {
        return tn_sample_categorical( policy( s, theta ) );
    }

    Tensor initial_theta() override {
        return tn_create( state_count(), ACTION_COUNT ); // zeros = uniform random policy
    }

    // d/dtheta ln pi(a|s) for a tabular softmax: row s = one_hot(a) - pi(.|s), all other rows 0.
    // Returns a reused internal buffer; it is overwritten on every call.
    Tensor theta_log_grad( int s, int a, const Tensor & theta ) override {
        policy( s, theta );
        tn_zero( thetaGrad );
        for ( int b = 0; b < ACTION_COUNT; b++ ) {
            tn_at( thetaGrad, s, b ) = ( b == a ? 1.0f : 0.0f ) - probs.data[b];
        }
        return thetaGrad;
    }

    // ---- Critic ------------------------------------------------------------

    float state_value( int s, const Tensor & w ) override {
        return tn_at( w, s );
    }

    Tensor initial_w() override {
        return tn_create( state_count() );
    }

    // d/dw v(s) = one_hot(s). Reused internal buffer.
    Tensor state_value_grad( int s, const Tensor & w ) override {
        (void)w;
        tn_one_hot( wGrad, s );
        return wGrad;
    }

  private:
    int  state = 0;
    int  steps = 0;
    bool done = true;

    Tensor thetaGrad;
    Tensor wGrad;
    Tensor probs;
};
