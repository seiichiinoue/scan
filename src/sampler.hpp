#pragma once
#include <random>
#include <chrono>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

using namespace std;

namespace scan {
    namespace sampler {
        int seed = chrono::system_clock::now().time_since_epoch().count();
        mt19937 mt(seed);
        minstd_rand minstd(seed);

        double gamma(double a, double b) {
            gamma_distribution<double> distribution(a, 1.0 / b);
            return distribution(mt);
        }
        double beta(double a, double b) {
            double ga = gamma(a, 1.0);
            double gb = gamma(b, 1.0);
            return ga / (ga + gb);
        }
        double bernoulli(double p) {
            uniform_real_distribution<double> rand(0, 1);
            double r = rand(mt);
            if (r > p) {
                return 0;
            }
            return 1;
        }
        double uniform(double min=0, double max=0) {
            uniform_real_distribution<double> rand(min, max);
            return rand(mt);
        }
        static double uniform_int(int min=0, int max=0) {
            uniform_int_distribution<> rand(min, max);
            return rand(mt);
        }
        int multinomial(size_t k, double* p) {
            // settings for gsl random number generator
            const gsl_rng_type* T;
            gsl_rng* _r;
            gsl_rng_env_setup();
            T = gsl_rng_default;
            _r = gsl_rng_alloc(T);
            gsl_rng_set(_r, seed);

            unsigned int* n = new unsigned int[k];
            const double* p_const = const_cast<const double*>(p); 
            gsl_ran_multinomial(_r, k, 1, p, n);
            int ret = k-1;
            for (int i=0; i<k; ++i) {
                if (n[i]) ret = i;
            }
            gsl_rng_free(_r);
            return ret;
        }
    }
}