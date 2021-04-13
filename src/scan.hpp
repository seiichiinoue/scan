#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <unordered_set>
#include <cassert>
#include <cmath>
#include <random>
#include <fstream>
#include <vector>
#include "sampler.hpp"
#include "common.hpp"
using namespace std;

namespace scan {
    class SCAN {
    public:
        int _n_k;
        int _n_t;
        double _gamma_a;
        double _gamma_b;
        int _context_window_width;
        
        int _vocab_size;
        int _num_docs;

        // parameters
        double _kappa_phi;
        double _Ekappa_phi;
        double _kappa_psi;
        int* _Z;
        double** _Phi;
        double*** _Psi;
        double** _EPhi;
        double*** _EPsi;

        normal_distribution<double> _normal_distribution_for_phi;
        normal_distribution<double> _normal_distribution_for_psi;

        SCAN() {
            _n_k = NUM_SENSE;
            _n_t = NUM_TIME;
            _gamma_a = GAMMA_A;
            _gamma_b = GAMMA_B;
            _context_window_width = CONTEXT_WINDOW_WIDTH;

            _vocab_size = 0;
            _num_docs = 0;

            _kappa_phi = KAPPA_PHI;
            _kappa_psi = KAPPA_PSI;
            _Ekappa_phi = 0.0;
            _Z = NULL;
            _Phi = NULL;
            _Psi = NULL;
            _EPhi = NULL;
            _EPsi = NULL;
            
            _normal_distribution_for_phi = normal_distribution<double>(0, 1);
            _normal_distribution_for_psi = normal_distribution<double>(0, 1);
        }
        ~SCAN() {}

        void initialize(int vocab_size, int num_docs) {
            _vocab_size = vocab_size;
            _num_docs = num_docs;

            _Z = new int[num_docs];
            _Phi = new double*[_n_t];
            _Psi = new double**[_n_t];
            _EPhi = new double*[_n_t];
            _EPsi = new double**[_n_t];

            for (int n=0; n<_num_docs; ++n) {
                _Z[n] = sampler::uniform_int(0, _n_k-1);
            }
            for (int t=0; t<_n_t; ++t) {
                _Phi[t] = new double[_n_k];
                _EPhi[t] = new double[_n_k];
                for (int k=0; k<_n_k; ++k) {
                    _Phi[t][k] = generate_noise_for_phi_from_normal_distribution();
                    _EPhi[t][k] = _Phi[t][k];
                }
            }
            for (int t=0; t<_n_t; ++t) {
                _Psi[t] = new double*[_n_k];
                _EPsi[t] = new double*[_n_k];
                for (int k=0; k<_n_k; ++k) {
                    _Psi[t][k] = new double[_vocab_size];
                    _EPsi[t][k] = new double[_vocab_size];
                    for (int v=0; v<_vocab_size; ++v) {
                        _Psi[t][k][v] = generate_noise_for_psi_from_normal_distribution();
                        _EPsi[t][k][v] = _Psi[t][k][v];
                    }
                }
            }
        }
        double generate_noise_for_phi_from_normal_distribution() {
            return _normal_distribution_for_phi(sampler::minstd);
        }
        double generate_noise_for_psi_from_normal_distribution() {
            return _normal_distribution_for_psi(sampler::minstd);
        }
        double generate_noise_for_phi_from_truncated_normal_distribution(double lb, double ub) {
            double sampled = generate_noise_for_phi_from_normal_distribution();
            while (sampled <= lb || sampled >= ub) {
                sampled = generate_noise_for_phi_from_normal_distribution();
            }
            return sampled;
        }
        double generate_noise_for_psi_from_truncated_normal_distribution(double lb, double ub) {
            double sampled = generate_noise_for_psi_from_normal_distribution();
            while (sampled <= lb || sampled >= ub) {
                sampled = generate_noise_for_psi_from_normal_distribution();
            }
            return sampled;
        }
        void logisitc_transformation(int t, double* vec, bool evalue=false) {
            double u = 0;
            
            for (int k=0; k<_n_k; ++k) {
                double val;
                if (evalue) {
                    val = std::exp(_EPhi[t][k]);
                } else {
                    val = std::exp(_Phi[t][k]);
                }
                vec[k] = val;
                u += val;
            }
            for (int k=0; k<_n_k; ++k) {
                vec[k] /= u;
            }
        }
        void logisitc_transformation(int t, int k, double* vec, bool evalue=false) {
            double u = 0;
            for (int v=0; v<_vocab_size; ++v) {
                double val;
                if (evalue) {
                    val = std::exp(_EPsi[t][k][v]);
                } else {
                    val = std::exp(_Psi[t][k][v]);
                }
                vec[v] = val;
                u += val;
            }
            for (int v=0; v<_vocab_size; ++v) {
                vec[v] /= u;
            }
        }
        template<class Archive>
        void serialize(Archive &archive, unsigned int version) {
            boost::serialization::split_free(archive, *this, version);
        }
        void save(string filename) {
            std::ofstream ofs(filename);
            boost::archive::binary_oarchive oarchive(ofs);
            oarchive << *this;
        }
        bool load(string filename) {
            std::ifstream ifs(filename);
            if (ifs.good()) {
                boost::archive::binary_iarchive iarchive(ifs);
                iarchive >> *this;
                return true;
            }
            return false;
        }
    };
}
// save model
namespace boost { namespace serialization {
template<class Archive>
    void save(Archive &archive, const scan::SCAN &scan, unsigned int version) {
        archive & scan._n_k;
        archive & scan._n_t;
        archive & scan._gamma_a;
        archive & scan._gamma_b;
        archive & scan._context_window_width;
        archive & scan._vocab_size;
        archive & scan._num_docs;
        archive & scan._kappa_phi;
        archive & scan._kappa_psi;
        archive & scan._Ekappa_phi;
        for (int n=0; n<scan._num_docs; ++n) {
            archive & scan._Z[n];
        }
        for (int t=0; t<scan._n_t; ++t) {
            for (int k=0; k<scan._n_k; ++k) {
                archive & scan._Phi[t][k];
                archive & scan._EPhi[t][k];
            }
        }
        for (int t=0; t<scan._n_t; ++t) {
            for (int k=0; k<scan._n_k; ++k) {
                for (int v=0; v<scan._vocab_size; ++v) {
                    archive & scan._Psi[t][k][v];
                    archive & scan._EPsi[t][k][v];
                }
            }
        }
    }
template<class Archive>
    void load(Archive &archive, scan::SCAN &scan, unsigned int version) {
        archive & scan._n_k;
        archive & scan._n_t;
        archive & scan._gamma_a;
        archive & scan._gamma_b;
        archive & scan._context_window_width;
        archive & scan._vocab_size;
        archive & scan._num_docs;
        archive & scan._kappa_phi;
        archive & scan._kappa_psi;
        archive & scan._Ekappa_phi;
        if (scan._Z == NULL) {
            scan._Z = new int[scan._num_docs];
        }
        if (scan._Phi == NULL) {
            scan._Phi = new double*[scan._n_t];
            for (int t=0; t<scan._n_t; ++t) {
                scan._Phi[t] = new double[scan._n_k];
            }
        }
        if (scan._EPhi == NULL) {
            scan._EPhi = new double*[scan._n_t];
            for (int t=0; t<scan._n_t; ++t) {
                scan._EPhi[t] = new double[scan._n_k];
            }
        }
        if (scan._Psi == NULL) {
            scan._Psi = new double**[scan._n_t];
            for (int t=0; t<scan._n_t; ++t) {
                scan._Psi[t] = new double*[scan._n_k];
                for (int k=0; k<scan._n_k; ++k) {
                    scan._Psi[t][k] = new double[scan._vocab_size];
                }
            }
        }
        if (scan._EPsi == NULL) {
            scan._EPsi = new double**[scan._n_t];
            for (int t=0; t<scan._n_t; ++t) {
                scan._EPsi[t] = new double*[scan._n_k];
                for (int k=0; k<scan._n_k; ++k) {
                    scan._EPsi[t][k] = new double[scan._vocab_size];
                }
            }
        }
        for (int n=0; n<scan._num_docs; ++n) {
            archive & scan._Z[n];
        }
        for (int t=0; t<scan._n_t; ++t) {
            for (int k=0; k<scan._n_k; ++k) {
                archive & scan._Phi[t][k];
                archive & scan._EPhi[t][k];
            }
        }
        for (int t=0; t<scan._n_t; ++t) {
            for (int k=0; k<scan._n_k; ++k) {
                for (int v=0; v<scan._vocab_size; ++v) {
                    archive & scan._Psi[t][k][v];
                    archive & scan._EPsi[t][k][v];
                }
            }
        }
    }
}}  // namespace boost::serialization